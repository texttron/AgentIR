from typing import List, Optional, Dict, Any
from evaluation.search_agent.tools.types import SearchTurn, RerankerOutputs, ToolTurn
from searcher.prompts import format_reasoning_query
from datasets import load_dataset
import json

class SearchToolHandler:
    name = "search"

    def __init__(self, searcher, k: int = 5, reasoning_aware: bool = False, prepend_oracle=False, allow_parallel_search=False, url_corpus_path=None, skip_failed_reasonings=True, parallel_use_duplicate_reasonings=True):

        self.searcher = searcher
        self.k = k
        self.reasoning_aware = reasoning_aware
        self.prepend_oracle = prepend_oracle
        self.allow_parallel_search = allow_parallel_search
        self.skip_failed_reasonings = skip_failed_reasonings
        self.parallel_use_duplicate_reasonings = parallel_use_duplicate_reasonings
        self.description = f"Performs a search on a knowledge source: supply a single 'query' string; the tool retrieves the top {self.k} most relevant results."

        self.docid_to_url = None
        if url_corpus_path is not None:
            self._load_docid_to_url(url_corpus_path)
    
    def _load_docid_to_url(self, url_corpus_path: str):
        self.docid_to_url = {}
        if url_corpus_path.endswith('.jsonl'):
            ds = load_dataset('json', data_files=url_corpus_path, split='train')
        else:
            ds = load_dataset(url_corpus_path, split='train')
        for row in ds:
            url = row['url']
            docid = row.get('docid')
            if docid is None:
                docid = row.get('id')
            docid = str(docid)
            self.docid_to_url[docid] = url

    def _format_results(self, results: List[dict], display_query: str):
        formatted = []
        for idx, r in enumerate(results, 1):
            snippet = r['snippet']
            title = ""

            if snippet.startswith("---\ntitle:"):
                lines = snippet.split("\n")
                if len(lines) > 1:
                    title = lines[1].replace("title:", "").strip().strip("\"")

            if not title:
                first_line = snippet.split('\n')[0].strip()
                title = first_line[:50] + "..." if len(first_line) > 50 else first_line

            # Format as [Title]\n{snippet}
            if self.docid_to_url is not None: # if we need urls for visit tool
                url = self.docid_to_url[str(r['docid'])]
                formatted_result = f"{idx}. [{title}]({url})\n{snippet}"
            else:
                formatted_result = f"{idx}. [{title}]\n{snippet}"
            formatted.append(formatted_result)

        return f"A search for '{display_query}' found {len(formatted)} results:\n\n## Web Results\n" + "\n\n".join(formatted)

    def search(self, query: str, **kwargs) -> dict:
        """
        Perform a search and return results.
        
        Args:
            query: Search query string
            **kwargs: Additional arguments including:
                - k: number of top k to retrieve
                - original_query: Display query for transparent query augmentation
                - oracle_question, correct_answer: For oracle reranker
        
        Returns:
            Dict with search results and rerank outputs
        """
        k = kwargs.pop('k', self.k)
        display_query = kwargs.get('original_query') or query
        
        try:
            search_result = self.searcher.search(query, k, store_rerank=True, **kwargs)

            results = search_result.get("results", [])

            if not results:
                return {
                    "response": f"No results found for '{display_query}'. Try with a more general query.",
                    "docids": []
                }

            search_result.update({
                "response": self._format_results(results, display_query),
                "docids": [str(r["docid"]) for r in results if "docid" in r],
            })
            
            return search_result

        except Exception as e:
            return {
                "response": f"Search error for query '{display_query}': {str(e)}",
                "docids": []
            }

    def handle_single(
        self,
        tool_name: str,
        original_query: str,
        current_thinking: Optional[str],
        global_question = None,
        correct_answer = None,
        positives = None,
        prev_failed = False,
    ) -> SearchTurn:
        
        search_turn = SearchTurn(
            tool_name=tool_name,
            reasoning=current_thinking,
            args={"query": original_query},
            query=original_query,
            success=False,
        )

        if self.reasoning_aware:
            query = self._apply_reasoning_augmentation(
                original_query,
                current_thinking,
                prev_failed=prev_failed,
            )
            search_turn.augmented_query = query
        else:
            query = original_query

        call_kwargs = {'oracle_question': global_question, 'original_query': original_query}

        # For oracle templates: pass the correct answer
        if correct_answer is not None:
            call_kwargs['correct_answer'] = correct_answer

        # Prepend oracle positive candidates if enabled
        if self.prepend_oracle and positives:
            call_kwargs['prepend_candidates'] = [pos.copy() for pos in positives]

        search_result = self.search(query, **call_kwargs)
        
        result = search_result.get('response')
        docids = search_result.get('docids')
        results_list = search_result.get('results')

        if search_result.get('rerank_output') is not None:
            search_turn.reranker = RerankerOutputs(
                output=search_result.get('rerank_output'),
                reasoning=search_result.get('rerank_reasoning'),
                pre_rerank_docids=search_result.get('pre_rerank_docids'),
                reranked_docids=search_result.get('reranked_docids')
            )

        search_turn.docids = docids
        search_turn.retrieved_results = results_list
        search_turn.response = result

        if docids is not None:
            search_turn.success = True

        return search_turn

    def handle(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        current_thinking: Optional[str],
        all_turns: List[ToolTurn],
        global_question = None,
        correct_answer = None,
        positives = None,
    ) -> List[SearchTurn]:

        try:
            queries = tool_args["query"]
        except:
            return [SearchTurn(
                tool_name=tool_name,
                reasoning=current_thinking,
                args=dict(tool_args),
                query=None,
                success=False,
                response="[Search] Invalid request format: Input must be a JSON object containing 'query' field"
            )]
        
        if isinstance(queries, str):
            queries = [queries]
        
        if isinstance(queries, list) and len(queries) == 0:
            return [SearchTurn(
                tool_name=tool_name,
                reasoning=current_thinking,
                args=dict(tool_args),
                query=queries,
                success=False,
                response="[Search] Invalid request format: 'query' cannot be empty"
            )]
        
        if not self.allow_parallel_search:
            if not isinstance(queries, list) or len(queries) != 1:
                return [SearchTurn(
                    tool_name=tool_name,
                    reasoning=current_thinking,
                    args=dict(tool_args),
                    query=queries,
                    success=False,
                    response="[Search] Invalid request format: 'query' must be a string, not an array"
                )]
        
        if not isinstance(queries, list):
            return [SearchTurn(
                tool_name=tool_name,
                reasoning=current_thinking,
                args=dict(tool_args),
                query=queries,
                success=False,
                response="[Search] Invalid request format: 'query' must be an array of strings"
            )]
        

        last_turn_failed = self._last_turn_failed(all_turns)

        new_turns = []
        for idx, query in enumerate(queries):
            # For the 1st query, use current_thinking; for subsequent queries, use None (unless parallel_use_duplicate_reasonings is enabled)
            thinking_for_query = current_thinking if (idx == 0 or self.parallel_use_duplicate_reasonings) else None
            
            search_turn = self.handle_single(
                tool_name=tool_name,
                original_query=query,
                current_thinking=thinking_for_query,
                global_question=global_question,
                correct_answer=correct_answer,
                positives=positives,
                prev_failed=last_turn_failed
            )
            
            new_turns.append(search_turn)
        
        return new_turns
    
    def handle_single_with_checks(
        self,
        tool_args: dict[str, Any],
        current_thinking: Optional[str],
        all_turns: List[ToolTurn],
        global_question=None,
        correct_answer=None,
        positives=None
    ) -> SearchTurn:
        last_turn_failed = self._last_turn_failed(all_turns)

        if 'query' not in tool_args:
            return SearchTurn(
                tool_name="search",
                reasoning=current_thinking,
                args=tool_args,
                query=None,
                success=False,
                response="Error: Field 'query' is required"
            )
        if not isinstance(tool_args['query'], str):
            return SearchTurn(
                tool_name="search",
                reasoning=current_thinking,
                args=tool_args,
                query=None,
                success=False,
                response="Error: Field 'query' must be a string"
            )
        
        query = tool_args['query']
        
        return self.handle_single(
            tool_name="search",
            original_query=query,
            current_thinking=current_thinking,
            global_question=global_question,
            correct_answer=correct_answer,
            positives=positives,
            prev_failed=last_turn_failed
        )

    def _apply_reasoning_augmentation(
        self,
        original_query: str,
        current_thinking: Optional[str],
        prev_failed: bool = False,
    ) -> str:
        if not self.reasoning_aware:
            return original_query

        if self.skip_failed_reasonings and prev_failed:
            current_thinking = None

        return format_reasoning_query(original_query, current_thinking)
    
    def _last_turn_failed(self, all_turns: List[ToolTurn]) -> bool:
        for turn in reversed(all_turns):
            if isinstance(turn, ToolTurn):
                return not turn.success
        return False


class OpenAISearchToolHandler(SearchToolHandler):
    def get_tool_definitions(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": self.searcher.search_description(self.k),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query string"}
                        },
                        "required": ["query"]
                    },
                },
            }
        ]
        return tools
    
    def _format_results(self, results: list[dict[str, Any]], display_query: str) -> str:
        formatted = []
        for cand in results:
            if cand.get("score") is None:
                formatted.append({"docid": cand["docid"], "snippet": cand["snippet"]})
            else:
                formatted.append({"docid": cand["docid"], "score": cand["score"], "snippet": cand["snippet"]})

        return json.dumps(formatted, indent=2)
