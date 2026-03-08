from __future__ import annotations

import os
import json
import argparse
import openai
import csv
import threading
import sys
from rich import print
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from evaluation.search_agent.tools.prompts import QUERY_TEMPLATE
from pathlib import Path
from datetime import datetime

from searcher.searchers import SearcherType
from evaluation.search_agent.tools.types import (
    AnswerTurn,
    ToolTurn,
    SearchTurn,
    build_result_array_from_turns,
)
from evaluation.search_agent.tools.tool_search import OpenAISearchToolHandler

class OSSSearchToolHandler(OpenAISearchToolHandler):
    def get_tool_definitions(self):
        tools = [
            {
                "type": "function",
                "name": "search",
                "description": self.searcher.search_description(self.k),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string",
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]
        return tools


def run_conversation_with_tools(client: openai.OpenAI, initial_request: dict, tool_handler: OSSSearchToolHandler, max_iterations: int = 100, global_question=None, correct_answer=None, positives=None):

    all_turns = []
    messages = initial_request['input']
    iteration = 1
    
    while iteration <= max_iterations:
        try:
            request = initial_request.copy()
            request['input'] = messages
            response = client.responses.create(
                **request,
            )
        except Exception as e:
            iteration += 1
            continue
        
        response_dict = response.model_dump(mode="python")
        
        if len(response_dict['output']) >= 1 and response_dict['output'][-1]['type'] == 'reasoning':
            # upstream vLLM bug, parsing tool calls into reasoning channel. Rare to encounter in higher vLLM versions.
            continue
                
        for i in range(len(response_dict['output'])):
            if response_dict['output'][i]['type'] == 'mcp_call':
                # upstream vLLM bug outputs mcp_call but only accepts function_call as input
                # Convert mcp_call to function_call format
                response_dict['output'][i] = {
                    'id': response_dict['output'][i]['id'],
                    'call_id': response_dict['output'][i]['id'],
                    'arguments': response_dict['output'][i]['arguments'],
                    'name': response_dict['output'][i]['name'],
                    'type': 'function_call',
                    'status': None
                }
            messages.append(response_dict['output'][i])
    
        cur_reasonings = [item for item in response_dict['output'] if item['type'] == 'reasoning']
        cur_texts = [item for item in response_dict['output'] if item['type'] == 'message']
        
        cur_reasoning = None
        if cur_reasonings:
            turn_reasonings = [' '.join([c['text'] for c in item['content']]) for item in cur_reasonings]
            cur_reasoning = '\n'.join(turn_reasonings)
        
        cur_text = None
        if cur_texts:
            parts = cur_texts[0]['content']
            cur_text = '\n'.join([part['text']for part in parts])
        
        function_calls = [item for item in response_dict['output'] if item['type'] in ['function_call', 'mcp_call', 'custom_tool_call']]

        if cur_texts:
            all_turns.append(AnswerTurn(
                reasoning=cur_reasoning if not function_calls else None,
                answer=cur_text,
            ))
        
        if not function_calls:
            if messages[-1]['type'] != 'message':
                continue
            return messages, all_turns, "completed"
        
        new_messages = messages.copy()

        for idx, tool_call in enumerate(function_calls):
            try:
                arguments = json.loads(tool_call['arguments'])

                thinking_for_turn = cur_reasoning if (idx == 0 or tool_handler.parallel_use_duplicate_reasonings) else None

                cur_turn = None

                if tool_call['name'] == 'search':
                    cur_turn = tool_handler.handle_single_with_checks(
                        arguments, 
                        thinking_for_turn, 
                        all_turns,
                        global_question=global_question,
                        correct_answer=correct_answer,
                        positives=positives
                    )
                else:
                    cur_turn = ToolTurn(
                        tool_name=tool_call['name'],
                        reasoning=thinking_for_turn,
                        args=str(arguments),
                        success=False,
                        response=f"Error: Tool {tool_call['name']} not found",
                    )
                
                all_turns.append(cur_turn)
                
                new_messages.append({
                    "type": "function_call_output",
                    "call_id": tool_call['call_id'],
                    "output": cur_turn.response
                })
                
            except Exception as e:
                error_msg = f"Error executing {tool_call['name']}: {str(e)}"
                new_messages.append({
                    "type": "function_call_output",
                    "call_id": tool_call['call_id'],
                    "output": error_msg
                })
                all_turns.append(ToolTurn(
                    tool_name=tool_call['name'],
                    args=str(arguments),
                    response=error_msg,
                    success=False,
                    error=error_msg,
                ))
        messages = new_messages
        iteration += 1
    
    print("max iterations reached, incomplete")
    return messages, all_turns, "incomplete"

def _persist_response(out_dir: str, initial_request: dict, messages: list, all_turns: list, status: str, *, query_id: str | None = None, args=None):
    os.makedirs(out_dir, exist_ok=True)

    normalized_results = build_result_array_from_turns(all_turns)

    tool_call_counts: dict[str, int] = {}
    tool_call_counts_all: dict[str, int] = {}
    for turn in all_turns:
        if isinstance(turn, ToolTurn):
            tool_call_counts_all[turn.tool_name] = tool_call_counts_all.get(turn.tool_name, 0) + 1
            if turn.success:
                tool_call_counts[turn.tool_name] = tool_call_counts.get(turn.tool_name, 0) + 1

    retrieved_docids = set()
    for turn in all_turns:
        if isinstance(turn, SearchTurn):
            retrieved_docids.update(turn.docids)

    agent_params = {
        "model": initial_request.get("model"),
        "reasoning": initial_request.get("reasoning"),
        "max_output_tokens": initial_request.get("max_output_tokens"),
    }
    if args is not None:
        agent_params["output_dir"] = str(out_dir)

    searcher_params = {}
    if args is not None:
        searcher_params["searcher_type"] = args.searcher_type
        searcher_params["reasoning_aware"] = args.reasoning_aware
        searcher_specific_params = [
            'index_path', 'model_name', 'normalize', 'max_length'
        ]
        for param in searcher_specific_params:
            if hasattr(args, param):
                searcher_params[param] = getattr(args, param)

        if hasattr(args, 'k'):
            searcher_params["k"] = args.k

    reranker_params = {}
    if args is not None:
        reranker_specific_params = [
            'reranker_type', 'reranker_model', 'reranker_temperature',
            'reranker_top_p', 'reranker_top_k', 'reranker_max_ranking_output_tokens',
            'reranker_enable_thinking', 'reranker_top',
            'prompt_template', 'QA_jsonl', 'prepend_oracle'
        ]
        for param in reranker_specific_params:
            if hasattr(args, param):
                reranker_params[param] = getattr(args, param)
        searcher_params["reranker"] = reranker_params


    normalized_record = {
        "agent": agent_params,
        "searcher": searcher_params,
        "query_id": query_id,
        "tool_call_counts": tool_call_counts,
        "tool_call_counts_raw": tool_call_counts_all,
        "status": status,
        "retrieved_docids": list(retrieved_docids),
        "result": normalized_results,
        "raw_messages": messages
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    filename = os.path.join(str(out_dir), f"run_{ts}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(normalized_record, f, indent=2, default=str)

    print("Saved response to", filename, "| tool call counts:", tool_call_counts)

def _process_tsv_dataset(tsv_path: str, client: openai.OpenAI, args, tool_handler: OSSSearchToolHandler, qa_dict: dict = None):
    """Process a TSV file of (id \t query) pairs sequentially and persist responses."""
    
    dataset_path = Path(tsv_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")
    
    out_dir = Path(args.output_dir).expanduser().resolve()
    
    queries = []
    with dataset_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            queries.append((row[0].strip(), row[1].strip()))
    
    processed_ids = set()
    if out_dir.exists():
        for json_path in out_dir.glob("run_*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    qid_saved = meta.get("query_id")
                    if qid_saved:
                        processed_ids.add(str(qid_saved))
            except Exception:
                continue
    
    remaining = [(qid, qtext) for qid, qtext in queries if qid not in processed_ids]
    
    print(
        f"Processing {len(remaining)} remaining queries (skipping {len(processed_ids)}) from {dataset_path} ..."
    )
    
    completed_lock = threading.Lock()
    completed_count = [0]
    
    def _handle_single_query(qid: str, qtext: str, pbar=None, qa_dict: dict = None):
        """Build request, send and persist response for one query."""

        correct_answer = ""
        positives = []
        if qa_dict is not None and qid in qa_dict:
            qa_entry = qa_dict[qid]
            correct_answer = qa_entry['answer']
            positives = qa_entry.get('positives', [])

        initial_request = {
            "model": args.model,
            "max_output_tokens": args.max_tokens,
            "input": [
                {"role": "user", "content": QUERY_TEMPLATE.format(Question=qtext)}
            ],
            "tools": tool_handler.get_tool_definitions(),
            "truncation": "auto",
            "reasoning": {
                "effort": args.reasoning_effort,
                "summary": "detailed"
            }
        }

        try:
            messages, all_turns, status = run_conversation_with_tools(
                client, initial_request, tool_handler, args.max_iterations,
                global_question=qtext, correct_answer=correct_answer, positives=positives
            )

            if status == "completed":
                with completed_lock:
                    completed_count[0] += 1
                    if pbar:
                        pbar.set_postfix(completed=completed_count[0])

            _persist_response(out_dir, initial_request, messages, all_turns, status, query_id=qid, args=args)

        except Exception as exc:
            print(f"[Error] Query id={qid} failed: {exc}")
            sys.exit(1)
    
    
    if args.num_threads <= 1:
        with tqdm(remaining, desc="Queries", unit="query") as pbar:
            for qid, qtext in pbar:
                _handle_single_query(qid, qtext, pbar, qa_dict)
    else:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor, \
            tqdm(total=len(remaining), desc="Queries", unit="query") as pbar:
            futures = [executor.submit(_handle_single_query, qid, qtext, pbar, qa_dict) for qid, qtext in remaining]
            
            for _ in as_completed(futures):
                pbar.update(1)

def main():
    parser = argparse.ArgumentParser(
        description="Call vLLM OpenAI Responses API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", default="topics-qrels/queries.tsv", help="User query *or* TSV file path")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model name served by vLLM")
    parser.add_argument("--max-tokens", type=int, default=20000, help="max_output_tokens for Responses API")
    parser.add_argument("--output-dir", default="runs/agentir/oss_120b", help="Directory to store run JSON files")
    parser.add_argument("--num-threads", type=int, default=1, help="Parallel threads for dataset mode")
    parser.add_argument("--max-iterations", type=int, default=100, help="Max conversation rounds with function calls")
    parser.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"], help="Reasoning effort")
    parser.add_argument("--model-url", default="http://localhost:8000/v1", help="Model URL")

    # Searcher selection and shared tool options
    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        required=True,
        help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}",
    )

    parser.add_argument("--k", type=int, default=5, help="Top-k search results to return for each query")
    parser.add_argument(
        "--reasoning-aware",
        action="store_true",
        help="Prepend the current reasoning to each search query as 'Reasoning: ... Query: ...'.",
    )
    parser.add_argument("--allow-failed-reasonings", action="store_true", help="Do not skip reasonings from failed turns")


    # QA jsonl for oracle reranker
    parser.add_argument("--QA-jsonl", type=str, default=None, help="Path to QA jsonl file with query_id, query, and answer fields")
    parser.add_argument("--prepend-oracle", action="store_true", help="Prepend positive docs from QA-jsonl to reranking candidates")

    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)
    
    args = parser.parse_args()

    if args.prepend_oracle:
        if args.QA_jsonl is None:
            parser.error("--prepend-oracle requires --QA-jsonl to be set")
        if not hasattr(args, 'reranker_type') or args.reranker_type is None:
            parser.error("--prepend-oracle requires --reranker-type to be set")
        if not args.query.strip().lower().endswith(".tsv"):
            parser.error("--prepend-oracle can only be used with TSV query files, not single queries")
    
    if args.reasoning_aware:
        print("Using reasoning-aware retrieval")

    qa_dict = None
    if args.QA_jsonl:
        qa_dict = {}
        qa_path = Path(args.QA_jsonl)
        if not qa_path.is_file():
            parser.error(f"QA jsonl file not found: {args.QA_jsonl}")

        with qa_path.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    qa_entry = json.loads(line)
                    query_id = qa_entry['query_id']
                    answer = qa_entry['answer']
                    positives = qa_entry.get('positives', [])
                    qa_dict[query_id] = {'answer': answer, 'positives': positives}
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"Error loading QA entry: {e}", qa_entry.get('query_id', 'unknown'))
                    continue

        print(f"Loaded {len(qa_dict)} QA pairs from {args.QA_jsonl}")
        if args.prepend_oracle:
            total_positives = sum(len(v['positives']) for v in qa_dict.values())
            print(f"Loaded {total_positives} total positive documents for oracle prepending")

    client = openai.OpenAI(
        base_url=args.model_url,
        api_key="EMPTY",
    )

    searcher = searcher_class(args)

    tool_handler = OSSSearchToolHandler(
        searcher=searcher,
        k=args.k,
        reasoning_aware=args.reasoning_aware,
        prepend_oracle=args.prepend_oracle,
        skip_failed_reasonings=not args.allow_failed_reasonings
    )
    
    if isinstance(args.query, str):
        qstr = args.query.strip()
        if qstr.lower().endswith(".tsv"):
            potential_path = Path(qstr)
            try:
                if potential_path.is_file():
                    print("Processing TSV dataset", potential_path)
                    _process_tsv_dataset(str(potential_path), client, args, tool_handler, qa_dict)
                    return
            except OSError:
                pass
    
    print("Processing single query", args.query)
    
    args.query = QUERY_TEMPLATE.format(Question=args.query)
    
    messages = [
        {
            "role": "user",
            "content": args.query
        }
    ]
    
    initial_request = {
        "model": args.model,
        "max_output_tokens": args.max_tokens,
        "input": messages,
        "tools": tool_handler.get_tool_definitions(),
        "truncation": "auto",
        "reasoning": {
            "effort": args.reasoning_effort,
            "summary": "detailed"
        }
    }
    
    messages, all_turns, status = run_conversation_with_tools(client, initial_request, tool_handler, args.max_iterations)

    _persist_response(args.output_dir, initial_request, messages, all_turns, status, query_id=None, args=args)
    
    print(messages)
    
if __name__ == "__main__":
    main()
