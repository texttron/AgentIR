import os
import json
import argparse
import csv
import re
import uuid
import threading
import sys
from typing import Any, Optional
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich import print
from zai import ZhipuAiClient
from dotenv import load_dotenv

from searcher.searchers import SearcherType
from evaluation.search_agent.tools.types import (
    ToolTurn,
    AnswerTurn,
    build_result_array_from_turns,
)
from evaluation.search_agent.tools.prompts import QUERY_TEMPLATE
from evaluation.search_agent.tools.tool_search import OpenAISearchToolHandler

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def get_argument_type(func_name, arg_key, tools):
    for t in tools:
        if t['function']['name'] == func_name:
            props = t['function']['parameters']['properties']
            if arg_key in props:
                return props[arg_key].get('type', 'string')
    return 'string'


def parse_tool_calls_from_text(text: str, defined_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_calls = []
    tool_call_strs = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    for call in tool_call_strs:
        func_name_match = re.match(r'([^\n<]+)', call.strip())
        func_name = func_name_match.group(1).strip() if func_name_match else None
        if func_name:
            pairs = re.findall(r'<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>', call, re.DOTALL)
            arguments = {}
            for arg_key, arg_value in pairs:
                arg_key = arg_key.strip()
                arg_value = arg_value.strip()
                arg_type = get_argument_type(func_name, arg_key, defined_tools)
                if arg_type != 'string':
                    try:
                        # Attempt to parse as JSON for non-string types
                        parsed_val = json.loads(arg_value)
                        arg_value = parsed_val
                    except Exception:
                        pass
                arguments[arg_key] = arg_value
                
            tool_calls.append({
                'id': "tool-call-" + str(uuid.uuid4()),
                'type': 'function',
                'function': {
                    'name': func_name,
                    'arguments': json.dumps(arguments)
                }
            })
    return tool_calls


def run_conversation_with_tools(
    client: ZhipuAiClient,
    *,
    query: str,
    model: str,
    max_tokens: int,
    tool_handler: OpenAISearchToolHandler,
    system_prompt: str | None = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_iterations: int = 100,
    parallel_use_duplicate_reasonings: bool = False,
):
    tools = tool_handler.get_tool_definitions()

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    formatted_query = QUERY_TEMPLATE.format(Question=query)
    messages.append({"role": "user", "content": formatted_query})

    cumulative_usage = {
        "prompt_tokens": 0,
        "prompt_tokens_cached": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
    }
    all_turns = []
    finish_reason: Optional[str] = None

    # Treat max_tokens as a global output budget across the entire conversation.
    global_max_tokens = max_tokens

    for iteration_idx in range(max_iterations):
        remaining_tokens = global_max_tokens - cumulative_usage["completion_tokens"]
        if remaining_tokens <= 0:
            print(f"Warning: Reached global max_tokens output budget ({global_max_tokens})")
            break
        create_kwargs = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "max_tokens": remaining_tokens,
            "thinking": {"type": "enabled", "clear_thinking": False},
        }
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if top_p is not None:
            create_kwargs["top_p"] = top_p

        completion = client.chat.completions.create(**create_kwargs)

        
        usage = getattr(completion, "usage", None)
        if usage is not None:
            cumulative_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
            cumulative_usage["completion_tokens"] += getattr(usage, "completion_tokens", 0)
            cumulative_usage["total_tokens"] += getattr(usage, "total_tokens", 0)
            comp_details = getattr(usage, "completion_tokens_details", None)
            if comp_details is not None:
                cumulative_usage["reasoning_tokens"] += getattr(comp_details, "reasoning_tokens", 0) or 0
            
            cached_this = 0
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details is not None and getattr(prompt_details, "cached_tokens", None) is not None:
                try:
                    cached_this = int(getattr(prompt_details, "cached_tokens", 0) or 0)
                except Exception:
                    cached_this = 0
            cumulative_usage["prompt_tokens_cached"] += cached_this
        
        choice = completion.choices[0]
        finish_reason = choice.finish_reason

        assistant_msg = choice.message.model_dump()
        
        current_thinking = None
        parsed_tool_calls = []
        if "reasoning_content" in assistant_msg and assistant_msg["reasoning_content"] is not None:
            reasoning_content = assistant_msg["reasoning_content"]
            current_thinking = reasoning_content
            parsed_tool_calls = parse_tool_calls_from_text(reasoning_content, tools)

            del assistant_msg["reasoning_content"]
        
        # Sometimes tool calls incorrectly in reasoning. Merge them
        if parsed_tool_calls:
            if assistant_msg.get("tool_calls") is None:
                assistant_msg["tool_calls"] = []
            
            # Deduplication based on function name and arguments
            existing_calls = [
                (tc["function"]["name"], tc["function"]["arguments"])
                for tc in assistant_msg["tool_calls"]
            ]
            for ptc in parsed_tool_calls:
                if (ptc["function"]["name"], ptc["function"]["arguments"]) not in existing_calls:
                    assistant_msg["tool_calls"].append(ptc)
        
        messages.append(assistant_msg)

        # Terminate if there are no tool calls
        current_tool_calls = assistant_msg.get("tool_calls") or []

        if assistant_msg['content']:
            all_turns.append(AnswerTurn(
                reasoning=current_thinking if not current_tool_calls else None,
                answer=assistant_msg["content"],
            ))

        if not current_tool_calls:
            break

        for idx, tool_call in enumerate(current_tool_calls):
            tname = tool_call["function"]["name"]
            targs_str = tool_call["function"]["arguments"]

            thinking_for_turn = current_thinking if (idx == 0 or parallel_use_duplicate_reasonings) else None

            try:
                message = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tname,
                    "content": None,
                }

                cur_turn = None

                targs = json.loads(targs_str)

                if tname == "search":
                    cur_turn = tool_handler.handle_single_with_checks(targs, thinking_for_turn, all_turns, global_question=query)
                else:
                    # Handle other tools with generic Turn
                    cur_turn = ToolTurn(
                        tool_name=tname,
                        reasoning=thinking_for_turn,
                        args=targs,
                        success=False,
                        response=f"Error: Tool {tname} not found",
                    )
                
                all_turns.append(cur_turn)
                message["content"] = cur_turn.response
                messages.append(message)
                    
            except Exception as e:
                error_msg = f"Error executing {tname}: {str(e)}"
                print(error_msg)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tname,
                    "content": error_msg,
                })
                
                all_turns.append(ToolTurn(
                    tool_name=tname,
                    reasoning=thinking_for_turn,
                    args=str(targs),
                    success=False,
                    error=error_msg,
                ))

    if finish_reason is None:
        print(f"Warning: Conversation hit max iterations ({max_iterations}) without final response")

    return all_turns, cumulative_usage, finish_reason, iteration_idx


def _persist_response(out_dir: str, *, model: str, query_id: str | None, system_prompt: str | None, max_tokens: int, all_turns: list, cumulative_usage: dict, finish_reason: Optional[str], num_iterations: int, args):
    os.makedirs(out_dir, exist_ok=True)

    tool_call_counts: dict[str, int] = {}
    tool_call_counts_all: dict[str, int] = {}
    retrieved_docids = set()
    for turn in all_turns:
        if isinstance(turn, ToolTurn):
            tool_call_counts_all[turn.tool_name] = tool_call_counts_all.get(turn.tool_name, 0) + 1
            if turn.success:
                tool_call_counts[turn.tool_name] = tool_call_counts.get(turn.tool_name, 0) + 1
            if hasattr(turn, "docids"):
                retrieved_docids.update(turn.docids)

    normalized_usage = {
        "input_tokens": cumulative_usage.get("prompt_tokens", 0),
        "input_tokens_cached": cumulative_usage.get("prompt_tokens_cached", 0),
        "output_tokens": cumulative_usage.get("completion_tokens", 0) + cumulative_usage.get("reasoning_tokens", 0),
        "included_reasoning_tokens": cumulative_usage.get("reasoning_tokens", 0),
        "total_tokens": cumulative_usage.get("total_tokens", 0),
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    filename = os.path.join(out_dir, f"run_{ts}.json")

    status = finish_reason
    if status == "stop":
        status = "completed"

    agent_params = {
        "model": model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": max_tokens,
        "output_dir": out_dir
    }

    searcher_params = {
        "searcher_type": args.searcher_type,
        "reasoning_aware": args.reasoning_aware,
    }
    searcher_specific_params = [
        'index_path', 'model_name', 'normalize', 'max_length'
    ]
    for param in searcher_specific_params:
        if hasattr(args, param):
            searcher_params[param] = getattr(args, param)

    reranker_params = {}
    reranker_specific_params = [
        'reranker_type', 'reranker_model', 'reranker_temperature',
        'reranker_top_p', 'reranker_top_k', 'reranker_max_ranking_output_tokens',
        'reranker_enable_thinking', 'reranker_model_url', 'reranker_top',
        'prompt_template', 'QA_jsonl', 'prepend_oracle'
    ]
    for param in reranker_specific_params:
        if hasattr(args, param):
            reranker_params[param] = getattr(args, param)
    searcher_params["reranker"] = reranker_params

    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "agent": agent_params,
            "searcher": searcher_params,
            "query_id": query_id,
            "tool_call_counts": tool_call_counts,
            "tool_call_counts_raw": tool_call_counts_all,
            "total_rounds": num_iterations,
            "usage": normalized_usage,
            "status": status,
            "retrieved_docids": list(retrieved_docids),
            "result": build_result_array_from_turns(all_turns),
        }, f, indent=2, default=str)

    print("Saved response to", filename, "| tool call counts:", tool_call_counts)


def _process_tsv_dataset(tsv_path: str, client: ZhipuAiClient, args, tool_handler: OpenAISearchToolHandler):
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

    def _handle_single_query(qid: str, qtext: str, pbar=None):
        try:
            all_turns, cumulative_usage, finish_reason, num_iterations = run_conversation_with_tools(
                client,
                query=qtext,
                model=args.model,
                max_tokens=args.max_tokens,
                tool_handler=tool_handler,
                system_prompt=args.system,
                temperature=args.temperature,
                top_p=args.top_p,
                max_iterations=args.max_iterations,
                parallel_use_duplicate_reasonings=not args.no_parallel_duplicate_reasonings,
            )

            with completed_lock:
                completed_count[0] += 1
                if pbar:
                    pbar.set_postfix(completed=completed_count[0])

            _persist_response(
                args.output_dir,
                model=args.model,
                query_id=qid,
                system_prompt=args.system,
                max_tokens=args.max_tokens,
                all_turns=all_turns,
                cumulative_usage=cumulative_usage,
                finish_reason=finish_reason,
                num_iterations=num_iterations,
                args=args,
            )
        except Exception as exc:
            print(f"[Error] Query id={qid} failed: {exc}")
            sys.exit(1)

    if args.num_threads <= 1:
        with tqdm(remaining, desc="Queries", unit="query") as pbar:
            for qid, qtext in pbar:
                _handle_single_query(qid, qtext, pbar)
    else:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor, \
            tqdm(total=len(remaining), desc="Queries", unit="query") as pbar:
            futures = [executor.submit(_handle_single_query, qid, qtext, pbar) for qid, qtext in remaining]
            for _ in as_completed(futures):
                pbar.update(1)


def main():
    parser = argparse.ArgumentParser(description="Call GLM Client.")
    parser.add_argument("--query", default="topics-qrels/queries.tsv", help="User query text or path to TSV. Wrap in quotes if contains spaces.")
    parser.add_argument("--model", default="glm-4.7", help="Model name (default: %(default)s)")
    parser.add_argument("--max_tokens", type=int, default=20000, help="Max tokens to generate (default: %(default)s)")
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument("--output-dir", default="runs/agentir/glm_4_7", help="Directory to store logs (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for the model (default: use model defaults)")
    parser.add_argument("--top_p", type=float, default=None, help="Top P for the model (default: use model defaults)")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of parallel threads for dataset processing (default: %(default)s)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of conversation rounds with function calls (default: %(default)s)",
    )
    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        required=True,
        help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}",
    )

    # Server configuration arguments for search tool
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Fixed number of search results to return for all queries in this session (default: 5).",
    )
    parser.add_argument(
        "--reasoning-aware",
        action="store_true",
        help="Prepend the current reasoning to each search query as 'Reasoning: ... Query: ...'.",
    )
    parser.add_argument(
        "--no-parallel-duplicate-reasonings",
        action="store_true",
        help="Use only the first reasoning for parallel search queries instead of duplicating it across all queries.",
    )
    parser.add_argument("--allow-failed-reasonings", action="store_true", help="Do not skip reasonings from failed turns")

    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)
    
    args = parser.parse_args()
    
    if args.reasoning_aware:
        print("Using reasoning-aware retrieval")

    api_key = os.getenv("ZAI_API_KEY")
    if not api_key:
        raise RuntimeError("ZAI_API_KEY is not set in environment")

    client = ZhipuAiClient(api_key=api_key)

    searcher = searcher_class(args)

    tool_handler = OpenAISearchToolHandler(
        searcher=searcher,
        k=args.k,
        reasoning_aware=args.reasoning_aware,
        parallel_use_duplicate_reasonings=not args.no_parallel_duplicate_reasonings,
        skip_failed_reasonings=not args.allow_failed_reasonings,
    )

    # If --query looks like a TSV path, process dataset
    if isinstance(args.query, str):
        qstr = args.query.strip()
        if qstr.lower().endswith(".tsv"):
            potential_path = Path(qstr)
            try:
                if potential_path.is_file():
                    _process_tsv_dataset(str(potential_path), client, args, tool_handler)
                    return
            except OSError:
                pass

    normalized_results, cumulative_usage, finish_reason, num_iterations = run_conversation_with_tools(
        client,
        query=args.query,
        model=args.model,
        max_tokens=args.max_tokens,
        tool_handler=tool_handler,
        system_prompt=args.system,
        temperature=args.temperature,
        top_p=args.top_p,
        max_iterations=args.max_iterations,
        parallel_use_duplicate_reasonings=not args.no_parallel_duplicate_reasonings,
    )

    _persist_response(
        args.output_dir,
        model=args.model,
        query_id=None,
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        all_turns=normalized_results,
        cumulative_usage=cumulative_usage,
        finish_reason=finish_reason,
        num_iterations=num_iterations,
        args=args,
    )

    # Print final output text if present
    final_texts = [item["output"] for item in normalized_results if item.get("type") == "output_text"]
    if final_texts:
        print(final_texts[-1])


if __name__ == "__main__":
    main()
