"""
Type definitions and result serialization for the evaluation search agent.
"""

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional


@dataclass
class RerankerOutputs:
    """Outputs from the reranker."""
    output: Optional[str] = None
    reasoning: Optional[str] = None
    pre_rerank_docids: Optional[List[str]] = None
    reranked_docids: Optional[List[str]] = None


@dataclass
class AnswerTurn:
    """A turn that contains the final answer."""
    reasoning: Optional[str] = None
    answer: Optional[str] = None

@dataclass
class ToolTurn:
    """
    Base class for a single tool call turn.
    
    This is kept general to support any tool type. For search-specific
    turns, use SearchTurn which adds additional fields.
    """
    tool_name: str
    reasoning: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    response: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


@dataclass
class SearchTurn(ToolTurn):
    """
    A search tool call turn with additional search-specific fields.
    
    Fields:
        query: The search query (extracted from args for convenience)
        docids: Document IDs returned from search
        augmented_query: The augmented query if query augmentation was applied
        reranker: Outputs from the reranker
    """
    query: Optional[str] = None
    docids: Optional[List[str]] = None
    augmented_query: Optional[str] = None
    retrieved_results: Optional[List[Dict[str, Any]]] = None
    reranker: Optional[RerankerOutputs] = None

@dataclass
class VisitTurn(ToolTurn):
    """
    A visit tool call turn with additional visit-specific fields.
    """
    docids: Optional[List[str]] = None

def build_result_array_from_turns(all_turns: list) -> list:
    """Build result array directly from structured turns.
    
    Args:
        all_turns: List of Turn/SearchTurn dataclass instances from the agent
        
    Returns:
        List of result entries in the save format, with reasoning, tool_call, and output_text items.
    """
    result_array = []
    last_reasoning_output = None

    for turn in all_turns:
        if isinstance(turn, AnswerTurn):
            if turn.reasoning:
                reasoning_output = turn.reasoning.strip()
                result_array.append({
                    "type": "reasoning",
                    "output": reasoning_output
                })
                last_reasoning_output = reasoning_output
            else:
                last_reasoning_output = None
            result_array.append({
                "type": "output_text",
                "output": turn.answer
            })
            continue

        tool_name = turn.tool_name
        reasoning = turn.reasoning
        args = turn.args or {}
        response = turn.response
        success = turn.success
        error = turn.error

        if reasoning:
            reasoning_output = reasoning.strip()
            emit_reasoning = reasoning_output != last_reasoning_output
            last_reasoning_output = reasoning_output
        else:
            emit_reasoning = False
            last_reasoning_output = None

        if emit_reasoning:
            result_array.append({
                "type": "reasoning",
                "output": reasoning_output
            })

        tool_call_entry = {
            "type": "tool_call",
            "tool_name": tool_name,
            "arguments": json.dumps(args),
            "output": response,
            "success": success
        }

        if error is not None:
            tool_call_entry["error"] = error

        if isinstance(turn, SearchTurn):
            docids = turn.docids
            augmented_query = getattr(turn, "augmented_query", None)
            reranker = getattr(turn, "reranker", None)
            rewriter = getattr(turn, "rewriter", None)
            summary = getattr(turn, "summary", None)

            tool_call_entry["docids"] = docids

            if augmented_query is not None:
                tool_call_entry["augmented_query"] = augmented_query

            if reranker:
                tool_call_entry["rerank_output"] = reranker.output
                tool_call_entry["rerank_reasoning"] = reranker.reasoning
                tool_call_entry["pre_rerank_docids"] = reranker.pre_rerank_docids
                tool_call_entry["reranked_docids"] = reranker.reranked_docids

            if rewriter:
                tool_call_entry["rewriter_output"] = rewriter.output
                tool_call_entry["rewriter_reasoning"] = rewriter.reasoning
                tool_call_entry["rewriter_prompt"] = rewriter.prompt
                if rewriter.selected_turns:
                    tool_call_entry["rewriter_selected_turns"] = rewriter.selected_turns

            if summary:
                tool_call_entry["summary"] = summary.text
                tool_call_entry["summarizer_prompt"] = summary.prompt
                tool_call_entry["summarizer_reasoning"] = summary.reasoning

        elif isinstance(turn, VisitTurn):
            docids = turn.docids
            tool_call_entry["docids"] = docids

        result_array.append(tool_call_entry)

    return result_array
