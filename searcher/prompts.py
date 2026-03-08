from typing import Optional

def format_reasoning_query(query: str, thinking: Optional[str] = None) -> str:
    reasoning = thinking if thinking else "Empty"
    return f"Reasoning: {reasoning}\n\nQuery: {query}"
