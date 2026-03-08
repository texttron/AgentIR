import argparse
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseReranker(ABC):
    """Abstract base class for all rerank implementations."""

    def __init__(self, args):
        """Initialize the base reranker.

        Args:
            args: Parsed arguments containing reranker configuration
        """
        self.topk = args.reranker_top
        self.reranker_model = args.reranker_model

    @classmethod
    @abstractmethod
    def parse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add reranker-specific arguments to the argument parser."""
        pass

    @abstractmethod
    def rerank(self, candidates: List[Dict[str, Any]], query: str, k: int = 10, content_key: str = "text", **kwargs) -> List[Dict[str, Any]]:
        """Rerank a list of candidates for a given query.
        
        Args:
            candidates: List of candidate dictionaries to rerank
            query: The query string to rank candidates against
            k: Number of top k results to return after reranking
            content_key: The key in each candidate dict that contains the text content to rank (default: "text")
            **kwargs: Additional reranker-specific arguments (e.g., return_raw)
            
        Returns:
            List of reranked candidate dictionaries (top k)
        """
        pass