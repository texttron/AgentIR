"""
Searchers package for different search implementations.
"""

from enum import Enum
from importlib import import_module
from .base import BaseSearcher

class SearcherType(Enum):
    """Enum for managing available searcher types and their CLI mappings."""
    BM25 = (
        "bm25",
        "searcher.searchers.bm25_searcher",
        "BM25Searcher",
        "Install the optional BM25 dependencies with `uv sync --extra bm25`.",
    )
    FAISS = ("faiss", "searcher.searchers.faiss_searcher", "FaissSearcher", None)
    REASONIR = ("reasonir", "searcher.searchers.faiss_searcher", "ReasonIrSearcher", None)
    
    def __init__(self, cli_name, module_name, class_name, install_hint):
        self.cli_name = cli_name
        self.module_name = module_name
        self.class_name = class_name
        self.install_hint = install_hint

    def load_searcher_class(self):
        """Import and return the searcher class on demand."""
        try:
            module = import_module(self.module_name)
        except ImportError as exc:
            if self.install_hint is not None:
                raise ImportError(
                    f"The '{self.cli_name}' searcher requires optional dependencies. "
                    f"{self.install_hint}"
                ) from exc
            raise
        return getattr(module, self.class_name)
    
    @classmethod
    def get_choices(cls):
        """Get list of CLI choices for argument parser."""
        return [searcher_type.cli_name for searcher_type in cls]
    
    @classmethod
    def get_searcher_class(cls, cli_name):
        """Get searcher class by CLI name."""
        for searcher_type in cls:
            if searcher_type.cli_name == cli_name:
                return searcher_type.load_searcher_class()
        raise ValueError(f"Unknown searcher type: {cli_name}")


__all__ = [
    "BaseSearcher",
    "SearcherType"
]
