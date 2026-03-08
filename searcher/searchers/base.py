"""
Abstract base class for search implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from searcher.rerankers import ListwiseReranker, create_reranker
from transformers import AutoTokenizer
import argparse
import threading


class BaseSearcher(ABC):
    """Abstract base class for all search implementations."""
    
    @classmethod
    def parse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add searcher-specific arguments to the argument parser."""
        parser.add_argument(
            '--reranker-type',
            type=str,
            default=None,
            choices=['listwise'],
            help='Type of reranker to use (default: None, no reranking)'
        )
        
        parser.add_argument(
            '--snippet-max-tokens',
            type=int,
            default=512,
            help='Number of tokens to include for each document snippet in search results using Qwen/Qwen3-0.6B tokenizer. Set to None to disable truncation (default: 512).'
        )
        
        # Parse known args to check if reranker is specified
        args, _ = parser.parse_known_args()
        
        # If reranker type is specified, add reranker-specific arguments
        if args.reranker_type is not None:
            if args.reranker_type == 'listwise':
                ListwiseReranker.parse_args(parser)
        
        # Call subclass-specific argument parsing
        cls._parse_searcher_args(parser)
    
    @classmethod
    @abstractmethod
    def _parse_searcher_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add searcher-specific arguments to the argument parser. Subclasses should implement this."""
        pass
    
    def __init__(self, args):
        """Initialize the searcher with parsed arguments."""
        # Initialize reranker if specified
        self.reranker = create_reranker(args)
        
        self.snippet_max_tokens = getattr(args, 'snippet_max_tokens', None)
        self.tokenizer = None
        self.tokenizer_lock = threading.Lock()  # Thread safety for tokenizer
        if self.snippet_max_tokens and self.snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        
        # Call subclass initialization
        self._init_searcher(args)
    
    @abstractmethod
    def _init_searcher(self, args):
        """Initialize searcher-specific components. Subclasses should implement this."""
        pass
    
    def search(self, query: str, k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Perform search and return results, with optional truncation and reranking.

        Args:
            query: Search query string
            k: Number of results to return
            **kwargs: Additional arguments for reranking (e.g., original_query, correct_answer, store_rerank)

        Returns:
            List of search results with format: {"docid": str, "score": float, "text": str, "snippet": str}
            OR dict with {"results": [...], "rerank_output": str} if store_rerank=True
        """
        # Extract store_rerank flag from kwargs (don't pop it, let reranker see it too)
        store_rerank = kwargs.get('store_rerank', False)

        results = self._retrieve(query, k)

        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer and results:
            with self.tokenizer_lock:
                for result in results:
                    text = result["text"]
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    if len(tokens) > self.snippet_max_tokens:
                        truncated_tokens = tokens[:self.snippet_max_tokens]
                        result["snippet"] = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                    else:
                        result["snippet"] = text
        else:
            for result in results:
                result["snippet"] = result["text"]

        # Handle prepend_candidates if present (for oracle reranking)
        prepend_candidates = kwargs.get('prepend_candidates', [])
        if prepend_candidates:
            prepend_docids_list = [str(pc['docid']) for pc in prepend_candidates]
            assert len(prepend_docids_list) == len(set(prepend_docids_list)), \
                f"prepend_candidates must have unique docids, found duplicates: {prepend_docids_list}"
            
            if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
                with self.tokenizer_lock:
                    for candidate in prepend_candidates:
                        text = candidate["text"]
                        tokens = self.tokenizer.encode(text, add_special_tokens=False)
                        if len(tokens) > self.snippet_max_tokens:
                            truncated_tokens = tokens[:self.snippet_max_tokens]
                            candidate["snippet"] = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                        else:
                            candidate["snippet"] = text
            else:
                for candidate in prepend_candidates:
                    candidate["snippet"] = candidate["text"]
            
            # Deduplicate: remove any results that are already in prepend_candidates
            m = len(prepend_candidates)
            original_k = len(results)  # Save original k before deduplication
            prepend_docids = set(str(pc['docid']) for pc in prepend_candidates)
            
            deduped_results = []
            for result in results:
                if str(result['docid']) not in prepend_docids:
                    deduped_results.append(result)
            
            # Prepend oracle candidates and take first (k - m) from deduped results to maintain window size k
            num_to_take = original_k - m
            if len(deduped_results) >= num_to_take:
                results = prepend_candidates + deduped_results[:num_to_take]
            else:
                results = prepend_candidates + deduped_results

        # Apply reranking if reranker is available
        if self.reranker is not None and results:
            # Store pre-reranking docids if store_rerank is enabled
            pre_rerank_docids = None
            if store_rerank:
                pre_rerank_docids = [r.get("docid") for r in results if "docid" in r]

            # Pass k=None to let the reranker use its configured topk parameter (--reranker-top)
            reranked_results = self.reranker.rerank(
                candidates=results,
                query=query,
                k=None,
                content_key="snippet",
                **kwargs
            )

            if store_rerank:
                assert isinstance(reranked_results, dict), f"When store_rerank=True, reranker must return a dict, got {type(reranked_results)}"
                result = {
                    "results": reranked_results.get("reranked_candidates", []),
                    "rerank_output": reranked_results["raw_response"],
                    "pre_rerank_docids": pre_rerank_docids,
                    "reranked_docids": reranked_results.get("reranked_docids", [])
                }
                if "rerank_reasoning" in reranked_results:
                    result["rerank_reasoning"] = reranked_results["rerank_reasoning"]
                return result

            return reranked_results

        if store_rerank:
            return {"results": results, "rerank_output": None}

        return results
    
    @abstractmethod
    def _retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform retrieval and return results. Subclasses should implement this.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of search results with format: {"docid": str, "score": float, "text": str}
        """
        pass
    
    @abstractmethod
    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve full document by ID.
        
        Args:
            docid: Document ID to retrieve
            
        Returns:
            Document dictionary with format: {"docid": str, "text": str} or None if not found
        """
        pass
    
    @property
    @abstractmethod
    def search_type(self) -> str:
        """Return the type of search (e.g., 'BM25', 'FAISS')."""
        pass 
    
    def search_description(self, k: int = 10) -> str:
        """
        Description of the search tool to be passed to the LLM.
        """
        return f"Perform a search on a knowledge source. Returns top-{k} hits with docid, score, and snippet. The snippet contains the document's contents (may be truncated based on token limits)."
    
    def get_document_description(self) -> str:
        """
        Description of the get_document tool to be passed to the LLM.
        """
        return "Retrieve a full document by its docid."