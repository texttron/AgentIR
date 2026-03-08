"""
BM25 searcher implementation using Pyserini.
"""

import json
import logging
import threading
from typing import Any, Dict, Optional

from pyserini.search.lucene import LuceneSearcher

from .base import BaseSearcher

logger = logging.getLogger(__name__)


class BM25Searcher(BaseSearcher):
    @classmethod
    def _parse_searcher_args(cls, parser):
        parser.add_argument(
            "--index-path",
            required=True,
            help="Name or path of Lucene index (e.g. msmarco-v1-passage).",
        )
    
    def _init_searcher(self, args):
        if not args.index_path:
            raise ValueError("index_path is required for BM25 searcher")
        
        self.args = args
        self.searcher = None
        self.searcher_lock = threading.Lock()  # Thread safety for searcher
        
        logger.info(f"Initializing BM25 searcher with index: {args.index_path}")
        
        try:
            self.searcher = LuceneSearcher(args.index_path)
        except Exception as exc:
            raise ValueError(
                f"Index '{args.index_path}' is not a valid local Lucene index path."
            ) from exc
        
        logger.info("BM25 searcher initialized successfully")
    
    def _retrieve(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        if not self.searcher:
            raise RuntimeError("Searcher not initialized")
        
        with self.searcher_lock:  # Thread-safe searcher access
            hits = self.searcher.search(query, k)
            results = []
            
            for hit in hits:
                raw = json.loads(hit.lucene_document.get("raw"))
                results.append({
                    "docid": hit.docid,
                    "score": hit.score,
                    "text": raw["contents"]
                })
        
        return results
    
    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        if not self.searcher:
            raise RuntimeError("Searcher not initialized")
        
        with self.searcher_lock:  # Thread-safe searcher access
            doc = self.searcher.doc(docid)
            if doc is None:
                return None
            
            return {
                "docid": docid,
                "text": json.loads(doc.raw())["contents"],
            }
    
    @property
    def search_type(self) -> str:
        return "BM25" 