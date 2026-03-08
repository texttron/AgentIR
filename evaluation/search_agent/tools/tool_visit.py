import os
import argparse
from urllib.parse import unquote
from typing import Dict, Any, Optional
from datasets import load_dataset

from evaluation.search_agent.tools.types import VisitTurn

VISIT_RESPONSE_TEMPLATE = """The useful information in {url} for user goal {goal} as follows:

Evidence in page:
{evidence}

Summary:
{summary}

"""


class VisitToolHandler:
    name = "visit"
    def __init__(
        self,
        corpus_path: str,
        max_truncation_length: int = 25000,
    ):
        """
        Initialize the VisitToolHandler.
        
        Args:
            corpus_path: HuggingFace dataset path with train split containing docid, text, url fields
            max_truncation_length: Maximum length for raw text truncation
        """
        self.corpus_path = corpus_path
        self.max_truncation_length = max_truncation_length
        
        self.docid_to_text = {}
        self.url_to_docid = {}
        self._load_corpus()
        
        self.description = "Visit webpage(s) and return the summary of the content."
    
    def _normalize_url(self, url: str) -> str:
        if not url:
            return url
        # Decode percent-encoded characters, replace spaces with underscores, and strip trailing slash
        return unquote(url).replace(' ', '_').rstrip('/')
    
    def _load_corpus(self):
        try:
            dataset_cache = os.getenv('HF_DATASETS_CACHE')
            cache_dir = dataset_cache if dataset_cache else None
            
            if self.corpus_path.endswith('.jsonl'):
                ds = load_dataset('json', data_files=self.corpus_path, split='train', cache_dir=cache_dir)
            else:
                ds = load_dataset(self.corpus_path, split='train', cache_dir=cache_dir)
            
            for row in ds:
                docid = row.get('docid')
                if docid is None:
                    docid = row.get('id')
                docid = str(docid)
                url = row['url']
                text = row.get('text')
                if not text:
                    text = row['contents']
                
                self.docid_to_text[docid] = text
                normalized_url = self._normalize_url(url)
                self.url_to_docid[normalized_url] = docid
        except Exception as e:
            print(f"[Visit] Error loading corpus: {e}")
            raise

    def visit_single_url(self, url: str, goal: str) -> str:
        normalized_url = self._normalize_url(url)
        docid = self.url_to_docid.get(normalized_url)
        
        if docid and docid in self.docid_to_text:
            evidence = self.docid_to_text[docid][:self.max_truncation_length]
            summary = "The webpage is potentially relevant."
        else:
            evidence = "The provided webpage content could not be accessed. Please check the URL or file format."
            summary = "The webpage content could not be processed, and therefore, no information is available."

        return VISIT_RESPONSE_TEMPLATE.format(
            url=url,
            goal=goal,
            evidence=evidence,
            summary=summary,
        )
    
    def handle(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        current_thinking: Optional[str]
    ) -> VisitTurn:
        """
        Handle visit tool call.
        
        Args:
            tool_name: Name of the tool (should be "visit")
            tool_args: Arguments containing 'url' and 'goal'
            current_thinking: Current thinking/reasoning from the agent
            
        Returns:
            VisitTurn with the visit results
        """
        visit_turn = VisitTurn(
            tool_name=tool_name,
            reasoning=current_thinking,
            args=tool_args,
            success=False
        )
        
        try:
            url = tool_args["url"]
            goal = tool_args["goal"]
        except KeyError as e:
            visit_turn.response = f"[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"
            visit_turn.error = str(e)
            return visit_turn
        
        if isinstance(url, str):
            urls = [url]
        elif isinstance(url, list):
            urls = url
        else:
            visit_turn.response = "[Visit] Invalid request format: 'url' must be a string or array"
            visit_turn.error = "Invalid url format"
            return visit_turn
        
        responses = []
        docids = []
        
        for u in urls:
            try:
                response = self.visit_single_url(u, goal)
                responses.append(response)
                
                docid = self.url_to_docid.get(self._normalize_url(u))
                if docid:
                    docids.append(docid)
            except Exception as e:
                error_response = f"Error fetching {u}: {str(e)}"
                responses.append(error_response)
        
        visit_turn.response = "\n=======\n".join(responses)
        visit_turn.docids = docids if docids else None
        visit_turn.success = len(responses) > 0
        
        return visit_turn
    
    @classmethod
    def parse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add visit-specific arguments to the argument parser."""
        parser.add_argument(
            '--visit-max-truncation-length',
            type=int,
            default=25000,
            help='Maximum length for raw text truncation in the visit tool (default: 25000)'
        )
