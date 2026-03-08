import argparse
import os
import re
import warnings
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

from .base import BaseReranker


@dataclass
class PromptTemplate:
    """Prompt template for reranking."""
    name: str
    system_message: str
    prefix: str
    body: str
    suffix: str
    output_validation_regex: str
    output_extraction_regex: str
    
    def parse_ranking_output(self, output: str, num_candidates: int) -> List[int]:
        """Parse the ranking output from the LLM.
        
        Args:
            output: Raw LLM output string
            num_candidates: Number of candidates that were ranked
            
        Returns:
            List of ranking indices (1-indexed)
        """
        output_stripped = output.strip()
        if not re.search(self.output_validation_regex, output_stripped):
            warnings.warn(
                f"LLM output does not match expected format. "
                f"Expected pattern: {self.output_validation_regex}, "
                f"Got: {output_stripped}"
            )
        
        # Extract all numbers from the output using the extraction regex
        matches = re.findall(self.output_extraction_regex, output)
        
        if not matches:
            warnings.warn(
                f"No ranking indices found in LLM output. "
                f"Returning original order. Output: {output_stripped}"
            )
            return list(range(1, num_candidates + 1))
        
        ranking = [int(m) for m in matches]
        
        # Remove duplicates (keep first occurrence) and filter valid IDs
        expected_ids = set(range(1, num_candidates + 1))
        seen = set()
        unique_ranking = []
        for rank_id in ranking:
            if rank_id not in seen and rank_id in expected_ids:
                seen.add(rank_id)
                unique_ranking.append(rank_id)
        
        if len(unique_ranking) < num_candidates:
            warnings.warn(
                f"Extracted {len(unique_ranking)} unique IDs but expected {num_candidates}. "
                f"Missing IDs will be appended in sorted order."
            )
        
        # Add missing candidates at the end
        missing_ids = expected_ids - seen
        if missing_ids:
            unique_ranking.extend(sorted(missing_ids))
        
        return unique_ranking


class PromptTemplates:
    """Available prompt templates."""
    
    RANK_GPT = PromptTemplate(
        name="rank_gpt",
        system_message="You are an intelligent assistant that can rank passages based on their relevancy to a query",
        prefix="I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.",
        body="[{rank}] {candidate}",
        suffix="Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [2] > [1], Answer concisely and directly and only respond with the ranking results, do not say any word or explain.",
        output_validation_regex=r"^\[\d+\]( > \[\d+\])*$",
        output_extraction_regex=r"\[(\d+)\]"
    )

    RANK_GPT_ORACLE = PromptTemplate(
        name="rank_gpt_oracle",
        system_message="You are an intelligent assistant that can rank passages to best help an user in correctly answering a question. You will be given a complex question, and its correct answer. A user, who is trying to solve the complex question, issues a simpler probing web query, which retrieved some passages. You are to rank these passages based on their relevancy to the user's probing query, while prioritizing their usefulness in helping the user to correctly answer the complex question.",
        prefix="A user is trying to answer a complex question: {question}. The correct answer is: {correct_answer}. A user, trying to solve the complex question, issued this probing web query: {query}, which retrieved {num} passages. I will provide you with these {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the probing query, as well as their usefulness in helping the user to correctly answer the complex question.",
        body="[{rank}] {candidate}",
        suffix="Complex Question: {question}.\nCorrect Answer: {correct_answer}.\nProbing Query: {query}.\nRank the {num} passages above based on their relevance to the probing query, as well as their usefulness in helping the user to correctly answer the complex question. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [2] > [1], Answer concisely and directly and only respond with the ranking results, do not say any word or explain.",
        output_validation_regex=r"^\[\d+\]( > \[\d+\])*$",
        output_extraction_regex=r"\[(\d+)\]"
    )

    @classmethod
    def get_by_name(cls, name: str) -> PromptTemplate:
        """Get a prompt template by name."""
        templates = {
            "rank_gpt": cls.RANK_GPT,
            "rank_gpt_oracle": cls.RANK_GPT_ORACLE
        }
        if name not in templates:
            raise ValueError(f"Unknown prompt template: {name}")
        return templates[name]
    
    @classmethod
    def get_available_names(cls) -> List[str]:
        """Get list of available template names."""
        return ["rank_gpt", "rank_gpt_oracle"]


class ListwiseReranker(BaseReranker):
    """Listwise reranker using LLM to rank passages."""

    def __init__(self, args):
        super().__init__(args)
        self.prompt_template = PromptTemplates.get_by_name(args.prompt_template)
        self.reranker_model_url = args.reranker_model_url
        self.temperature = args.reranker_temperature
        self.top_p = args.reranker_top_p
        self.top_k = args.reranker_top_k
        self.enable_thinking = args.reranker_enable_thinking
        self.max_ranking_output_tokens = args.reranker_max_ranking_output_tokens

        load_dotenv()

        api_key = "EMPTY"
        if args.reranker_api_var:
            api_key = os.getenv(args.reranker_api_var, "EMPTY")

        self.client = OpenAI(api_key=api_key, base_url=args.reranker_model_url)
    
    @classmethod
    def parse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add reranker-specific arguments to the argument parser."""
        parser.add_argument(
            '--reranker-model',
            type=str,
            required=True,
            help='Model name for the reranker'
        )
        parser.add_argument(
            '--prompt-template',
            type=str,
            default='rank_gpt',
            choices=PromptTemplates.get_available_names(),
            help='Prompt template to use for reranking'
        )
        parser.add_argument(
            '--reranker-top',
            type=int,
            default=10,
            help='Number of top results to return after reranking (presented to LLM)'
        )
        parser.add_argument(
            '--reranker-model-url',
            type=str,
            default='http://localhost:18000/v1',
            help='URL of the OpenAI-compatible API server'
        )
        parser.add_argument(
            '--reranker-temperature',
            type=float,
            default=None,
            help='Sampling temperature (default: None, use model default)'
        )
        parser.add_argument(
            '--reranker-top-p',
            type=float,
            default=None,
            help='Nucleus sampling top_p parameter (default: None, use model default)'
        )
        parser.add_argument(
            '--reranker-top-k',
            type=int,
            default=None,
            help='Top-k sampling parameter (default: None, use model default)'
        )
        parser.add_argument(
            '--reranker-enable-thinking',
            action='store_true',
            help='Enable thinking mode for the model'
        )
        parser.add_argument(
            '--reranker-max-ranking-output-tokens',
            type=int,
            default=10240,
            help='Maximum number of output tokens for ranking'
        )
        parser.add_argument(
            '--reranker-api-var',
            type=str,
            default="ZAI_API_KEY",
            help='Environment variable name containing the API key for the reranker'
        )
    
    def _construct_prompt(self, candidate_texts: List[str], query: str, **format_kwargs) -> str:
        """Construct the ranking prompt from template.
        
        Args:
            candidate_texts: List of candidate text strings extracted from candidate dicts
            query: The query string
            **format_kwargs: Additional kwargs for template formatting (e.g., question, correct_answer)
        """
        num_candidates = len(candidate_texts)
        
        format_args = {"num": num_candidates, "query": query, **format_kwargs}
        
        prefix = self.prompt_template.prefix.format(**format_args)
        
        body_parts = []
        for idx, candidate_text in enumerate(candidate_texts, start=1):
            body_part = self.prompt_template.body.format(rank=idx, candidate=candidate_text)
            body_parts.append(body_part)
        
        body = "\n".join(body_parts)
        
        suffix = self.prompt_template.suffix.format(**format_args)
        
        return f"{prefix}\n\n{body}\n\n{suffix}"
    
    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        query: str,
        k: int = 10,
        content_key: str = "text",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Rerank a list of candidates for a given query.
        
        Args:
            candidates: List of candidate dictionaries to rerank
            query: The query string to rank candidates against
            k: Number of top k results to return after reranking (overrides self.topk if provided)
            content_key: The key in each candidate dict that contains the text content to rank (default: "text")
            **kwargs: Additional arguments including:
                - oracle_question (str): For oracle templates, the original complex question
                - correct_answer (str): For oracle templates, the correct answer to the complex question
                - store_rerank (bool): If True, also return the LLM's raw response in addition to the ranking list
            
        Returns:
            List of reranked candidate dictionaries (top k), optionally with raw LLM response
        """
        if not candidates:
            return []
        
        oracle_question = kwargs.get('oracle_question')
        correct_answer = kwargs.get('correct_answer')
        store_rerank = kwargs.get('store_rerank', False)

        # Use provided k or fall back to self.topk
        k = k if k is not None else self.topk

        candidate_texts = [candidate[content_key] for candidate in candidates]

        format_kwargs = {}
        if oracle_question is not None:
            format_kwargs['question'] = oracle_question
        if correct_answer is not None:
            format_kwargs['correct_answer'] = correct_answer
        
        prompt = self._construct_prompt(candidate_texts, query, **format_kwargs)
        
        messages = [
            {"role": "system", "content": self.prompt_template.system_message},
            {"role": "user", "content": prompt}
        ]
        
        api_params = {
            "model": self.reranker_model,
            "messages": messages,
            "max_tokens": self.max_ranking_output_tokens
        }
        
        # Add optional sampling parameters only if they are set
        if self.temperature is not None:
            api_params["temperature"] = self.temperature
        if self.top_p is not None:
            api_params["top_p"] = self.top_p
            
        if self.reranker_model_url == 'https://api.z.ai/api/paas/v4/':
            api_params['extra_body'] = {
                "thinking": {
                    "type": "enabled" if self.enable_thinking else "disabled"
                }
            }
        else:
            api_params["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            }
        
        if self.top_k is not None:
            if 'extra_body' not in api_params:
                api_params['extra_body'] = {}
            api_params["extra_body"]["top_k"] = self.top_k
        
        response = self.client.chat.completions.create(**api_params)
        
        raw_output = response.choices[0].message.content
        
        reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
        
        if raw_output:
            ranking_indices = self.prompt_template.parse_ranking_output(raw_output, len(candidates))
        else:
            warnings.warn("Reranker returned empty content. Returning original order.")
            ranking_indices = list(range(1, len(candidates) + 1))
        
        reranked_candidates = []
        for rank_id in ranking_indices[:k]:
            if 1 <= rank_id <= len(candidates):
                reranked_candidates.append(candidates[rank_id - 1])

        if store_rerank:
            reranked_docids = []
            for rank_id in ranking_indices:
                if 1 <= rank_id <= len(candidates):
                    reranked_docids.append(candidates[rank_id - 1]['docid'])
            
            result = {
                "reranked_candidates": reranked_candidates,
                "raw_response": raw_output,
                "reranked_docids": reranked_docids
            }
            
            if reasoning_content is not None:
                result["rerank_reasoning"] = reasoning_content
            
            return result

        return reranked_candidates
