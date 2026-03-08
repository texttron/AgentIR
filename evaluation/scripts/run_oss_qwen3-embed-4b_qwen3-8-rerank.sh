#!/usr/bin/env bash

set -euo pipefail

NUM_THREADS=5
AGENT_PORT=8000
RERANK_PORT=18000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        --agent-port)
            AGENT_PORT="$2"
            shift 2
            ;;
        --rerank-port)
            RERANK_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--num-threads N] [--agent-port PORT] [--rerank-port PORT]" >&2
            exit 1
            ;;
    esac
done

python search_agent/oss_client.py \
    --model openai/gpt-oss-120b \
    --output-dir runs/qwen3-4_qwen3-8-rerank/oss_120b \
    --searcher-type faiss \
    --index-path "indexes/qwen3-embedding-4b/corpus*.pkl" \
    --model-name "Qwen/Qwen3-Embedding-4B" \
    --normalize \
    --model-url http://localhost:$AGENT_PORT/v1 \
    --num-threads "${NUM_THREADS}" \
    --query topics-qrels/queries.tsv \
    --reranker-type listwise \
    --reranker-model Qwen/Qwen3-8B \
    --reranker-model-url http://localhost:$RERANK_PORT/v1 \
    --reranker-top 5 \
    --reranker-temperature 0.7 \
    --reranker-top-p 0.8 \
    --reranker-top-k 20 \
    --reranker-max-ranking-output-tokens 1024 \
    --k 20

# --k 20: retriever passes top 20 results for reranker to see
# -- reranker-top 5: reranker passes top 5 results after rerank to agent
# --reranker-temperature, --reranker-top-p, --reranker-top-k: reranker sampling params. This is according to Qwen3-8B's documentation
# --reranker-max-ranking-output-tokens 1024: max 1024 output tokens from the reranker