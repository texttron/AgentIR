#!/usr/bin/env bash

set -euo pipefail

NUM_THREADS=5
PORT=8000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--num-threads N] [--port PORT]" >&2
            exit 1
            ;;
    esac
done

python search_agent/tongyi_client.py \
    --output-dir runs/agentir/tongyi_search-only \
    --searcher-type faiss \
    --index-path "indexes/AgentIR_browsecomp-plus/corpus*.pkl" \
    --model-name "Tevatron/AgentIR-4B" \
    --normalize \
    --max_workers "${NUM_THREADS}" \
    --reasoning-aware \
    --task-prefix "Instruct: Given a user's reasoning followed by a web search query, retrieve relevant passages that answer the query while incorporating the user's reasoning\nQuery:" \
    --port "${PORT}" \
    --query topics-qrels/queries.tsv
