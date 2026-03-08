#!/usr/bin/env bash

set -euo pipefail

NUM_THREADS=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--num-threads N]" >&2
            exit 1
            ;;
    esac
done

python search_agent/glm_client.py \
    --model glm-4.7 \
    --output-dir runs/agentir/glm_4_7 \
    --num-threads "${NUM_THREADS}" \
    --searcher-type faiss \
    --index-path "indexes/AgentIR_browsecomp-plus/corpus*.pkl" \
    --model-name "Tevatron/AgentIR-4B" \
    --normalize \
    --reasoning-aware \
    --task-prefix "Instruct: Given a user's reasoning followed by a web search query, retrieve relevant passages that answer the query while incorporating the user's reasoning\nQuery:" \
    --query topics-qrels/queries.tsv
