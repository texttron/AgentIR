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

python search_agent/oss_client.py \
    --model openai/gpt-oss-120b \
    --output-dir runs/qwen3-4/oss_120b \
    --searcher-type bm25 \
    --index-path "indexes/bm25" \
    --model-url http://localhost:$PORT/v1 \
    --num-threads "${NUM_THREADS}" \
    --query topics-qrels/queries.tsv