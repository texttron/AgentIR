# Evaluation

This directory contains scaffolds for end-to-end Deep Research, for Tongyi-DeepResearch, GPT-OSS, and GLM.

## BrowseComp-Plus Setup

To download [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) data, run:
```
python prepare_data.py
```
which downloads the benchmark into `data/` and `topics-qrels/`, with pre-built AgentIR-4B index in `indexes/`.

> To encode the corpus yourself using AgentIR-4B, please see [scripts/embed_bcp.sh](scripts/embed_bcp.sh).

### Tongyi-DeepResearch

To evaluate Tongyi-DeepResearch with AgentIR-4B, first host the agent:
```
vllm serve Alibaba-NLP/Tongyi-DeepResearch-30B-A3B --tensor-parallel-size {num_gpus} --port 6008 --gpu-memory-utilization 0.8 
```
where you may replace `{num_gpus}` with the number of GPUs you have available.
> We use `--gpu-memory-utilization 0.8` as we will later be running the embedding model on the same GPU used for hosting the agent. If you will run the embedding model (`scripts/run_tongyi_agentir.sh`) on a separate GPU, this is not needed.

Then, in a separate terminal, run:
```
bash scripts/run_tongyi_agentir.sh --port 6008 --num-threads 10
```

The results will be saved to `runs/agentir/tongyi`. You can then evaluate it using:
```
python evaluate_bcp.py --input_dir runs/agentir/tongyi --tensor_parallel_size {num_gpus}
```
where you may replace `{num_gpus}` with the number of GPUs you have available.

**Note**: The Tongyi-DeepResearch model is trained to use a `visit` tool and issue parallel tool calls (multiple search calls at once). The script above enables these functions. For fair comparison against the official baselines in BrowseComp-Plus without these features, you can also run `bash scripts/run_tongyi_agentir_search-only.sh --port 6008 --num-threads 10`, which disables them.

### GPT-OSS

We recommend creating a separate environment for GPT-OSS:
```
cd oss_env
uv sync
source .venv/bin/activate

vllm serve openai/gpt-oss-120b --tensor-parallel-size {num_gpus} --gpu-memory-utilization 0.8 --tool-call-parser openai --enable-auto-tool-choice --port 6008
```
where you may replace `{num_gpus}` with the number of GPUs you use.

Then, in a separate terminal (using the main `agentir` project env, **not** `oss_env`):
```
bash scripts/run_oss_agentir.sh --port 6008 --num-threads 10
```

The results will be saved to `runs/agentir/oss_120b`. You can then evaluate it using:
```
python evaluate_bcp.py --input_dir runs/agentir/oss_120b --tensor_parallel_size {num_gpus}
```

You can also change the model name in `vllm` hosting and `scripts/run_oss_agentir.sh` to use `openai/gpt-oss-20b` instead of 120b.

### GLM

Setting GLM api key:
```
export ZAI_API_KEY={YOUR_API_KEY}
```

Then, you may run:
```
bash scripts/run_glm_agentir.sh --num-threads 10
```

The results will be saved to `runs/agentir/glm_4_7`. You can then evaluate it using:
```
python evaluate_bcp.py --input_dir runs/agentir/glm_4_7 --tensor_parallel_size {num_gpus}
```

The script above runs GLM-4.7. You can also tweak `--model` to other GLM models.

### Additional Scripts

#### Other Retrievers
The agent scaffolds here are compatible with other retrievers and indexes from [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus).

```
bash scripts/download_additional_indexes.sh # download BrowseComp-Plus pre-built indexes
```

The index usage is similar to the instructions in [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus). To provide some examples:

##### Dense Retrievers

First host the `vllm` server for GPT-OSS as in instructions above. Then, you may see [scripts/run_oss_qwen3-embed-4b.sh](scripts/run_oss_qwen3-embed-4b.sh), which swaps the index parameters to use Qwen3-Embedding-4B. The usage for other agents, and other embedding indexes are similar.

##### BM25

You may see [scripts/run_oss_bm25.sh](scripts/run_oss_bm25.sh) for using the BM25 index. Note that to run the BM25 searcher, you need to first `uv sync --extra bm25` to install `pyserini`, and then install java 21:
```
conda install -c conda-forge openjdk=21
```

#### Reranking

To perform LLM reranking on top of a first-stage retriever, we need to host another vllm server for the reranker. Below we provide an example where we use GPT-OSS as the agent, Qwen3-Embedding-4B as the retriever, and Qwen3-8B as the reranker.

Host GPT-OSS with vllm as before:
```
cd oss_env
source .venv/bin/activate

vllm serve openai/gpt-oss-120b --tensor-parallel-size {num_gpus} --gpu-memory-utilization 0.8 --tool-call-parser openai --enable-auto-tool-choice --port {agent_port}
```

Then, in a separate terminal, host Qwen3-8B as the LLM reranker (using main `agentir` env):
```
vllm serve Qwen/Qwen3-8B --port {reranker_port}
```

Lastly, run the script (using main `agentir` env):
```
bash scripts/run_oss_qwen3-embed-4b_qwen3-8-rerank.sh --agent-port {agent_port} --rerank-port {reranker_port}
```

