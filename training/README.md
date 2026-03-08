# Training

To train AgentIR, we finetune Qwen3-Embedding-4B using LoRA.

First, we download the processed webshaper training data:
```
bash download_data.sh
```
> Note: This data was constructed using Tongyi-DeepResearch to generate rollouts. To ensure that each search query has a distinct reasoning, we've restricted the agent from issuing parallel tool calls. However, it sometimes still issues them and errors. We ruled out these erroneous reasonings after tool errors; thus, some queries in the training data have empty reasonings.

Then, launch training with:
```
bash train.sh
```

This saves the trained LoRA checkpoint to `models/AgentIR-4B-lora`. You may load this as a LoRA with Qwen3-Embedding-4B. Alternatively, you can also merge them into one model checkpoint at `models/AgentIR`:
```
python merge_lora.py
```

Then, you can use `models/AgentIR` in place of `Tevatron/AgentIR-4B`!
