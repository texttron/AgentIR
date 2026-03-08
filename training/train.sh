OUTPUT_DIR=models/AgentIR-4B-lora
mkdir -p $OUTPUT_DIR

MODEL_NAME=Qwen/Qwen3-Embedding-4B
DATASET_PATH=data/webshaper_train.jsonl

PROMPT="Instruct: Given a user's reasoning followed by a web search query, retrieve relevant passages that answer the query while incorporating the user's reasoning\nQuery:"

MASTER_PORT=60000

deepspeed --include localhost:0 --master_port "${MASTER_PORT}" --module tevatron.retriever.driver.train   --deepspeed deepspeed/ds_zero3_config.json   --output_dir $OUTPUT_DIR   --model_name_or_path $MODEL_NAME --lora --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj --save_strategy epoch  --query_prefix "$PROMPT" --passage_prefix "" --fp16 --pooling last --padding_side left --normalize --temperature 0.01 --per_device_train_batch_size 4 --gradient_checkpointing --train_group_size 8 --learning_rate 1e-4 --query_max_len 8192 --passage_max_len 4096 --num_train_epochs 2 --logging_steps 10 --overwrite_output_dir --gradient_accumulation_steps 2 --dataset_name json --dataset_path $DATASET_PATH
