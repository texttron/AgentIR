OUTPUT_DIR=indexes/AgentIR_browsecomp-plus
mkdir -p $OUTPUT_DIR

for shard in 0 1 # we embed 1 shard per GPU, you can tweak the number of shards 
do
CUDA_VISIBLE_DEVICES=$shard python -m tevatron.retriever.driver.encode \
  --model_name_or_path Tevatron/AgentIR-4B \
  --dataset_name Tevatron/browsecomp-plus-corpus \
  --encode_output_path $OUTPUT_DIR/corpus.shard.$shard.pkl \
  --passage_max_len 4096 \
  --normalize \
  --pooling eos \
  --passage_prefix "" \
  --per_device_eval_batch_size 24  \
  --fp16 \
  --dataset_number_of_shards 2 \
  --dataset_shard_index $shard &
done
wait