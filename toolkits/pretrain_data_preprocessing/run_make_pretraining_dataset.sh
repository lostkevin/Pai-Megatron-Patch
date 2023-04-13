#! /bin/bash
MEGATRON_PATH=/workspace/PAI-Megatron-Patch/Megatron-LM
PATCH_PATH=/workspace/PAI-Megatron-Patch
export PYTHONPATH=${PATCH_PATH}:${MEGATRON_PATH}:$PYTHONPATH

START_TIME=$SECONDS
input_data_dir=$1
tokenizer=$2
output_data_dir=$3

INPUT="${input_data_dir}/00.jsonl.zst"


if [ $tokenizer = "jiebabpe" ]; then

if [ ! -f tokenizer.json ]; then
  wget https://easynlp-dev.oss-cn-zhangjiakou.aliyuncs.com/225247/RapidformerPro/tokenizer.json
fi

python preprocess_data.py \
  --input ${INPUT} \
  --language zh \
  --output-prefix ${output_data_dir}/wudao_jiebabpe \
  --dataset-impl mmap \
  --vocab tokenizer.json \
  --patch-tokenizer-type JiebaBPETokenizer \
  --workers 16 \
  --append-eod

elif [ $tokenizer = "bloombpe" ]; then

  python preprocess_data.py \
  --input ${INPUT} \
  --language zh \
  --output-prefix ${output_data_dir}/wudao_bloombpe \
  --dataset-impl mmap \
  --patch-tokenizer-type BloomTokenizerFromHF \
  --workers 16 \
  --append-eod

fi

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
