#! /bin/bash
MEGATRON_PATH=/workspace/PAI-Megatron-Patch/Megatron-LM
PATCH_PATH=/workspace/PAI-Megatron-Patch
export PYTHONPATH=${PATCH_PATH}:${MEGATRON_PATH}:$PYTHONPATH

START_TIME=$SECONDS
input_data_dir=$1
tokenizer=$2
output_data_dir=$3

INPUT="${input_data_dir}/00.jsonl.zst,"`
       `"${input_data_dir}/01.jsonl.zst,"`
       `"${input_data_dir}/02.jsonl.zst,"`
       `"${input_data_dir}/03.jsonl.zst,"`
       `"${input_data_dir}/04.jsonl.zst,"`
       `"${input_data_dir}/05.jsonl.zst,"`
       `"${input_data_dir}/06.jsonl.zst,"`
       `"${input_data_dir}/07.jsonl.zst,"`
       `"${input_data_dir}/08.jsonl.zst,"`
       `"${input_data_dir}/09.jsonl.zst"


if [ $tokenizer = "jiebabpe" ]; then

if [ ! -f tokenizer.json ]; then
  wget https://easynlp-dev.oss-cn-zhangjiakou.aliyuncs.com/225247/RapidformerPro/tokenizer.json
fi

python preprocess_data_gpt_neox.py \
  --input ${INPUT} \
  --language zh \
  --output-prefix ${output_data_dir}/wudao_jiebabpe \
  --dataset-impl mmap \
  --vocab tokenizer.json \
  --tokenizer-type JiebaBPETokenizer \
  --workers 16 \
  --append-eod

elif [ $tokenizer = "bloombpe" ]; then

  python preprocess_data_gpt_neox.py \
  --input ${INPUT} \
  --language zh \
  --output-prefix ${output_data_dir}/wudao_bloombpe \
  --dataset-impl mmap \
  --tokenizer-type BloomTokenizerFromHF \
  --workers 16 \
  --append-eod

fi

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
