#!/bin/bash
# sh run_finetune_megatron_seq2seq_glm.sh dsw /workspace/PAI-Megatron-Patch/Megatron-LM/ /workspace/PAI-Megatron-Patch/ 2B 4 608 160 5e-6 5e-7 bf16 1 1 sel true false cnn_dm_original /mnt/GLM-datasets/cnn_dm/  /mnt/glm-ckpts/blocklm-2b-512-to-megatron/ 10 /mnt/output_megatron_glm
set -e
ENV=$1
MEGATRON_PATH=$2
MEGATRON_PATCH_PATH=$3
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
if [ $ENV = dsw ]; then
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8

elif [ $ENV = dlc ]; then

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}

fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

MODEL_SIZE=$4
BATCH_SIZE=$5
SOURCE_SEQ_LEN=$6
TARGET_SEQ_LEN=$7
LR=$8
MIN_LR=$9
PR=${10}
TP=${11}
PP=${12}
AC=${13}
DO=${14}
FL=${15}
SP=${16}
TASK=${17}
DATASET_DIR=${18}
PRETRAIN_CHECKPOINT_PATH=${19}
EPOCH=${20}
OUTPUT_BASEPATH=${21}


if [ ! -f gpt2-vocab.json ]; then
  wget https://easynlp-dev.oss-cn-zhangjiakou.aliyuncs.com/225247/RapidformerPro/gpt2-vocab.json
fi

if [ ! -f gpt2-merges.txt ]; then
  wget https://easynlp-dev.oss-cn-zhangjiakou.aliyuncs.com/225247/RapidformerPro/gpt2-merges.txt
fi


if [ $MODEL_SIZE = 2B ]; then

NUM_LAYERS=36
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=32
SEQ_LEN=1024

elif [ $MODEL_SIZE = 10B ]; then

NUM_LAYERS=48
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=64
SEQ_LEN=1024

elif [ $MODEL_SIZE = 130B ]; then

NUM_LAYERS=70
HIDDEN_SIZE=12288
NUM_ATTN_HEADS=96
SEQ_LEN=2048

fi

if [ $AC = full ]; then
    activation_checkpoint_options=" \
		    --recompute-method uniform \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
                    "
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16"
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
fi

if [ $DO = true ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

if [ $FL = true ]; then
    flash_options=" \
		    --use-flash-attn"

elif [ $FL = false ]; then
    flash_options=" \
                    "
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

NAME="${ENV}-finetune-megatron-bloom-${MODEL_SIZE}-ep-${EPOCH}-lr-${LR}-bs-${BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}-tp-${TP}-pp-${PP}-ac-${AC}-do-${DO}-fl-${FL}-sp-${SP}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}

FINETUNE_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

megatron_options="  \
        --load ${PRETRAIN_CHECKPOINT_PATH} \
        --save ${FINETUNE_CHECKPOINT_PATH} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --keep-last \
        --micro-batch-size ${BATCH_SIZE} \
        --epochs ${EPOCH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style linear \
        --lr-warmup-fraction 0.06 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.01 \
        --num-workers 0\
        --log-interval 1 \
        --eval-interval 100 \
        --eval-iters 10 \
        --save-interval 100000000 \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --finetune \
        --no-load-optim \
        --DDP-impl local\
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --source-seq-len ${SOURCE_SEQ_LEN} \
        --target-seq-len ${TARGET_SEQ_LEN} \
        --task ${TASK} \
        --data-dir ${DATASET_DIR} \
        --patch-tokenizer-type GLMGPT2BPETokenizer \
        --position-embedding-type block \
        --openai-gelu \
        --vocab-file gpt2-vocab.json \
		    --merge-file gpt2-merges.txt \
        "

run_cmd="CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_megatron_seq2seq_glm.py
${megatron_options} ${activation_checkpoint_options} ${do_options} ${pr_options} ${sp_options} ${flash_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
