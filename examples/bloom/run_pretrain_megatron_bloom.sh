#!/bin/bash
# sh run_pretrain_megatron_bloom.sh 1.7B 1 16 256 1e-5 1e-6 bf16 2 2 sel z1 false 100 /mnt/wudao/wudao_jiebabpe_text_document none 1000000 10000
MEGATRON_PATH=/workspace/PAI-Megatron-Patch/Megatron-LM
PATCH_PATH=/workspace/PAI-Megatron-Patch
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=${PATCH_PATH}:${MEGATRON_PATH}:$PYTHONPATH
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

MODEL_SIZE=$1
BATCH_SIZE=$2
GLOBAL_BATCH_SIZE=$3
SEQ_LEN=$4
LR=$5
MIN_LR=$6
PR=$7
TP=$8
PP=$9
AC=${10}
DO=${11}
SP=${12}
SAVE_INTERVAL=${13}
DATASET_PATH=${14}
PRETRAIN_CHECKPOINT_PATH=${15}
TRAIN_TOKENS=${16}
WARMUP_TOKENS=${17}

if [ $MODEL_SIZE = 1.1B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=1536
NUM_ATTN_HEADS=16

elif [ $MODEL_SIZE = 1.7B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16

elif [ $MODEL_SIZE = 7.1B ]; then

NUM_LAYERS=30
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32

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

if [ $DO = z1 ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = z2 ]; then
    do_options=" \
    --no-contiguous-buffers-in-local-ddp \
    --zero-2-memory-optimization
                    "
elif [ $DO = none ]; then
    do_options=" \
                    "
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

CP_NAME="pretrain-megatron-bloom-${MODEL_SIZE}-lr-${LR}-bs-${BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}--do-${DO}-tp-${TP}-pp-${PP}-ac-${AC}-sp-${SP}-tt-${TRAIN_TOKENS}-wt-${WARMUP_TOKENS}"
OUTPUT_BASEPATH=/mnt/output_megatron_bloom
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${CP_NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}

CONTINUE_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${CP_NAME}"

megatron_options=" \
        --load ${PRETRAIN_CHECKPOINT_PATH} \
        --save ${CONTINUE_PRETRAIN_CHECKPOINT_PATH} \
        --data-path ${DATASET_PATH} \
        --split 98,2,0 \
        --data-impl mmap \
        --data-path ${DATASET_PATH}
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style linear \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --init-method-std 0.006 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --log-interval 1 \
        --eval-interval 100 \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --DDP-impl local \
        --no-load-optim \
        --no-load-rng \
        --finetune \
        --embed-layernorm \
        --glu-activation geglu \
        --position-embedding-type alibi \
        --patch-tokenizer-type BloomTokenizerFromHF
        "

run_cmd="python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_megatron_bloom.py
 ${megatron_options} ${activation_checkpoint_options} ${do_options} ${pr_options} ${sp_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
