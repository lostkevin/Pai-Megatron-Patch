#!/bin/bash
#sh run_finetune_megatron_bloom.sh 1.7B 4 256 1e-5 1e-6 bf16 1 sel z1 false chat /mnt/ChatGPT/instruct.json /mnt/ChatGPT/instruct.json /mnt/bloom-ckpts/bloomz-1b7-to-megatron/ 6
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
SEQ_LEN=$3
LR=$4
MIN_LR=$5
PR=$6
TP=$7
AC=$8
DO=$9
SP=${10}
TASK_NAME=${11}
TRAIN_DATASET_PATH=${12}
VALID_DATASET_PATH=${13}
PRETRAIN_CHECKPOINT_PATH=${14}
EPOCH=${15}

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

FT_NAME="finetune-${TASK_NAME}-megatron-bloom-${MODEL_SIZE}-lr-${LR}-ep-${EPOCH}-bs-${BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}--do-${DO}-tp-${TP}-ac-${AC}-sp-${SP}"
OUTPUT_BASEPATH=/mnt/output_megatron_bloom
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${FT_NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}

FINETUNE_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${FT_NAME}"
LOGGING_PATH="${OUTPUT_BASEPATH}/log/${FT_NAME}_${current_time}"


megatron_options="  \
        --load ${PRETRAIN_CHECKPOINT_PATH} \
        --save ${FINETUNE_CHECKPOINT_PATH} \
        --train-data ${TRAIN_DATASET_PATH} \
        --valid-data ${VALID_DATASET_PATH} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings 2048 \
        --keep-last \
        --micro-batch-size ${BATCH_SIZE} \
        --epochs ${EPOCH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
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
        --task ${TASK_NAME} \
        --tensor-model-parallel-size ${TP} \
        --patch-tokenizer-type BloomTokenizerFromHF \
        --embed-layernorm \
        --glu-activation geglu \
        --position-embedding-type alibi
        "

run_cmd="python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_megatron_bloom.py
${megatron_options} ${activation_checkpoint_options} ${do_options} ${pr_options} ${sp_options}"


echo ${run_cmd}
eval ${run_cmd}
set +x
