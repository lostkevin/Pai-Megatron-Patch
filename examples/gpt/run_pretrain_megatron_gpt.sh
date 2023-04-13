#!/bin/bash
# sh run_pretrain_megatron_gpt.sh debug jiebabpe 1.7B 1 16 256 1e-5 1e-6 bf16 2 2 sel z1 false 100 /mnt/wudao/wudao_jiebabpe_text_document none 1000000 10000
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

MODE=$1
TOKENIZER=$2
MODEL_SIZE=$3
BATCH_SIZE=$4
GLOBAL_BATCH_SIZE=$5
SEQ_LEN=$6
LR=$7
MIN_LR=$8
PR=$9
TP=${10}
PP=${11}
AC=${12}
DO=${13}
SP=${14}
SAVE_INTERVAL=${15}
DATASET_PATH=${16}
PRETRAIN_CHECKPOINT_PATH=${17}
TRAIN_TOKENS=${18}
WARMUP_TOKENS=${19}

if [ ! -f gpt2-vocab.json ]; then
  wget https://easynlp-dev.oss-cn-zhangjiakou.aliyuncs.com/225247/RapidformerPro/gpt2-vocab.json
fi

if [ ! -f gpt2-merges.txt ]; then
  wget https://easynlp-dev.oss-cn-zhangjiakou.aliyuncs.com/225247/RapidformerPro/gpt2-merges.txt
fi

if [ $TOKENIZER = jiebabpe ]; then

    if [ ! -f tokenizer.json ]; then
      wget https://easynlp-dev.oss-cn-zhangjiakou.aliyuncs.com/225247/RapidformerPro/tokenizer.json
    fi

    tokenizer_options=" \
		    --patch-tokenizer-type JiebaBPETokenizer \
		    --patch-vocab-file tokenizer.json \
		    --vocab-file gpt2-vocab.json \
		    --merge-file gpt2-merges.txt \
		    "

elif [ $TOKENIZER = bloombpe ]; then

    tokenizer_options=" \
		    --patch-tokenizer-type BloomTokenizerFromHF
		    --vocab-file gpt2-vocab.json \
		    --merge-file gpt2-merges.txt \
		    "
fi

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

if [ $FL = true ]; then
    flash_options=" \
		    --use-flash-attn"

elif [ $FL = false ]; then
    flash_options=" \
                    "
fi

TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

CP_NAME="${MODE}-continue-pretrain-${TASK_NAME}-${TOKENIZER}-megatron-gpt-${MODEL_SIZE}-lr-${LR}-bs-${BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}--do-${DO}-tp-${TP}-ac-${AC}-sp-${SP}-tt-${TRAIN_TOKENS}-wt-${WARMUP_TOKENS}"
OUTPUT_BASEPATH=/mnt/output_megatron_gpt
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
        --DDP-impl local
        "

run_cmd="python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_megatron_gpt.py ${tokenizer_options}
 ${megatron_options} ${activation_checkpoint_options} ${do_options} ${pr_options} ${sp_options} ${flash_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
