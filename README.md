## PAI-Megatron-Patch
PAI-Megatron-Patch是一款对开源Megatron框架进行非代码侵入设计的补丁工具，方便用户低门槛的使用Megatron引擎来训练Bloom，GLM等大模型训练。目前支持下列系列大模型基于MegatronLM框架的继续预训练，有监督微调（SFT），和人类反馈强化学习（RLHF）：
* bloom
* chatglm
* falcon
* galactica
* llama
* glm130B
相关的megatronLM版本的模型ckpt，请联系阿里云产品经理获取。
## 安装指南
1. 在DSW的Terminal中进入工作目录：
```bash
cd /mnt/workspace/
```
2. 下载开源社区Megatron源代码：
```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
```
3. 获取Megatron版模型权重（以bloom7B为例）
```bash
cd /mnt/workspace/
mkdir bloom-ckpts
cd bloom-ckpts
wget https://atp-modelzoo.oss-cn-hangzhou.aliyuncs.com/release/models/pai-megatron-patch/bloom-ckpts/bloom7b1-hf2mg-tp1-pp1.tgz
tar -zxf bloom7b1-hf2mg-tp1-pp1.tgz
```
4. 将PAI-Megatron-Patch源代码拷贝到工作目录/mnt/workspace/下
## 继续预训练
### 数据预处理
建议在灵骏智算平台中的DSW实例中准备预训练数据，以下是针对wudao2.0数据集的准备流程：
1. 下载WuDaoCorpora2.0开源数据集到/mnt/workspace/工作目录下，文件夹命名为WuDaoCorpus2.0_base_200G。我们提供了部分样例数据帮助用户提熟悉流程，参考以下代码下载并解压得到WuDaoCorpus2.0_base_sample：
```bash
cd /mnt/workspace/
mkdir bloom-datasets
cd bloom-datasets
wget https://atp-modelzoo.oss-cn-hangzhou.aliyuncs.com/release/datasets/WuDaoCorpus2.0_base_sample.tgz
tar zxvf WuDaoCorpus2.0_base_sample.tgz 
```
2. 对Wudao数据执行数据集清洗并打包成zst压缩文件格式。具体流程可参考如下的bash脚本。
```bash
#! /bin/bash
export WORK_DIR=/mnt/workspace

cd ${WORK_DIR}/PAI-Megatron-Patch/toolkits/pretrain_data_preprocessing

mkdir -p ${WORK_DIR}/cleaned_wudao_dataset
#清洗数据
python clean_raw_text.py -i ${WORK_DIR}/WuDaoCorpus2.0_base_sample  -o ${WORK_DIR}/cleaned_wudao_dataset -p 32

#合并清洗后的数据，并根据100（根据情况进行修改）进行分块
mkdir ${WORK_DIR}/wudao
find ${WORK_DIR}/cleaned_wudao_dataset -name "*.json" -exec cat {} + > ${WORK_DIR}/wudao/merged_wudao_cleaned.json
split -l 1000 --numeric-suffixes --additional-suffix=.jsonl ${WORK_DIR}/wudao/merged_wudao_cleaned.json ${WORK_DIR}/wudao/

#数据压缩
apt-get update
apt-get install zstd
mkdir ${WORK_DIR}/wudao/cleaned_zst
zstd -z ${WORK_DIR}/wudao/00.jsonl -o ${WORK_DIR}/wudao/cleaned_zst/00.jsonl.zst &
zstd -z ${WORK_DIR}/wudao/01.jsonl -o ${WORK_DIR}/wudao/cleaned_zst/01.jsonl.zst &
zstd -z ${WORK_DIR}/wudao/02.jsonl -o ${WORK_DIR}/wudao/cleaned_zst/02.jsonl.zst &
zstd -z ${WORK_DIR}/wudao/03.jsonl -o ${WORK_DIR}/wudao/cleaned_zst/03.jsonl.zst &
zstd -z ${WORK_DIR}/wudao/04.jsonl -o ${WORK_DIR}/wudao/cleaned_zst/04.jsonl.zst &
zstd -z ${WORK_DIR}/wudao/05.jsonl -o ${WORK_DIR}/wudao/cleaned_zst/05.jsonl.zst &
zstd -z ${WORK_DIR}/wudao/06.jsonl -o ${WORK_DIR}/wudao/cleaned_zst/06.jsonl.zst &
zstd -z ${WORK_DIR}/wudao/07.jsonl -o ${WORK_DIR}/wudao/cleaned_zst/07.jsonl.zst &
zstd -z ${WORK_DIR}/wudao/08.jsonl -o ${WORK_DIR}/wudao/cleaned_zst/08.jsonl.zst &
zstd -z ${WORK_DIR}/wudao/09.jsonl -o ${WORK_DIR}/wudao/cleaned_zst/09.jsonl.zst &
```
3. 制作MMAP格式预训练数据集。
在DSW的Terminal中进入代码目录：/mnt/workspace/PAI-Megatron-Patch/toolkits/pretrain_data_preprocessing。查看run_make_pretraining_dataset.sh脚本内容。里面有五个启动参数需要在运行时输入，具体参数列表如下：
```bash
MEGATRON_PATH=$1                   # 设置开源Megatron的代码路径
MEGATRON_PATCH_PATH=$2             # 设置Megatron Patch的代码路径
input_data_dir=$3                  # 打包后的wudao数据集的文件夹路径
tokenizer=$4                       # bloombpe
output_data_dir=$5                 # 输出到bin和idx文件目录  
```
运行示例如下所示：
```bash
export WORK_DIR=/mnt/workspace
bash run_make_pretraining_dataset.sh ${WORK_DIR}/Megatron-LM ${WORK_DIR}/PAI-Megatron-Patch/ ${WORK_DIR}/wudao/cleaned_zst bloombpe ${WORK_DIR}/wudao
```
### 基于PAI-DSW继续预训练
基于上一章中下载转换好的模型的存放位置，以及上一节中处理好的训练数据，在DSW环境调试继续预训练模型。DSW的Terminal中运行run_pretrain_megatron_bloom.sh脚本，需要传入的参数列表如下：
```bash
ENV=$1                             # 运行环境配置：dsw,dlc
MEGATRON_PATH=$2                   # 设置开源Megatron的代码路径
MEGATRON_PATCH_PATH=$3             # 设置Megatron Patch的代码路径
MODEL_SIZE=$4                      # 模型结构参数量级: 1.1B,1.7B,7.1B
BATCH_SIZE=$5                      # 每卡训练一次迭代样本数: 4, 8
GLOBAL_BATCH_SIZE=$6               # 一次迭代总样本数: 32, 64
LR=$7                              # 学习率: 1e-5, 5e-5
MIN_LR=$8                          # 最小学习率: 1e-6, 5e-6
SEQ_LEN=$9                         # 序列长度: 256, 512
EXTRA_VOCAB_SIZE=${10}             # 扩充词表大小
PR=${11}                           # 训练精度: fp16, bf16
TP=${12}                           # 模型并行度
PP=${13}                           # 流水并行度
AC=${14}                           # 梯度检查点类型: full, sel
DO=${15}                           # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${16}													 # 是否打开flash attention
SP=${17}                           # 是否打开序列并行加速: true, false
SAVE_INTERVAL=${18}                # 保存ckpt的间隔迭代数
DATASET_PATH=${19}                 # mmap数据集路径
PRETRAIN_CHECKPOINT_PATH=${20}     # 预训练模型路径
TRAIN_TOKENS=${21}                 # 训练token数
WARMUP_TOKENS=${22}                # 预热tokens数
OUTPUT_BASEPATH=${23}              # 训练输出文件路径
```
DSW单机运行示例如下：
```bash
export WORK_DIR=/mnt/workspace
cd ${WORK_DIR}/PAI-Megatron-Patch/examples/bloom
sh run_pretrain_megatron_bloom.sh \
dsw \
${WORK_DIR}/Megatron-LM \
${WORK_DIR}/PAI-Megatron-Patch/ \
7.1B  \
4     \
32    \
1e-5  \
1e-6  \
256   \
0     \
bf16  \
1     \
1     \
sel   \
true  \
false \
false \
10000 \
${WORK_DIR}/wudao/wudao_bloombpe_text_document \
${WORK_DIR}/bloom-ckpts/bloomz-7b1-to-megatron-tp1-pp1 \
1000000 \
100000  \
${WORK_DIR}/output_megatron_bloom/
```
## 有监督微调（SFT）
基于上一章继续预训练的模型，或者在开源模型的基础上，可以进一步进行有监督微调（SFT），SFT的数据准备可参考附录。DSW的Terminal中运行run_finetune_megatron_bloom.sh脚本，需要传入的参数列表如下：
```bash
export WORK_DIR=/mnt/workspace
cd ${WORK_DIR}/PAI-Megatron-Patch/examples/bloom
sh run_finetune_megatron_bloom.sh \
dsw \
${WORK_DIR}/Megatron-LM \
${WORK_DIR}/PAI-Megatron-Patch/ \
7.1B  \
4     \
1e-5  \
1e-6  \
256   \
128   \
1     \
bf16  \
1     \
1     \
sel   \
true  \
false \
false \
${WORK_DIR}/bloom-datasets/instruct_latest.json \
${WORK_DIR}/bloom-datasets/instruct_latest.json \
${WORK_DIR}/bloom-ckpts/bloomz-7b1-to-megatron-tp1-pp1 \
16 \
${WORK_DIR}/output_megatron_bloom/
```
DSW单机运行示例如下：
```bash
export WORK_DIR=/mnt/workspace
cd ${WORK_DIR}/PAI-Megatron-Patch/examples/bloom
sh run_finetune_megatron_bloom.sh \
dsw \
${WORK_DIR}/Megatron-LM \
${WORK_DIR}/PAI-Megatron-Patch/ \
7.1B  \
4     \
1e-5  \
1e-6  \
256   \
128   \
1     \
bf16  \
1     \
1     \
sel   \
true  \
false \
false \
${WORK_DIR}/bloom-datasets/instruct_latest.json \
${WORK_DIR}/bloom-datasets/instruct_latest.json \
${WORK_DIR}/bloom-ckpts/bloomz-7b1-to-megatron-tp1-pp1 \
16 \
${WORK_DIR}/output_megatron_bloom/
```
## 人类反馈强化学习（RLHF）
一般来说，SFT微调过的模型在对话场景已经会有不错的表现了。如果想进一步的提升模型效果，可以再加上RLHF训练。包括奖励模型（reward model）的训练和强化学习（ppo）的训练。
### 奖励模型（RM）
#### 调试奖励模型训练脚本
对于SFT模型，可以训练一个评估其生成能力的奖励模型（RM），RM数据可使用现有开源数据。在DSW的Terminal中，首先运行如下脚本：
```bash
git clone https://github.com/CarperAI/trlx.git
cp trlx_bloom_rlhf.py trlx_bloom_rlhf_test.py trlx/examples/summarize_rlhf/
cp train_reward_model_bloom.py reward_model_bloom.py ds_config_bloom.json trlx/examples/summarize_rlhf/reward_model/
cp -f ds_config_trlx_gptj_summarize.json trlx/examples/summarize_rlhf/configs/
cd trlx
pip install -e .
```
训练奖励模型，下面以bloom1b模型为例，说明奖励模型的训练过程：
```bash
cd examples/summarize_rlhf/reward_model/ && deepspeed train_reward_model_bloom.py
```
#### 模型checkpoint格式转换
对于使用HuggingFace或DeepSpeed框架训练得到的reward模型checkpoint，可以通过模型格式转换工具中的reward_model_convertor_megatron.sh脚本（），将其转为megatron格式，并连同同样转为megatron格式的SFT模型，供后续强化学习（PPO）训练使用。脚本运行方式如下：
```bash
sh model_convertor_huggingface_megatron.sh \
./Megatron-LM \
./convert_models/hf-reward/reward-bloom-1b1 \
./convert_models/megatron-reward/megatron-reward-1b1-8 \
8 \
1 \
false
```
最终得到的模型文件目录结构如下：
```bash
.
├── convert_models
│   ├── hf-reward
│   │   └── reward-bloom-1b1
│   │       ├── pytorch_model.bin
│   │       └── ...
│   ├── hf-sft
│   │   └── bloom-7b1-sft
│   │       ├── pytorch_model.bin
│   │       └── ...
│   ├── megatron-reward
│   │   └── megatron-reward-7b1-8
│   │       ├── latest_checkpointed_iteration.txt
│   │       ├── release
│   │       │   ├── mp_rank_00
│   │       │   │   └── model_rng.pt
│   │       │   ├── mp_rank_01
│   │       │   │   └── model_rng.pt
│   │       │   └── ...
│   │       └── ...
│   └── megatron-sft
│       └── bloom-1b1-8
│           ├── latest_checkpointed_iteration.txt
│           ├── release
│           │   ├── mp_rank_00
│           │   │   └── model_rng.pt
│           │   ├── mp_rank_01
│           │   │   └── model_rng.pt
│           │   └── ...
│           └── ...
└── hf-to-megatron
    ├── bloom
    │   ├── checkpoint_reshaping_and_interoperability.py
    │   ├── model_convertor_huggingface_megatron.sh
    │   ├── reward_model_convertor_megatron.sh
    │   ├── reward_model_to_megatron.py
    │   └── ...
    └── ...
```
### PPO训练
#### SFT&RM模型准备
PPO训练共涉及到6个模型，其中3个SFT模型（1个用于训练、2个用于推理，本示例采用的bloom7B1），3个RM模型（1个用于训练、2个用于推理，本示例采用的bloom1B1）。同时各模型文件夹需针对Megatron分布式要求进行结构调整。基于前述步骤中准备好的Megatron版本的SFT模型和RM模型，运行脚本，需要传入的参数如下：
```bash
SOURCE_RM_PATH=$1					# Megatron版RM模型地址
SOURCE_SFT_PATH=$2				# Megatron版SFT模型地址
TARGET_RM_PATH=$3					# 调整好的PPO阶段RM模型存放地址
TARGET_SFT_PATH=$4				# 调整好的PPO阶段SFT模型存放地址
```
运行sft_rm_model_prepare.sh脚本，准备好PPO训练所需的各个模型：
```bash
bash sft_rm_model_prepare.sh \
/mnt/workspace/convertor/convert_models/hf-reward/reward-bloom-1b1 \
/mnt/workspace/convertor/convert_models/hf-sft/bloom-7b1-sft \
/mnt/workspace/exp/bloom_1b1/ \
/mnt/workspace/exp/bloom_7b1/
```
调整好的PPO阶段训练所使用的模型文件目录结构如下：
```bash
.
├── rw_pred
│   ├── latest_checkpointed_iteration.txt
│   ├── release
│   │   ├── mp_rank_00
│   │   │   └── model_optim_rng.pt
│   │   ├── mp_rank_01
│   │   │   └── model_optim_rng.pt
│   │   └── ...
│   └── ...
├── rw_train
│   ├── latest_checkpointed_iteration.txt
│   ├── release
│   │   ├── mp_rank_00
│   │   │   └── model_rng.pt
│   │   ├── mp_rank_00_000
│   │   │   ├── model_rng.pt
│   │   │   └── optim.pt
│   │   ├── mp_rank_01
│   │   │   └── model_rng.pt
│   │   ├── mp_rank_01_000
│   │   │   ├── model_rng.pt
│   │   │   └── optim.pt
│   │   └── ...
│   └── ...
├── sft_pred
│   ├── latest_checkpointed_iteration.txt
│   ├── release
│   │   ├── mp_rank_00
│   │   │   └── model_optim_rng.pt
│   │   ├── mp_rank_01
│   │   │   └── model_optim_rng.pt
│   │   └── ...
│   └── ...
└── sft_train
    ├── latest_checkpointed_iteration.txt
    ├── release
    │   ├── mp_rank_00
    │   │   └── model_rng.pt
    │   ├── mp_rank_00_000
    │   │   ├── model_rng.pt
    │   │   └── optim.pt
    │   ├── mp_rank_01
    │   │   └── model_rng.pt
    │   ├── mp_rank_01_000
    │   │   ├── model_rng.pt
    │   │   └── optim.pt
    │   └── ...
    └── ...
```
#### 调试PPO训练脚本
1. 获取训练代码
请联系阿里云产品经理获取RLHF训练框架rlhf，以及其依赖的Megatron版本Megatron-LM源代码，将两部分代码并列拷贝至/mnt/workspace/RLHF/。
2. 配置训练参数
基于前两章训练的SFT模型和Reward模型，可以进行PPO训练。DSW的Terminal中，首先进入指定文件夹：
```bash
cd /mnt/workspace/RLHF/rlhf/examples/chatgpt/
```
配置./configs/目录下的yaml配置文件，参数列表如下：
```bash
runtime_env:
    working_dir: ./

models:
    ppo_policy:
        model_config_file: configs/ppo_policy.yaml
        num_device: 8
        gpu_per_process: 1
        trainable: True
    policy:
        model_config_file: configs/policy.yaml
        num_device: 8
        gpu_per_process: 1
        trainable: False
        return_rlhf_data: True
    reference:
        model_config_file: configs/reference.yaml
        num_device: 8
        gpu_per_process: 1
        trainable: False
        return_rlhf_data: True
    reward_train:
        model_config_file: configs/reward_train.yaml
        num_device: 1
        gpu_per_process: 1
        trainable: True
    reward_forward:
        model_config_file: configs/reward_forward.yaml
        num_device: 1
        gpu_per_process: 1
        trainable: False
        return_rlhf_data: True
    reward_ref:
        model_config_file: configs/reward_ref.yaml
        num_device: 1
        gpu_per_process: 1
        trainable: False
        return_rlhf_data: True

rlhf:
    colocation:
        - policy,ppo_policy,reward_forward,reference,reward_ref,reward_train
    generation_batch_size: 2
    train_micro_batch_size: 2
    num_ppo_episode: 5
    sample_per_episode: 10
    num_training_epoch: 1
```
分别配置./config/目录下的ppo_policy.yaml、policy.yaml、reference.yaml、reward_train.yaml、reward_forward.yaml、reward_ref.yaml六个yaml配置文件，举例ppo_policy.yaml文件配置如下：
```bash
tensor_model_parallel_size: 8
pipeline_model_parallel_size: 1
num_layers: 30
hidden_size: 4096
num_attention_heads: 32
micro_batch_size: 2
seq_length: 1024
max_position_embeddings: 2048
train_iters: 5
lr_decay_iters: 320000
save: /workspace/exp/bloom_7b1/sft_out
load: /workspace/exp/bloom_7b1/sft_train
data_path:
    - my-gpt2_text_document
vocab_file: gpt2-vocab.json
merge_file: gpt2-merges.txt
data_impl: mmap
split: "949,50,1"
distributed_backend: nccl
lr: 0.00015
min_lr: 0.00001
lr_decay_style: cosine
weight_decay: 0.02
clip_grad: 1.0
lr_warmup_fraction: 0.01
log_interval: 1
save_interval: 1000
eval_interval: 1000
eval_iters: 10
bf16: True
use_checkpoint_opt_param_scheduler: False
no_load_optim: True
finetune: True
glu_activation: geglu
position_embedding_type: alibi
embed_layernorm: True

recompute_granularity: selective
sequence_parallel: False
use_distributed_optimizer: True
```
3. 运行训练脚本
配置好所有yaml文件后，运行run_rlhf.sh脚本，需要传入的参数列表如下：
```bash
nohup bash ./run_rlhf.sh 0,1,2,3,4,5,6,7 > ./log/log.txt 2>&1 &
```
## 模型离线推理
模型训练完成后，可以用megatronLM直接进行离线推理，评估模型效果。考虑有用户可能已经在生产链路中集成了huggingface进行推理，我们也提供了模型从megatron转换成huggingface的工具
### 调试推理脚本
```bash
ENV=$1                          # 运行环境: dlc, dsw
MEGATRON_PATH=$2                # 设置开源Megatron的代码路径
MEGATRON_PATCH_PATH=$3          # 设置Megatron Patch的代码路径
TOKENIZER=$4                    # 选择对应的tokenizer，bloombpe
CHECKPOINT_PATH=$5              # 预训练模型路径
MODEL_SIZE=$6                   # 模型结构参数量级: 1.1B, 1.7B, 7.1B
TP=$7                           # 模型并行度
BS=$8                           # 每卡推理一次迭代样本数: 1, 4, 8
SEQ_LEN=$9											# 序列长度: 256, 512, 1024
TOP_K=${10}                     # 采样策略中选择排在前面的候选词数量(0-n): 0, 5, 10, 20
INPUT_SEQ_LEN=${11}             # 输入序列长度: 512
OUTPUT_SEQ_LEN=${12}            # 输出序列长度: 256
INPUT_FILE=${13}                # 需要推理的文本文件: input.txt, 每行为一个样本
OUTPUT_FILE=${14}               # 推理输出的文件: output.txt
# TOP_K和TOP_P必须有一个为0
TOP_P=${15}                     # 采样策略中选择排在前面的候选词百分比(0-1): 0, 0.85, 0.95
TEMPERATURE=${16}               # 采样策略中温度惩罚: 1-n
REPETITION_PENALTY=${17}        # 避免生成是产生大量重复，可以设置为(1-2)默认为1.2
```
```bash
export WORK_DIR=/mnt/workspace
cd ${WORK_DIR}/PAI-Megatron-Patch/examples/bloom
bash run_text_generation_bloom.sh \
dsw \
${WORK_DIR}/Megatron-LM \
${WORK_DIR}/PAI-Megatron-Patch \
bloombpe \
../../../bloomwcp-shrink_mg_test \
7.1B \
1 \
2 \
1024 \
10 \
512 \
512 \
/mnt/workspace/cn.preds.txt \
/mnt/workspace/bloom_pred.txt \
0 \
1.0 \
1.2
```
### 转换成huggingface进行推理
根据上述训练好的模型ckpt，转换成huggingface版本，参考下列代码，注意模型地址换成上述训练完后保存好的。模型转换代码需要获取hf-to-megatron代码，参考“模型下载及格式转换”章节。
```bash
cd /mnt/workspace/hf-to-megatron/bloom
sh model_convertor_huggingface_megatron.sh \
/root/Megatron-LM        \
/mnt/workspace/bloom-ckpts/bloom7b1-hf2mg-tp1-pp1/release  \
/mnt/workspace/bloom-ckpts/bloom7b1-mg2hf-tp1-pp1  
1  \
1  \
true
```
按照上面的步骤，将megatron版本的ckpt转换到huggingface版本并存储到/mnt/workspace/bloom-ckpts/bloom7b1-mg2hf-tp1-pp1后，就可以使用huggingface来做离线文本生成推理了，具体的步骤可以参考如下一些链接：
* Huggingface通用文本生成教学：https://huggingface.co/blog/how-to-generate
* Belle文本生成示例：https://huggingface.co/BelleGroup/BELLE-7B-2M
* LLama文本生成示例：https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1