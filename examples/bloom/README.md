## DSW单机调试预训练任务

### 获取模型的Checkpoint
Megatron训练引擎+Megatron模型可以提供对大模型的训练加速能力，这些能力主要包括：混合精度训练，选择性激活重算，Zero优化器状态切分 ，序列并行等等。我们在PAI-Megatron-Patch中提供了Megatron版本的Bloom模型实现，以及用Megatron引擎加载Megatron版本的Bloom模型的checkpoint来进行继续预训练的流程。请联系阿里云的产品经理下载阿里云PAI平台转换后的bloom模型的checkpoint。我们提供的Megatron版的checkpoint的列表如下：

| huggingface版本的ckpt | megatron版本的ckpt |
| --- | --- |
| [https://huggingface.co/bigscience/bloomz-1b7](https://huggingface.co/bigscience/bloomz-1b7) | bloomz-1b7-to-megatron |
| [https://huggingface.co/bigscience/bloomz-7b1](https://huggingface.co/bigscience/bloomz-7b1) | bloomz-7b1-to-megatron |
| [https://huggingface.co/bigscience/bloomz](https://huggingface.co/bigscience/bloomz) | bloomz-to-megatron-tp8-pp10 |

请将Megatron版本的Bloom模型的ckpt下载到/cpfs01/user/paigpt/bloom-ckpts文件夹下面。

1. 在DSW的Terminal中进入工作目录：/cpfs01/user/paigpt/
2. 下载开源社区Megatron源代码：
```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout 07916bf24553f0d635c4083a8dd5b31755caa82b
```

3. 联系阿里云产品经理获取PAI-Megatron-Patch源代码并拷贝到工作目录/cpfs01/user/paigpt/下
4. DSW的Terminal中运行run_pretrain_megatron_bloom.sh脚本，需要传入的参数列表如下
```
ENV=$1                             # 运行环境配置：dsw,dlc
MEGATRON_PATH=$2                   # 设置开源Megatron的代码路径
MEGATRON_PATCH_PATH=$3             # 设置Megatron Patch的代码路径
MODEL_SIZE=$4                      # 模型结构参数量级：1.1B，1.7B，7.1B
BATCH_SIZE=$5                      # 每卡训练一次迭代样本数: 4, 8
GLOBAL_BATCH_SIZE=$6               # 一次迭代总样本数: 32, 64
SEQ_LEN=$7                         # 序列长度: 256, 512
LR=$8                              # 学习率: 1e-5, 5e-5
MIN_LR=$9                          # 最小学习率: 1e-6, 5e-6
PR=${10}                           # 训练精度: fp16, bf16
TP=${11}                           # 模型并行度
PP=${12}                           # 流水并行度
AC=${13}                           # 梯度检查点类型：full，sel
DO=${14}                           # 是否使用Megatron版Zero-1降显存优化器: true, false
SP=${15}                           # 是否打开序列并行加速：true，false
SAVE_INTERVAL=${16}                # 保存ckpt的间隔迭代数
DATASET_PATH=${17}                 # mmap数据集路径
PRETRAIN_CHECKPOINT_PATH=${18}     # 预训练模型路径
TRAIN_TOKENS=${19}                 # 训练token数
WARMUP_TOKENS=${20}                # 预热tokens数
OUTPUT_BASEPATH=${21}              # 训练输出文件路径
```
DSW单机运行示例如下：
```bash
export WORK_DIR=/cpfs01/user/paigpt
sh run_pretrain_megatron_bloom.sh dsw ${WORK_DIR}/Megatron-LM ${WORK_DIR}/PAI-Megatron-Patch/ 1.7B 4 32 256 1e-5 1e-6 bf16 1 1 sel true false 10000 ${WORK_DIR}/wudao/wudao_bloombpe_text_document ${WORK_DIR}/bloom-ckpts/bloomz-1b7-to-megatron 1000000 100000 ${WORK_DIR}/output_megatron_bloom/
```
#### DLC多机运行预训练任务
单机开发调试完成后，就可以在DLC环境中配置多机多卡分布式任务。使用和dsw同样的训练脚本run_pretrain_megatron_bloom.sh来运行
```
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export WORK_DIR=/cpfs01/user/paigpt
cd /cpfs01/user/paigpt/PAI-Megatron-Patch/examples/bloom
bash run_pretrain_megatron_bloom.sh
dlc
${WORK_DIR}/Megatron-LM
${WORK_DIR}/PAI-Megatron-Patch/
1.7B
4
32
256
1e-5
1e-6
bf16
1
1
sel
true
false
10000
${WORK_DIR}/wudao/wudao_bloombpe_text_document
${WORK_DIR}/bloom-ckpts/bloomz-1b7-to-megatron
10000000
1000000
${WORK_DIR}/output_megatron_bloom/
```
### 有监督微调Bloom模型
#### DSW单机微调Bloom模型

1. 在DSW的Terminal中进入工作目录：/cpfs01/user/paigpt/
2. 下载开源社区Megatron源代码：
```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout 07916bf24553f0d635c4083a8dd5b31755caa82b
```

3. 联系阿里云产品经理获取PAI-Megatron-Patch源代码并拷贝到工作目录/cpfs01/user/paigpt/下
4. DSW的Terminal中运行run_finetune_megatron_bloom.sh脚本，需要传入的参数列表如下
```
ENV=$1                          # 运行环境：dlc，dsw
MEGATRON_PATH=$2                # 设置开源Megatron的代码路径
MEGATRON_PATCH_PATH=$3          # 设置Megatron Patch的代码路径
MODEL_SIZE=$4                   # 模型结构参数量级：1.1B，1.7B，7.1B
BATCH_SIZE=$5                   # 每卡训练一次迭代样本数: 4, 8
SEQ_LEN=$6                      # 序列长度: 256, 512
LR=$7                           # 学习率: 1e-5, 5e-5
MIN_LR=$8                       # 最小学习率: 1e-6, 5e-6
PR=$9                           # 训练精度: fp16, bf16
TP=${10}                        # 模型并行度
PP=${11}                        # 流水并行度
AC=${12}                        # 梯度检查点类型：full，sel
DO=${13}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
SP=${14}                        # 是否打开序列并行：true，false
TRAIN_DATASET_PATH=${15}        # 训练数据集路径
VALID_DATASET_PATH=${16}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${17}  # 预训练模型路径
EPOCH=${18}                     # 训练迭代轮次: 16
OUTPUT_BASEPATH=${19}           # 训练输出文件路径
```
DSW单机运行示例如下：
```bash
export WORK_DIR=/cpfs01/user/paigpt
sh run_finetune_megatron_bloom.sh dsw ${WORK_DIR}/Megatron-LM ${WORK_DIR}/PAI-Megatron-Patch/ 1.7B 4 256 1e-5 1e-6 bf16 1 1 sel true false ${WORK_DIR}/instruct_latest.json ${WORK_DIR}/dev.json ${WORK_DIR}/bloom-ckpts/bloomz-1b7-to-megatron 16 ${WORK_DIR}/output_megatron_bloom/
```
#### DLC多机微调Bloom模型
单机开发调试完成后，就可以在DLC环境中配置多机多卡分布式任务。使用和dsw同样的训练脚本run_finetune_megatron_bloom.sh来运行
```
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export WORK_DIR=/cpfs01/user/paigpt
cd /cpfs01/user/paigpt/PAI-Megatron-Patch/examples/bloom
bash run_finetune_megatron_bloom.sh
dlc
${WORK_DIR}/Megatron-LM
${WORK_DIR}/PAI-Megatron-Patch/
1.7B
4
256
1e-5
1e-6
bf16
1
1
sel
true
false
${WORK_DIR}/instruct_latest.json
${WORK_DIR}/dev.json
${WORK_DIR}/bloom-ckpts/bloomz-1b7-to-megatron
16
${WORK_DIR}/output_megatron_bloom/
```
