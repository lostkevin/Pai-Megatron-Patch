## PAI-Megatron-Patch
PAI-Megatron-Patch是一款对开源Megatron框架进行非代码侵入设计的补丁工具，方便用户低门槛的使用Megatron引擎来训练Bloom，GLM等大模型训练。

### 安装指南
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout 07916bf24553f0d635c4083a8dd5b31755caa82b


### 测试Bloom模型
sh run_finetune_megatron_bloom.sh 1.7B 4 256 1e-5 1e-6 bf16 1 sel z1 false chat /mnt/ChatGPT/instruct.json /mnt/ChatGPT/instruct.json /mnt/bloom-ckpts/bloomz-1b7-to-megatron/ 6
