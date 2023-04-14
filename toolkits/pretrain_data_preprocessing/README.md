## 数据预处理
建议在灵骏智算平台中的DSW实例中准备预训练数据，以下是针对wudao2.0数据集的准备流程：

1. 下载WuDaoCorpora2.0开源数据集到/cpfs01/user/paigpt工作目录下，文件夹命名为WuDaoCorpus2.0_base_200G

2. 对Wudao数据执行数据集清洗并打包成zst压缩文件格式。具体流程可参考如下的bash脚本。
```bash
#! /bin/bash

cd /cpfs01/user/paigpt/PAI-Megatron-Patch/toolkits/pretrain_data_preprocessing

#清洗数据
python clean_raw_text.py -i WuDaoCorpus2.0_base_200G  -o cleaned_wudao_dataset -p 32

#合并清洗后的数据，并分块
mkdir wudao
find cleaned_wudao_dataset -name "*.json" -exec cat {} + > wudao/merged_wudao_cleaned.json
split -l 6000000 --numeric-suffixes --additional-suffix=.jsonl /cpfs01/user/paigpt/wudao/merged_wudao_cleaned.json /cpfs01/user/paigpt/wudao/


#数据压缩
mkdir /cpfs01/user/paigpt/wudao/cleaned_zst
zstd -z /cpfs01/user/paigpt/wudao/00.jsonl -o /cpfs01/user/paigpt/wudao/cleaned_zst/00.jsonl.zst &
zstd -z /cpfs01/user/paigpt/wudao/01.jsonl -o /cpfs01/user/paigpt/wudao/cleaned_zst/01.jsonl.zst &
zstd -z /cpfs01/user/paigpt/wudao/02.jsonl -o /cpfs01/user/paigpt/wudao/cleaned_zst/02.jsonl.zst &
zstd -z /cpfs01/user/paigpt/wudao/03.jsonl -o /cpfs01/user/paigpt/wudao/cleaned_zst/03.jsonl.zst &
zstd -z /cpfs01/user/paigpt/wudao/04.jsonl -o /cpfs01/user/paigpt/wudao/cleaned_zst/04.jsonl.zst &
zstd -z /cpfs01/user/paigpt/wudao/05.jsonl -o /cpfs01/user/paigpt/wudao/cleaned_zst/05.jsonl.zst &
zstd -z /cpfs01/user/paigpt/wudao/06.jsonl -o /cpfs01/user/paigpt/wudao/cleaned_zst/06.jsonl.zst &
zstd -z /cpfs01/user/paigpt/wudao/07.jsonl -o /cpfs01/user/paigpt/wudao/cleaned_zst/07.jsonl.zst &
zstd -z /cpfs01/user/paigpt/wudao/08.jsonl -o /cpfs01/user/paigpt/wudao/cleaned_zst/08.jsonl.zst &
zstd -z /cpfs01/user/paigpt/wudao/09.jsonl -o /cpfs01/user/paigpt/wudao/cleaned_zst/09.jsonl.zst &
```

3. 制作MMAP格式预训练数据集。

在DSW的Terminal中进入代码目录：/cpfs01/user/paigpt/PAI-Megatron-Patch/toolkits/pretrain_data_preprocessing。查看run_make_pretraining_dataset.sh脚本内容。里面有三个启动参数需要在运行时输入，分别是：input_data_dir，tokenizer和output_data_dir。input_data_dir就是上面打包后的wudao数据集的文件夹路径/cpfs01/user/paigpt/wudao/cleaned_zst，tokenizer选择bloombpe，output_data_dir设置为/cpfs01/user/paigpt/wudao。执行如下命令
```bash
bash run_make_pretraining_dataset.sh /cpfs01/user/paigpt/wudao/cleaned_zst bloombpe /cpfs01/user/paigpt/wudao
```




