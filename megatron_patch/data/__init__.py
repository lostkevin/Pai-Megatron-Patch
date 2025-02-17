# Copyright (c) 2025 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
try:
    from megatron import get_args
except:
    from megatron.training import get_args

from megatron_patch.tokenizer import build_tokenizer
from .llama import LLamaRawDataset
from .json_sft import SFTDataset

def build_evaluation_dataset(dataset):
    args = get_args()
    build_tokenizer(args)
    if dataset == 'LLama-SFT' or dataset == 'LLama-Pretrain-Raw':
        val_dataset = LLamaRawDataset(args.valid_data_path, args.max_padding_length)
        return val_dataset
    else:
        raise NotImplementedError('dataset {} is not implemented.'.format(dataset))

def build_finetune_dataset(dataset):
    args = get_args()
    build_tokenizer(args)
    if dataset == 'LLama-SFT':
        train_dataset = LLamaRawDataset(args.train_data_path, args.max_padding_length)
        valid_dataset = LLamaRawDataset(args.valid_data_path, args.max_padding_length)
        return train_dataset, valid_dataset
    elif dataset in [
        'LLava-SFT',
        'Qwen-VL-SFT',
        'ChatGLM-SFT',
        'Bloom-SFT',
        'Starcoder-SFT'
    ]:
        raise NotImplementedError(f"Dataset {dataset} is no longer supported in Pai-Megatron-Patch anymore, downgrade to v0.10.2 or lower to use it.")
    else:
        raise NotImplementedError('dataset {} is not implemented.'.format(dataset))

def build_pretrain_dataset_from_original(dataset):
    args = get_args()
    build_tokenizer(args)
    if dataset == 'LLama-SFT-Raw':
        train_dataset = SFTDataset(args.train_data_path, args.max_padding_length)
        return train_dataset, train_dataset, train_dataset
    elif dataset == 'LLama-Pretrain-Raw':
        # NOTE: 有较多模型微调使用该数据集，与LLama-SFT-Raw的差异在于不能使用模版
        train_dataset = LLamaRawDataset(args.train_data_path, args.max_padding_length)
        return train_dataset, train_dataset, train_dataset
    elif dataset in [
        'LLava-Pretrain-Raw',
        'ChatGLM-Pretrain-Raw',
        'Starcoder-Pretrain-Raw',
    ]:
        raise NotImplementedError(f"Dataset {dataset} is no longer supported in Pai-Megatron-Patch anymore, downgrade to v0.10.2 or lower to use it.")
    else:
        raise NotImplementedError('dataset {} is not implemented.'.format(dataset))

