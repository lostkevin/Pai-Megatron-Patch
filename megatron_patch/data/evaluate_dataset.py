# Copyright (c) 2023 Alibaba PAI Team.
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

import copy
import io
import json
import math
import os
from bisect import bisect_right
from itertools import accumulate

import numpy as np
import torch

from megatron import get_args
from megatron_patch.tokenizer import get_tokenizer


class GLM130BDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, max_seq_length, generation_length):
        self.path = path
        self.max_seq_length = max_seq_length
        self.generation_length = generation_length
        self.dtype = np.int64
        self.tokenizer = tokenizer

        self.tokenizer = get_tokenizer()
        self.mask_id = self.tokenizer.get_command('[MASK]')
        self.gmask_id = self.tokenizer.get_command('[gMASK]')
        self.data = []
        self.process_single_file(self.path)

    def process_single_file(self, path):
        num_sequences = []
        with open(os.path.join(path), 'r', encoding='utf-8') as file:
            raw_text = file.read()
            tokens = self.tokenizer.tokenize(raw_text)
            self.num_tokenized_tokens = len(tokens)
            self.num_original_tokens = len(raw_text.strip().split(' '))
            self.data.append({
                'raw_text':
                tokens,
                'num_original_tokens':
                len(raw_text.strip().split(' ')),
                'num_sequences':
                max(
                    math.ceil(
                        max(len(tokens) - (self.max_seq_length - 1), 0) /
                        self.generation_length) + 1,
                    1,
                ),
            })
            num_sequences.append(self.data[-1]['num_sequences'])
        self.weights = list(accumulate(num_sequences))
        self.left_weights = [0] + self.weights[:-1]

    def __len__(self):
        return self.data[0]['num_sequences']

    def __getitem__(self, idx):
        document_idx = bisect_right(self.weights, idx)
        idx = idx - self.left_weights[document_idx]
        start_idx = idx * self.generation_length
        end_idx = start_idx + self.max_seq_length - 1  # for additional [gMASK]
        tokens = self.data[document_idx]['raw_text'][start_idx:end_idx]

        mask_id = self.gmask_id
        sop_id = self.tokenizer.get_command('sop')

        if idx == 0:
            prompt, text = [], tokens
        else:
            prompt_length = self.max_seq_length - 1 - self.generation_length
            prompt, text = tokens[:prompt_length], tokens[prompt_length:]

        seq_length = len(prompt) + len(text) + 1
        attention_mask = np.tril(
            np.ones((seq_length, seq_length), dtype=np.int64))
        attention_mask[:len(prompt) + 1, :len(prompt) + 1] = 1
        return {
            'tokens':
            np.array(prompt + [mask_id, sop_id] + text[:-1], dtype=np.int64),
            'targets':
            np.array(prompt + [mask_id] + text, dtype=np.int64),
            'position_ids':
            np.arange(0, seq_length, dtype=np.int64),
            'attention_mask':
            attention_mask < 0.5,
            'loss_mask':
            np.array([0] * (len(prompt) + 1) + [1] * len(text),
                     dtype=np.int64),
        }


class AlpacaDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, max_padding_length):
        self.IGNORE_INDEX = -100
        self.tokenizer = tokenizer
        self.max_padding_length = max_padding_length
        PROMPT_DICT = {
            'prompt_input':
            ('Below is an instruction that describes a task,'
             ' paired with an input that provides further context. '
             'Write a response that appropriately completes the request.\n\n'
             '### Instruction:\n{instruction}'
             '\n\n### Input:\n{input}\n\n### Response:'),
            'prompt_no_input':
            ('Below is an instruction that describes a task. '
             'Write a response that appropriately completes the request.\n\n'
             '### Instruction:\n{instruction}\n\n### Response:'),
        }

        list_data_dict = self.jload(path)
        prompt_input, prompt_no_input = PROMPT_DICT[
            'prompt_input'], PROMPT_DICT['prompt_no_input']

        sources = [
            prompt_input.format_map(example) if example.get('input', '') != ''
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}"
            for example in list_data_dict
        ]

        data_dict = self.preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict['input_ids']
        self.labels = data_dict['labels']
        self.samples = []
        for inputs, labels in zip(self.input_ids, self.labels):
            self.samples.append([inputs, labels])

        print('  >> total number of samples: {}'.format(len(self.samples)))

    def _make_r_io_base(self, f, mode: str):
        if not isinstance(f, io.IOBase):
            f = open(f, mode=mode)
        return f

    def jload(self, f, mode='r'):
        """Load a .json file into a dictionary."""
        f = self._make_r_io_base(f, mode)
        jdict = json.load(f)
        f.close()
        return jdict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idx = 0
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample)

    def preprocess(self, sources, targets, tokenizer):
        """Preprocess the data by tokenizing."""
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [
            self.tokenize(strings, tokenizer)
            for strings in (examples, sources)
        ]
        input_ids = examples_tokenized['input_ids']
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels,
                                     sources_tokenized['input_ids_lens']):
            label[:source_len] = self.IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    def tokenize(self, strings, tokenizer):
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors='np',
                padding='max_length',
                max_length=self.max_padding_length,
                truncation=True,
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            (tokenized.input_ids != tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def gpt_convert_example_to_feature(self, sample):
        input_ids, labels = sample
        loss_mask = np.ones(labels.shape, dtype=np.int64)
        loss_mask[labels == self.IGNORE_INDEX] = 0
        loss_mask[labels == self.tokenizer.pad_token_id] = 0
        train_sample = {
            'input_ids': input_ids,
            'labels': labels,
            'loss_mask': loss_mask
        }

        return train_sample


def build_evaluation_dataset(task):
    """Helper function to select and build dataset."""
    args = get_args()
    tokenizer = get_tokenizer()

    if task == 'WIKITEXT103-GLM130B':
        val_dataset = GLM130BDataset(args.data_path[0], tokenizer,
                                     args.seq_length, args.generation_length)
        return val_dataset

    elif task == 'Alpaca-7B':
        val_dataset = AlpacaDataset(args.data_path[0], tokenizer,
                                    args.max_padding_length)
        return val_dataset

    raise NotImplementedError('dataset for {} task is not '
                              'implemented.'.format(task))
