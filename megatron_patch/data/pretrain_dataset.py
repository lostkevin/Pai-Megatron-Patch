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

from megatron import print_rank_0
from megatron.data.gpt_dataset import (_build_index_mappings,
                                       get_indexed_dataset_,
                                       get_train_valid_test_split_)
from megatron_patch.tokenizer import get_tokenizer


class GLM130BDataset_Ori(torch.utils.data.Dataset):
    def __init__(self, path, max_seq_length, generation_length):
        self.path = path
        self.max_seq_length = max_seq_length
        self.generation_length = generation_length
        self.dtype = np.int64
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
            attention_mask,
            'loss_mask':
            np.array([0] * (len(prompt) + 1) + [1] * len(text),
                     dtype=np.int64),
        }


class GLM130BDataset_IdxMap(torch.utils.data.Dataset):
    def __init__(self,
                 name,
                 data_prefix,
                 documents,
                 indexed_dataset,
                 num_samples,
                 seq_length,
                 generation_length,
                 seed,
                 return_doc_ids=False):

        self.max_seq_length = seq_length
        self.generation_length = generation_length
        self.tokenizer = get_tokenizer()
        self.mask_id = self.tokenizer.get_command('[MASK]')
        self.gmask_id = self.tokenizer.get_command('[gMASK]')

        self.name = name
        self.indexed_dataset = indexed_dataset
        self.return_doc_ids = return_doc_ids

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx, self.index_prefix = \
            _build_index_mappings(self.name, data_prefix,
                                  documents, self.indexed_dataset.sizes,
                                  num_samples, seq_length, seed)

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        doc_ids = []
        if doc_index_f == doc_index_l:
            doc_ids.append(self.doc_idx[doc_index_f])
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            doc_ids.append(self.doc_idx[doc_index_f])
            sample_list = [
                self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                         offset=offset_f)
            ]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                doc_ids.append(self.doc_idx[i])
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            doc_ids.append(self.doc_idx[doc_index_l])
            sample_list.append(
                self.indexed_dataset.get(self.doc_idx[doc_index_l],
                                         length=offset_l + 1))
            sample = np.concatenate(sample_list)

        tokens = sample[:-2].tolist()
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
            attention_mask,
            'loss_mask':
            np.array([0] * (len(prompt) + 1) + [1] * len(text),
                     dtype=np.int64),
        }


class AlpacaDataset_Ori(torch.utils.data.Dataset):
    def __init__(self, datapath, max_padding_length):
        self.IGNORE_INDEX = -100
        self.tokenizer = get_tokenizer()
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

        list_data_dict = self.jload(datapath)
        prompt_input, prompt_no_input = PROMPT_DICT[
            'prompt_input'], PROMPT_DICT['prompt_no_input']
        sources = [
            prompt_input.format_map(example) if example.get('input', '') != ''
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{self.tokenizer.eos_token}"
            for example in list_data_dict
        ]
        data_dict = self.preprocess(sources, targets, self.tokenizer)

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
        attention_mask = np.ones(input_ids.shape, dtype=np.int64)
        attention_mask[attention_mask == self.tokenizer.pad_token_id] = 0
        train_sample = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }

        return train_sample


def build_pretrain_glm130b_datasets_from_idxmap(data_prefix,
                                                data_impl,
                                                splits_string,
                                                train_valid_test_num_samples,
                                                seq_length,
                                                generation_length,
                                                seed,
                                                skip_warmup,
                                                return_doc_ids=False):
    """Build train, valid, and test datasets."""
    data_prefix = data_prefix[0]
    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix, data_impl, skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))

    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index],
                                  stop=splits[index + 1],
                                  step=1,
                                  dtype=np.int32)
            dataset = GLM130BDataset_IdxMap(
                name, data_prefix, documents, indexed_dataset,
                train_valid_test_num_samples[index], seq_length,
                generation_length, seed, return_doc_ids)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


def build_pretrain_glm130b_datasets_from_original(data_prefix, max_seq_length,
                                                  generation_length):
    def build_dataset():

        dataset = GLM130BDataset_Ori(data_prefix[0], max_seq_length,
                                     generation_length)

        return dataset

    train_dataset = build_dataset()
    valid_dataset = build_dataset()
    test_dataset = build_dataset()

    return (train_dataset, valid_dataset, test_dataset)


def build_pretrain_alpaca_datasets_from_original(data_prefix,
                                                 max_padding_length):
    def build_dataset():

        dataset = AlpacaDataset_Ori(data_prefix[0], max_padding_length)

        return dataset

    train_dataset = build_dataset()
    valid_dataset = build_dataset()
    test_dataset = build_dataset()

    return (train_dataset, valid_dataset, test_dataset)
