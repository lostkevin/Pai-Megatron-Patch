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

import math
import os
import time
from bisect import bisect_right
from itertools import accumulate

import numpy as np
import torch
from torch.utils.data import Dataset

from megatron import print_rank_0
from megatron.core import mpu
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron_patch.tokenizer import get_tokenizer


class GLM130BDataset_mmap(Dataset):
    """Datasets for glm model pretraining."""
    def __init__(self, name, indexed_dataset, data_prefix, num_epochs,
                 max_num_samples, max_seq_length, generation_length,
                short_seq_prob, seed):

        self.name = name
        self.seed = seed
        self.max_seq_length = max_seq_length
        self.generation_length = generation_length

        # Dataset.
        self.indexed_dataset = indexed_dataset

        # Build the samples mapping.
        self.samples_mapping = _get_samples_mapping(
            self.indexed_dataset, data_prefix, num_epochs, max_num_samples,
            self.max_seq_length, short_seq_prob, self.seed, self.name)

        # Vocab stuff.
        self.tokenizer = get_tokenizer()
        self.mask_id = self.tokenizer.get_command('[MASK]')
        self.gmask_id = self.tokenizer.get_command('[gMASK]')
        self.pad_id = 0

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        start_idx, end_idx, seq_length = self.samples_mapping[idx]
        sample = [self.indexed_dataset[i] for i in range(start_idx, end_idx)]
        return self.build_training_sample(sample, idx)

    def build_training_sample(self, sample, idx):
        tokens = sample[0].tolist()

        def pad_to(text, max_len, pad_id):
            if len(text) > max_len:
                text = text[:max_len]
            else:
                text = text + [pad_id] * (max_len - len(text))
            return text

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

        tokens = prompt + [mask_id, sop_id] + text[:-1]
        targets = prompt + [mask_id] + text
        loss_mask = [0] * (len(prompt) + 1) + [1] * len(text)

        #tokens = pad_to(tokens, self.max_seq_length, self.pad_id)
        #targets = pad_to(targets, self.max_seq_length, self.pad_id)
        #loss_mask = pad_to(loss_mask, self.max_seq_length, self.pad_id)

        return {
            'tokens':
            np.array(tokens, dtype=np.int64),
            'targets':
            np.array(targets, dtype=np.int64),
            'position_ids':
            np.arange(0, seq_length, dtype=np.int64),
            'attention_mask': attention_mask,
            'loss_mask':
            np.array(loss_mask,
                     dtype=np.int64),
        }

class GLM130BDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_seq_length, generation_length):
        self.path = path
        self.max_seq_length = max_seq_length
        self.generation_length = generation_length
        self.dtype = np.int64
        self.tokenizer = get_tokenizer()

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
            'attention_mask': attention_mask,
            'loss_mask':
            np.array([0] * (len(prompt) + 1) + [1] * len(text),
                     dtype=np.int64),
        }

def _get_a_and_b_segments(sample, np_rng):
    """Divide sample into a and b segments."""

    # Number of sentences in the sample.
    n_sentences = len(sample)
    # Make sure we always have two sentences.
    assert n_sentences > 1, 'make sure each sample has at least two sentences.'

    # First part:
    # `a_end` is how many sentences go into the `A`.
    a_end = 1
    if n_sentences >= 3:
        # Note that randin in numpy is exclusive.
        a_end = np_rng.randint(1, n_sentences)
    tokens_a = []
    for j in range(a_end):
        tokens_a.extend(sample[j])

    # Second part:
    tokens_b = []
    for j in range(a_end, n_sentences):
        tokens_b.extend(sample[j])

    # Random next:
    is_next_random = False
    if np_rng.random() < 0.5:
        is_next_random = True
        tokens_a, tokens_b = tokens_b, tokens_a

    return tokens_a, tokens_b, is_next_random


def _truncate_tokens(tokens, max_num_tokens, np_rng):
    """Truncates tokens to a maximum sequence length."""
    assert len(tokens) > 0
    if len(tokens) <= max_num_tokens:
        return False
    while len(tokens) > max_num_tokens:
        if np_rng.random() < 0.5:
            del tokens[0]
        else:
            tokens.pop()
    return True


def _pad_and_convert_to_numpy(tokens, pad_id, max_seq_length):
    """Pad sequences and convert them to numpy."""

    # Some checks.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0

    # Tokens and token types.
    filler = [pad_id] * padding_length
    tokens_np = np.array(tokens + filler, dtype=np.int64)
    return tokens_np


def build_pretrain_glm_datasets(data_prefix, data_impl, splits_string,
                                train_valid_test_num_samples, max_seq_length,
                                source_seq_length, target_seq_length,
                                short_seq_prob, seed, skip_warmup):

    data_prefix = data_prefix[0]
    indexed_dataset = _get_indexed_dataset(data_prefix, data_impl, skip_warmup)
    # Get start and end indices of train/valid/train into doc-idx
    # Note that doc-idx is desinged to be num-docs + 1 so we can
    # easily iterate over it.
    total_num_of_documents = indexed_dataset.doc_idx.shape[0] - 1
    splits = _get_train_valid_test_split(splits_string, total_num_of_documents)

    def build_dataset(index, name):

        dataset = None
        if splits[index + 1] > splits[index]:
            # Get the pointer to the original doc-idx so we can set it later.
            doc_idx_ptr = indexed_dataset.get_doc_idx()
            # Slice the doc-idx
            start_index = splits[index]
            # Add +1 so we can index into the dataset to get the upper bound.
            end_index = splits[index + 1] + 1
            # New doc_idx view.
            indexed_dataset.set_doc_idx(doc_idx_ptr[start_index:end_index])
            # Build the dataset accordingly.
            kwargs = dict(
                name=name,
                data_prefix=data_prefix,
                num_epochs=None,
                max_num_samples=train_valid_test_num_samples[index],
                max_seq_length=max_seq_length,
                seed=seed,
            )

            dataset = GLMDataset(indexed_dataset=indexed_dataset,
                                 source_seq_length=source_seq_length,
                                 target_seq_length=target_seq_length,
                                 short_seq_prob=short_seq_prob,
                                 **kwargs)
            # Set the original pointer so dataset remains the main dataset.
            indexed_dataset.set_doc_idx(doc_idx_ptr)
            # Checks.
            assert indexed_dataset.doc_idx[0] == 0
            assert indexed_dataset.doc_idx.shape[0] == \
                   (total_num_of_documents + 1)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)

"""
def build_pretrain_glm130b_datasets(data_prefix,
                                    data_impl,
                                    splits_string,
                                    train_valid_test_num_samples,
                                    max_seq_length,
                                    generation_length,
                                    short_seq_prob,
                                    seed,
                                    skip_warmup):

    data_prefix = data_prefix[0]
    indexed_dataset = _get_indexed_dataset(data_prefix, data_impl, skip_warmup)
    # Get start and end indices of train/valid/train into doc-idx
    # Note that doc-idx is desinged to be num-docs + 1 so we can
    # easily iterate over it.
    total_num_of_documents = indexed_dataset.doc_idx.shape[0] - 1
    splits = _get_train_valid_test_split(splits_string, total_num_of_documents)

    def build_dataset(index, name):

        dataset = None
        if splits[index + 1] > splits[index]:
            # Get the pointer to the original doc-idx so we can set it later.
            doc_idx_ptr = indexed_dataset.get_doc_idx()
            # Slice the doc-idx
            start_index = splits[index]
            # Add +1 so we can index into the dataset to get the upper bound.
            end_index = splits[index + 1] + 1
            # New doc_idx view.
            indexed_dataset.set_doc_idx(doc_idx_ptr[start_index:end_index])
            # Build the dataset accordingly.
            kwargs = dict(
                name=name,
                data_prefix=data_prefix,
                num_epochs=None,
                max_num_samples=train_valid_test_num_samples[index],
                max_seq_length=max_seq_length,
                seed=seed,
            )

            dataset = GLM130BDataset(indexed_dataset=indexed_dataset,
                                 generation_length=generation_length,
                                 short_seq_prob=short_seq_prob,
                                 **kwargs)

            # Set the original pointer so dataset remains the main dataset.
            indexed_dataset.set_doc_idx(doc_idx_ptr)
            # Checks.
            assert indexed_dataset.doc_idx[0] == 0
            assert indexed_dataset.doc_idx.shape[0] == \
                   (total_num_of_documents + 1)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)
"""

def build_pretrain_glm130b_datasets(data_prefix,
                                    data_impl,
                                    splits_string,
                                    train_valid_test_num_samples,
                                    max_seq_length,
                                    generation_length,
                                    short_seq_prob,
                                    seed,
                                    skip_warmup):

    def build_dataset(index, name):

        dataset = GLM130BDataset(data_prefix[0], max_seq_length, generation_length)

        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


def _get_indexed_dataset(data_prefix, data_impl, skip_warmup):
    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)

    assert indexed_dataset.sizes.shape[0] == indexed_dataset.doc_idx[-1]

    return indexed_dataset


def _get_train_valid_test_split(splits_string, size):
    """ Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(',') != -1:
        splits = [float(s) for s in splits_string.split(',')]
    elif splits_string.find('/') != -1:
        splits = [float(s) for s in splits_string.split('/')]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] +
                            int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index


def _get_samples_mapping(indexed_dataset, data_prefix, num_epochs,
                         max_num_samples, max_seq_length, short_seq_prob, seed,
                         name):
    """Get a list that maps a sample index to a
    starting sentence index, end sentence index, and length"""
    if not num_epochs:
        if not max_num_samples:
            raise ValueError('Need to specify either max_num_samples '
                             'or num_epochs')
        num_epochs = np.iinfo(np.int32).max - 1
    if not max_num_samples:
        max_num_samples = np.iinfo(np.int64).max - 1

    # Filename of the index mapping
    indexmap_filename = data_prefix
    indexmap_filename += '_{}_indexmap'.format(name)
    if num_epochs != (np.iinfo(np.int32).max - 1):
        indexmap_filename += '_{}ep'.format(num_epochs)
    if max_num_samples != (np.iinfo(np.int64).max - 1):
        indexmap_filename += '_{}mns'.format(max_num_samples)
    indexmap_filename += '_{}msl'.format(max_seq_length)
    indexmap_filename += '_{:0.2f}ssp'.format(short_seq_prob)
    indexmap_filename += '_{}s'.format(seed)
    indexmap_filename += '.npy'

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0 and \
            not os.path.isfile(indexmap_filename):
        print_rank_0('WARNING: could not find index '
                     'map file {}, building the indices on rank 0 ...'.format(
                         indexmap_filename))

        # Make sure the types match the helpers input types.
        assert indexed_dataset.doc_idx.dtype == np.int64
        assert indexed_dataset.sizes.dtype == np.int32

        # Build samples mapping
        verbose = torch.distributed.get_rank() == 0
        start_time = time.time()
        print_rank_0('Building samples index mapping for {} ...'.format(name))
        # First compile and then import.
        from megatron.data import helpers
        samples_mapping = helpers.build_mapping(indexed_dataset.doc_idx,
                                                indexed_dataset.sizes,
                                                num_epochs, max_num_samples,
                                                max_seq_length, short_seq_prob,
                                                seed, verbose, 1)

        print('Done building samples index mapping')
        np.save(indexmap_filename, samples_mapping, allow_pickle=True)
        print_rank_0('Saved the index mapping '
                     'in {}'.format(indexmap_filename))
        # Make sure all the ranks have built the mapping
        print_rank_0(' > elasped time to build and save'
                     ' samples mapping (seconds): {:4f}'.format(time.time() -
                                                                start_time))
    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(counts,
                                 group=mpu.get_pipeline_model_parallel_group())
    assert counts[0].item() == (
        torch.distributed.get_world_size() // torch.distributed.get_world_size(
            group=mpu.get_tensor_model_parallel_group()))

    # Load indexed dataset.
    print_rank_0('Loading indexed mapping from {}'.format(indexmap_filename))
    start_time = time.time()
    samples_mapping = np.load(indexmap_filename,
                              allow_pickle=True,
                              mmap_mode='r')
    print_rank_0('Loaded indexed file in {:3.3f} seconds'.format(time.time() -
                                                                 start_time))
    print_rank_0('Total number of samples: {}'.format(
        samples_mapping.shape[0]))

    return samples_mapping
