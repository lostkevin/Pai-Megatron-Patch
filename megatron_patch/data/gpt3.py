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

import hashlib
import time
import os
import random
import re
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset
from megatron import print_rank_0
from megatron.core import mpu

class AbstractDataset(ABC, Dataset):
    """GLUE base dataset class."""
    def __init__(self, data_dir, data_name, file_name, tokenizer,
                 max_seq_length):
        """
        Initializes the dataset.
        Args:
            data_dir (str): The directory containing the dataset files.
            data_name (str): The name of the dataset.
            file_name (str): The name of the dataset file.
            tokenizer (Tokenizer): The tokenizer to use for encoding the dataset.
            max_seq_length (int): The maximum sequence length for the input.
        """
        # Store inputs.
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.dataset_name = data_name
        self.samples = self.process_samples_from_single_path(
            os.path.join(data_dir, data_name, file_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]

        ids, types, paddings = self.build_tokens_types_paddings_from_text(
            raw_sample['text_a'], raw_sample['text_b'], self.tokenizer,
            self.max_seq_length)

        sample = self.build_sample(ids, types, paddings, raw_sample['label'],
                                   raw_sample['uid'])
        return sample

    @abstractmethod
    def process_samples_from_single_path(self, datapath):
        """Abstract method that takes a single path / filename and
        returns a list of dataset samples, each sample being a dict of
            {'text_a': string, 'text_b': string, 'label': int, 'uid': int}
        """
        pass

    def build_tokens_types_paddings_from_text(self, text_a, text_b, tokenizer,
                                              max_seq_length):
        """Build token types and paddings,
        trim if needed, and pad if needed."""
        text_a_ids = tokenizer.tokenize(text_a)
        text_b_ids = None
        if text_b is not None:
            text_b_ids = tokenizer.tokenize(text_b)

        return self.build_tokens_types_paddings_from_ids(
            text_a_ids, text_b_ids, max_seq_length, tokenizer.cls,
            tokenizer.sep, tokenizer.pad)

    def build_tokens_types_paddings_from_ids(self, text_a_ids, text_b_ids,
                                             max_seq_length, cls_id, sep_id,
                                             pad_id):
        """
        Builds the token types and paddings based on the input text ids,
        and trims and pads the sequences if necessary.
        Args:
            text_a_ids (list[int]): The token ids of the input text A.
            text_b_ids (list[int]): The token ids of the input text B, or None if there is no text B.
            max_seq_length (int): The maximum sequence length.
            cls_id (int): The id of the [CLS] token.
            sep_id (int): The id of the [SEP] token.
            pad_id (int): The id of the padding token.
        Returns:
            tuple: The token ids, token types, and token paddings.
        """

        ids = []
        types = []
        paddings = []

        # [CLS].
        ids.append(cls_id)
        types.append(0)
        paddings.append(1)

        # A.
        len_text_a = len(text_a_ids)
        ids.extend(text_a_ids)
        types.extend([0] * len_text_a)
        paddings.extend([1] * len_text_a)

        # [SEP].
        ids.append(sep_id)
        types.append(0)
        paddings.append(1)

        # B.
        if text_b_ids is not None:
            len_text_b = len(text_b_ids)
            ids.extend(text_b_ids)
            types.extend([1] * len_text_b)
            paddings.extend([1] * len_text_b)

        # Cap the size.
        trimmed = False
        if len(ids) >= max_seq_length:
            max_seq_length_m1 = max_seq_length - 1
            ids = ids[0:max_seq_length_m1]
            types = types[0:max_seq_length_m1]
            paddings = paddings[0:max_seq_length_m1]
            trimmed = True

        # [SEP].
        if (text_b_ids is not None) or trimmed:
            ids.append(sep_id)
            if text_b_ids is None:
                types.append(0)
            else:
                types.append(1)
            paddings.append(1)

        # Padding.
        padding_length = max_seq_length - len(ids)
        if padding_length > 0:
            ids.extend([pad_id] * padding_length)
            types.extend([pad_id] * padding_length)
            paddings.extend([0] * padding_length)

        return ids, types, paddings

    def build_sample(self, ids, types, paddings, label, unique_id):
        """
        Converts the token ids, types, paddings, label, and unique ID to a NumPy array and
        returns a sample to be consumed by the batch producer.
        Args:
            ids (list[int]): The token ids.
            types (list[int]): The token types.
            paddings (list[int]): The paddings.
            label (int): The label.
            unique_id (int): The unique ID.
        Returns:
            dict: The sample dictionary containing the token ids, types, paddings, label, and unique ID.
        """

        ids_np = np.array(ids, dtype=np.int64)
        types_np = np.array(types, dtype=np.int64)
        paddings_np = np.array(paddings, dtype=np.int64)
        sample = ({
            'text': ids_np,
            'types': types_np,
            'padding_mask': paddings_np,
            'label': int(label),
            'uid': int(unique_id)
        })

        return sample

    def clean_text(self, text):
        """
        Cleans the text by removing newlines and multiple spaces, and adjusting the end of sentence dot.
        Args:
            text (str): The text to be cleaned.
        Returns:
            str: The cleaned text.
        """

        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        for _ in range(3):
            text = text.replace(' . ', '. ')

        return text

    def truncate(self, tokenizer, array, max_length):
        """
        Truncates an array to a maximum length or pads it with zeros if its length is less than `max_length`.
        Args:
            tokenizer: The tokenizer used to encode the input.
            array: The numpy array to truncate or pad.
            max_length: The maximum length of the array.
        Returns:
            A numpy array of length `max_length` containing the contents of `array`, truncated if necessary or padded with zeros.
        """
        if len(array) < max_length:
            return np.pad(array, (0, max_length - len(array)),
                          constant_values=tokenizer.eod)
        else:
            return array[:max_length]


class GPTDataset(AbstractDataset):
    """GPT dataset class."""
    def __init__(self, datapaths, tokenizer, max_seq_length):
        """
        Initializes a new instance of the GPTDataset class.
        Args:
            datapaths (list): List of file paths containing the raw text data.
            tokenizer: Instance of the tokenizer used to tokenize the text data.
            max_seq_length (int): Maximum sequence length for input to the model.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.samples = []
        for datapath in datapaths:
            self.samples.extend(
                self.process_samples_from_single_path(datapath))
        print('  >> total number of samples: {}'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample, self.tokenizer,
                                                   self.max_seq_length)

    def clean_text(self, raw):
        """
        Cleans the input text by removing URLs, extra spaces, and special characters, and adjusting the end of sentence dot.
        Args:
            text (str): The raw text to be processed.
        Returns:
            str: The cleaned text.
        """

        httpcom = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|['
                             r'!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        raw = httpcom.sub('', raw)

        space = re.compile(r' +')
        raw = space.sub(' ', raw)

        fil = re.compile(
            u'[^0-9a-zA-Z\u4e00-\u9fa5.， ,\\-。'
            u'%《*》/•、&＆(—)（+）：？!！“”·]+', re.UNICODE)
        raw = fil.sub('', raw)
        return raw.strip()

    def process_samples_from_single_path(self, filename):
        """
        Process a single file and return a list of samples.
        """
        print(' > Processing {} ...'.format(filename))
        samples = []
        total = 0
        with open(filename, encoding='utf-8-sig') as f:
            for line in f:
                row = line.strip()
                sample = {
                    'text': row,
                }
                total += 1
                samples.append(sample)

        print(' >> processed {} samples.'.format(len(samples)))
        random.shuffle(samples)
        return samples

    def gpt_convert_example_to_feature(self, sample, tokenizer,
                                       max_seq_length):
        """
        Convert a single sample into a format suitable for GPT training.
        """
        tokens = tokenizer.tokenize(sample['text'])
        input_ids = np.array(tokens)
        input_ids = self.truncate(tokenizer, input_ids, max_seq_length)
        train_sample = {'input_ids': np.array(input_ids)}
        return train_sample

def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence lenght, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_shuffle_idx(num_samples, total_size, np_rng):
    """Build the range [0, size) and shuffle."""
    print(' > building shuffle index with split [0, {}) and [{}, {}) '
          '...'.format(num_samples, num_samples, total_size), flush=True)

    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples,
                                  step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size,
                                 step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))

def _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0:len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs-1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))

def _build_index_mappings(name, data_prefix, documents, sizes,
                          splits_string, num_samples, seq_length, seed,
                          *,
                          data_cache_path):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)

    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    desc = "GPT Dataset\n\n"
    desc += f"Data prefix {data_prefix}\n"
    desc += f"Dataset name {name}\n"
    desc += f"Number of samples {num_samples}\n"
    desc += f"Sequence length {seq_length}\n"
    desc += f"Random seed {seed}\n"
    desc += f"Split {splits_string}\n"
    desc_hash = hashlib.md5(desc.encode('utf-8')).hexdigest()
    desc_filename = desc_hash + ".dsc"
    doc_idx_filename = desc_hash + '_doc_idx.npy'
    sample_idx_filename = desc_hash + '_sample_idx.npy'
    shuffle_idx_filename = desc_hash + '_shuffle_idx.npy'

    # Look for cache in main data dir first to avoid unnecessary
    # duplication, then look in data-cache-path if specified,
    # If nothing is found, use the last path looked in
    build_indices = True
    prefixes = [os.path.join(os.path.dirname(data_prefix), 'index-cache')]
    if data_cache_path is not None:
        prefixes.append(data_cache_path)
    for prefix in prefixes:
        idx_path = {
            'desc': os.path.join(prefix, desc_filename),
            'doc': os.path.join(prefix, doc_idx_filename),
            'sample': os.path.join(prefix, sample_idx_filename),
            'shuffle': os.path.join(prefix, shuffle_idx_filename)
        }
        for f in idx_path.values():
            if not os.path.isfile(f):
                break
        else:
            # Found our files!
            build_indices = False
            break
    data_cache_dir = os.path.dirname(idx_path['desc'])
    data_cache_success = True

    # Build the indexed mapping if not exist.
    if build_indices and torch.distributed.get_rank() == 0:
        print_rank_0(' > WARNING: could not find index map files, building '
                     'the indices on rank 0 ...')

        # For the last epoch, decide whether include the entire epoch
        # in the global shuffle or not.

        # If we need only one epoch, then separating last epoch  does
        # not mean anything.
        if num_epochs == 1:
            separate_last_epoch = False
            print(' > only one epoch required, setting '
                  'separate_last_epoch to False', flush=True)

        else:
            # Get the number of samples for the last epoch
            num_samples_from_epochs_minus_one = (
                (num_epochs - 1) * tokens_per_epoch - 1) // seq_length
            last_epoch_num_samples = num_samples - \
                                     num_samples_from_epochs_minus_one
            assert last_epoch_num_samples >= 0, \
                'last epoch number of samples should be non-negative.'
            num_samples_per_epoch = (tokens_per_epoch - 1) // seq_length
            assert last_epoch_num_samples <= (num_samples_per_epoch + 1), \
                'last epoch number of samples exceeded max value.'
            # If we have less than 80% of the samples for the last epoch,
            # seperate out the epoch and treat it differently.
            # Note: the 80% number is just based on common sense and can
            # be adjusted if needed.
            separate_last_epoch = (last_epoch_num_samples <
                                   int(0.80 * num_samples_per_epoch))
            if separate_last_epoch:
                string = ' > last epoch number of samples ({}) is smaller '\
                         'than 80% of number of samples per epoch ({}), '\
                         'setting separate_last_epoch to True'
            else:
                string = ' > last epoch number of samples ({}) is larger '\
                         'than 80% of number of samples per epoch ({}), '\
                         'setting separate_last_epoch to False'
            print(string.format(last_epoch_num_samples,
                                num_samples_per_epoch), flush=True)


        try:
            os.makedirs(data_cache_dir, exist_ok=True)

            # description
            with open(idx_path['desc'], 'wt') as fd:
                fd.write(desc)

            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng,
                                     separate_last_epoch)
            np.save(idx_path['doc'], doc_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save doc-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            # First compile and then import.
            try:
                from megatron.data import helpers
            except:
                from megatron.core.datasets import helpers
            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32
            sample_idx = helpers.build_sample_idx(sizes, doc_idx, seq_length,
                                                  num_epochs, tokens_per_epoch)
            np.save(idx_path['sample'], sample_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save sample-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1
            shuffle_idx = _build_shuffle_idx(num_samples_,
                                             sample_idx.shape[0] - 1, np_rng)
            np.save(idx_path['shuffle'], shuffle_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save shuffle-idx mapping'
                         ' (seconds): {:4f}'.format(time.time() - start_time))
        except OSError:
            print(f'There was an error trying to create the data cache directory ({data_cache_dir})')
            print('or a file in it. This defaults to a directory "index-cache" within the directory')
            print('the data files are in and can be set with the --data-cache-path argument. Please')
            print('ensure you have write access to this directory or specify one that you do have')
            print('write access to.')
            data_cache_success = False

    counts = torch.cuda.LongTensor([data_cache_success])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
    if counts[0].item() != (
        torch.distributed.get_world_size() //
        torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group())):
        print_rank_0("Data index creation unsuccessful, exiting.")
        exit()

    # Load mappings.
    start_time = time.time()
    print_rank_0(f" > loading doc-idx mapping from {idx_path['doc']}")
    doc_idx = np.load(idx_path['doc'], allow_pickle=True, mmap_mode='r')

    print_rank_0(f" > loading sample-idx mapping from {idx_path['sample']}")
    sample_idx = np.load(idx_path['sample'], allow_pickle=True, mmap_mode='r')

    print_rank_0(f" > loading shuffle-idx mapping from {idx_path['shuffle']}")
    shuffle_idx = np.load(idx_path['shuffle'], allow_pickle=True, mmap_mode='r')

    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(
        sample_idx.shape[0]))
    print_rank_0('    total number of epochs: {}'.format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx, desc, desc_hash