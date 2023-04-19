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

import json
import os
import random
import re
from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import Dataset

from .processor import SummmaryProcessor


class AbstractDataset(ABC, Dataset):
    """GLUE base dataset class."""
    def __init__(self, data_dir, data_name, file_name, tokenizer,
                 max_seq_length):
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
        """Build token types and paddings,
         trim if needed, and pad if needed."""

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
        """Convert to numpy and return a sample
         consumed by the batch producer."""

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
        """Remove new lines and multiple spaces and
         adjust end of sentence dot."""

        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        for _ in range(3):
            text = text.replace(' . ', '. ')

        return text

    def truncate(self, tokenizer, array, max_length):
        if len(array) < max_length:
            return np.pad(array, (0, max_length - len(array)),
                          constant_values=tokenizer.eod)
        else:
            return array[:max_length]


class GPTDataset(AbstractDataset):
    """GLUE base dataset class."""
    def __init__(self, datapaths, tokenizer, max_seq_length):
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
        httpcom = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|['
                             r'!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  # 匹配模式
        raw = httpcom.sub('', raw)

        space = re.compile(r' +')
        raw = space.sub(' ', raw)

        fil = re.compile(
            u'[^0-9a-zA-Z\u4e00-\u9fa5.， ,\\-。'
            u'%《*》/•、&＆(—)（+）：？!！“”·]+', re.UNICODE)
        raw = fil.sub('', raw)
        return raw.strip()

    def process_samples_from_single_path(self, filename):
        """"Implement abstract method."""
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
        tokens = tokenizer.tokenize(sample['text'])
        input_ids = np.array(tokens)
        input_ids = self.truncate(tokenizer, input_ids, max_seq_length)
        train_sample = {'input_ids': np.array(input_ids)}
        return train_sample


class BloomDataset(GPTDataset):
    def __init__(self, datapaths, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.prompt = ''
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

    def process_samples_from_single_path(self, filename):
        print(' > Processing {} ...'.format(filename))
        samples = []
        total = 0
        with open(filename, encoding='utf-8-sig') as f:
            for example in f:
                text = json.loads(example)['text']
                # prompt = text.split("\n")[0]
                # answer = text.replace(prompt, "").strip()
                sample = {
                    'prompt':
                    text + '</s>' if not text.endswith('</s>') else text,
                    'answer': text,
                }
                total += 1
                samples.append(sample)

        print(' >> processed {} samples.'.format(len(samples)))
        random.shuffle(samples)
        return samples

    def gpt_convert_example_to_feature(self, sample, tokenizer,
                                       max_seq_length):
        tokens = tokenizer(sample['prompt'])
        input_ids = tokens['input_ids']
        input_ids = self.truncate(tokenizer, input_ids, max_seq_length)
        train_sample = {'input_ids': np.array(input_ids)}
        return train_sample


class ChatGLMDataset(GPTDataset):
    def __init__(self, datapaths, tokenizer, max_source_length,
                 max_target_length):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.samples = []
        for datapath in datapaths:
            self.samples.extend(
                self.process_samples_from_single_path(datapath))
        print('  >> total number of samples: {}'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample, self.tokenizer)

    def process_samples_from_single_path(self, filename):
        print(' > Processing {} ...'.format(filename))
        samples = []
        total = 0
        with open(filename, encoding='utf-8-sig') as f:
            for example in f:
                text = json.loads(example)['text']
                prompt = text.split('\n')[0]
                answer = text.replace(prompt, '').strip()
                sample = {
                    'source': prompt,
                    'target': answer,
                }
                total += 1
                samples.append(sample)

        print(' >> processed {} samples.'.format(len(samples)))
        random.shuffle(samples)
        return samples

    def gpt_convert_example_to_feature(self, sample, tokenizer):

        prompt, answer = sample['source'], sample['target']
        a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
        b_ids = tokenizer.encode(text=answer, add_special_tokens=False)
        max_seq_length = self.max_source_length + self.max_target_length
        if len(a_ids) > self.max_source_length - 1:
            a_ids = a_ids[:self.max_source_length - 1]

        if len(b_ids) > self.max_target_length - 2:
            b_ids = b_ids[:self.max_target_length - 2]

        input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
        context_length = input_ids.index(tokenizer.bos_token_id)
        mask_position = context_length - 1
        labels = [-100] * context_length + input_ids[mask_position + 1:]

        pad_len = max_seq_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [tokenizer.pad_token_id] * pad_len
        labels = [(label if label != tokenizer.pad_token_id else -100)
                  for label in labels]

        train_sample = {
            'input_ids': np.array(input_ids, dtype=np.int64),
            'labels': np.array(labels, dtype=np.int64)
        }

        return train_sample


class GLMDataset(GPTDataset):
    def __init__(self, datapaths, tokenizer, max_source_seq_length,
                 max_target_seq_length):
        self.tokenizer = tokenizer
        self.prompt = ''
        self.samples = []
        self.random = random.Random(1234)
        self.blank_maskratio = 0.1
        self.max_src_length, self.max_tgt_length =\
            max_source_seq_length, max_target_seq_length
        for datapath in datapaths:
            self.samples.extend(
                self.process_samples_from_single_path(datapath))
        print('  >> total number of samples: {}'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample, self.tokenizer)

    def process_samples_from_single_path(self, filename):
        print(' > Processing {} ...'.format(filename))
        samples = []
        total = 0
        with open(filename, encoding='utf-8-sig') as f:
            for example in f:
                json_obj = json.loads(example)
                prompt = json_obj['question']
                answer = json_obj['answer']
                source_tokenized_text = self.tokenizer._tokenize(prompt)
                target_tokenized_text = self.tokenizer._tokenize(answer)
                sample = {
                    'source': ' '.join(source_tokenized_text),
                    'target': ' '.join(target_tokenized_text),
                }
                total += 1
                samples.append(sample)

        print(' >> processed {} samples.'.format(len(samples)))
        random.shuffle(samples)
        return samples

    def mask_text(self, text):
        tokens = text.split()
        mask_ratio = self.blank_maskratio
        n = len(tokens)
        indices = sorted(self.random.sample(range(n), int(n * mask_ratio)))
        masked_src, masked_tgt = '', []
        for i, idx in enumerate(indices):
            if i == 0 or idx != indices[i - 1] + 1:
                masked_tgt.append('')
            masked_tgt[-1] += ' ' + tokens[idx]
            tokens[idx] = '[MASK]'
        for i, token in enumerate(tokens):
            if i != 0 and token == '[MASK]' and tokens[i - 1] == '[MASK]':
                continue
            masked_src += ' ' + token
        return masked_src, masked_tgt

    def gpt_convert_example_to_feature(self, sample, tokenizer):
        # GLM BlankLMDataset
        source_text = sample['target']
        mask_id = tokenizer.mask_token_id
        sop_id = tokenizer.cls_token_id
        eop_id = tokenizer.eop_token_id
        pad_id = tokenizer.pad_token_id
        masked_src, masked_tgt = self.mask_text(source_text)
        source_text = masked_src

        def pad_to(text, max_len, pad_id):
            if len(text) > max_len:
                text = text[:max_len]
            else:
                text = text + [pad_id] * (max_len - len(text))
            return text

        source_tokens = tokenizer.convert_tokens_to_ids(source_text.split())
        source_tokens = pad_to(source_tokens, self.max_src_length, pad_id)
        sep = len(source_tokens)
        position_ids = list(range(len(source_tokens)))
        block_position_ids = [0] * len(source_tokens)
        mask_positions = [
            i for i, x in enumerate(source_tokens) if x == mask_id
        ]
        assert len(mask_positions) <= len(masked_tgt)
        tokens = source_tokens
        target_ids = [0] * len(source_tokens)
        loss_mask = [0] * len(source_tokens)
        for i, mask_pos in enumerate(mask_positions):
            tgt_text = masked_tgt[i]
            tgt_tokens = tokenizer.convert_tokens_to_ids(tgt_text.split())
            tokens += [sop_id] + tgt_tokens
            target_ids += tgt_tokens + [eop_id]
            loss_mask += [1] * (len(tgt_tokens) + 1)
            position_ids += [mask_pos] * (len(tgt_tokens) + 1)
            block_position_ids += [i + 1 for i in range(len(tgt_tokens) + 1)]

        max_length = self.max_src_length + int(
            self.max_src_length * self.blank_maskratio)

        tokens = pad_to(tokens, max_length, pad_id)
        target_ids = pad_to(target_ids, max_length, pad_id)
        loss_mask = pad_to(loss_mask, max_length, 0)
        position_ids = pad_to(position_ids, max_length, 0)
        block_position_ids = pad_to(block_position_ids, max_length, 0)
        position_ids = [position_ids, block_position_ids]
        train_sample = {
            'text': np.array(tokens, dtype=np.int64),
            'target': np.array(target_ids, dtype=np.int64),
            'attention_mask': np.array(sep, dtype=np.int64),
            'loss_mask': np.array(loss_mask, dtype=np.int64),
            'position_id': np.array(position_ids, dtype=np.int64)
        }

        return train_sample


class GLMSeq2SeqDataset(GPTDataset):
    def __init__(self, data_dir, task, tokenizer, max_source_seq_length,
                 max_target_seq_length):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.task = task
        self.max_src_length, self.max_tgt_length =\
            max_source_seq_length, max_target_seq_length
        from megatron import get_args
        args = get_args()
        if args.patch_tokenizer_type == 'GLMGPT2BPETokenizer':
            self.cls_id = self.tokenizer.get_command('ENC').Id
            self.mask_token = 'sMASK'
            self.mask_id = self.tokenizer.get_command(self.mask_token).Id
            self.pad_id = self.tokenizer.get_command('pad').Id
            self.sop_id = self.tokenizer.get_command('sop').Id
            self.eop_id = self.tokenizer.get_command('eop').Id

        elif args.patch_tokenizer_type == 'IcetkGLM130BTokenizer':
            self.mask_token = '[sMASK]'
            self.cls_id = self.tokenizer.get_command('ENC')
            self.mask_id = self.tokenizer.get_command(self.mask_token)
            self.pad_id = 0
            self.sop_id = self.tokenizer.get_command('sop')
            self.eop_id = self.tokenizer.get_command('eop')

        self.split = 'train'
        if self.task in ['gigaword', 'cnn_dm', 'cnn_dm_original']:
            self.processor = SummmaryProcessor(self.task, self.data_dir,
                                               tokenizer)
        self.samples = self.processor.create_examples('train')
        print('  >> total number of samples: {}'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample, self.tokenizer)

    def gpt_convert_example_to_feature(self, sample, tokenizer):
        source_text, target_text = sample['text_a'], sample['text_b']
        source_tokens = self.tokenizer.EncodeAsIds(' ' +
                                                   source_text).tokenization
        prompt = [self.cls_id, self.mask_id
                  ] + self.tokenizer.EncodeAsIds(' Content:').tokenization
        if len(source_tokens) > self.max_src_length - len(prompt):
            source_tokens = source_tokens[:self.max_src_length - len(prompt)]
        source_tokens = prompt + source_tokens

        if len(source_tokens) < self.max_src_length:
            source_tokens = source_tokens + [self.pad_id] * (
                self.max_src_length - len(source_tokens))
        sep = len(source_tokens)
        position_ids = list(range(len(source_tokens)))
        block_position_ids = [0] * len(source_tokens)
        mask_pos = source_tokens.index(self.mask_id)
        if self.split == 'train':
            target_tokens = self.tokenizer.EncodeAsIds(
                ' ' + target_text).tokenization
            target_tokens = target_tokens + [self.eop_id]
            if len(target_tokens) > self.max_tgt_length:
                target_tokens = target_tokens[:self.max_tgt_length]
                target_truncated = True
            loss_mask = [1] * len(target_tokens)
            if len(target_tokens) < self.max_tgt_length:
                loss_mask += [0] * (self.max_tgt_length - len(target_tokens))
                target_tokens += [self.pad_id] * (self.max_tgt_length -
                                                  len(target_tokens))
            tokens = source_tokens + [self.sop_id] + target_tokens[:-1]
            loss_mask = [0] * len(source_tokens) + loss_mask
            target_ids = [0] * len(source_tokens) + target_tokens
            position_ids += [mask_pos] * len(target_tokens)
            block_position_ids += list(range(1, len(target_tokens) + 1))
            position_ids = [position_ids, block_position_ids]
            train_sample = {
                'text': np.array(tokens, dtype=np.int64),
                'target': np.array(target_ids, dtype=np.int64),
                'attention_mask': np.array(sep, dtype=np.int64),
                'loss_mask': np.array(loss_mask, dtype=np.int64),
                'position_id': np.array(position_ids, dtype=np.int64)
            }

            return train_sample


class Seq2SeqDataset_old(GPTDataset):
    def __init__(self, args, split, tokenizer):
        self.args = args
        self.task, self.data_dir = args.task.lower(), args.data_dir
        self.max_src_length, self.max_tgt_length = args.source_seq_len, args.target_seq_len
        self.split = split
        self.tokenizer = tokenizer
        self.dataset_name = split
        if self.task in ['gigaword', 'cnn_dm', 'cnn_dm_original']:
            self.processor = SummmaryProcessor(self.task, self.data_dir,
                                               tokenizer)

        example_list = self.processor.create_examples(split)
        self.example_list = example_list
        self.examples = {example['guid']: example for example in example_list}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        example = self.example_list[idx]
        cls_id = self.tokenizer.get_command('ENC').Id
        mask_token = 'sMASK' if self.args.task_mask else 'MASK'
        mask_id = self.tokenizer.get_command(mask_token).Id
        pad_id = self.tokenizer.get_command('pad').Id
        sop_id = self.tokenizer.get_command('sop').Id
        eop_id = self.tokenizer.get_command('eop').Id
        if self.task in ['gigaword', 'cnn_dm', 'cnn_dm_original', 'xsum']:
            source_text, target_text = example.text_a, example.text_b
            source_tokens = self.tokenizer.EncodeAsIds(
                ' ' + source_text).tokenization
            prompt = [cls_id, mask_id
                      ] + self.tokenizer.EncodeAsIds(' Content:').tokenization
            if len(source_tokens) > self.max_src_length - len(prompt):
                source_tokens = source_tokens[:self.max_src_length -
                                              len(prompt)]
            source_tokens = prompt + source_tokens
        elif self.task == 'squad_generation':
            source_text = example.text_a
            target_text, answer = example.meta['question'], example.meta[
                'answer']
            source_tokens = self.tokenizer.EncodeAsIds(
                source_text.rstrip() + ' Question:').tokenization
            answer_tokens = self.tokenizer.EncodeAsIds(' Answer: ' +
                                                       answer).tokenization
            if len(source_tokens
                   ) > self.max_src_length - len(answer_tokens) - 2:
                max_src_length = self.max_src_length - len(answer_tokens) - 2
                answer_pattern = self.tokenizer.EncodeAsIds(
                    ' ' + answer).tokenization

                def sub_finder(mylist, pattern):
                    matches = []
                    for i in range(len(mylist)):
                        if mylist[i] == pattern[0] and mylist[
                                i:i + len(pattern)] == pattern:
                            matches.append(i)
                    return matches

                answer_indices = sub_finder(source_tokens, answer_pattern)
                if len(answer_indices) == 0:
                    print(f'Answer {answer} not exists in the source text')
                    source_tokens = source_tokens[:max_src_length]
                else:
                    start_index = max(answer_indices[0] - max_src_length // 2,
                                      0)
                    source_tokens = source_tokens[start_index:start_index +
                                                  max_src_length]
            source_tokens = [cls_id] + source_tokens + [mask_id
                                                        ] + answer_tokens
        elif self.task in ['squad', 'squad_v1']:
            source_text = example.text_a
            target_text = example.meta['answer'].strip()
            question = example.meta['question'].strip()
            source_tokens = self.tokenizer.EncodeAsIds(
                ' ' + source_text.rstrip()).tokenization
            question_tokens = self.tokenizer.EncodeAsIds(' ' +
                                                         question).tokenization
            period_id = self.tokenizer.TokenToId('.')
            max_src_length = self.max_src_length - len(question_tokens) - 3
            if max_src_length <= 0:
                print(question)
            assert max_src_length > 0
            source_tokens = [cls_id] + question_tokens + [
                mask_id, period_id
            ] + source_tokens[:max_src_length]
        elif self.task in ['cmrc']:
            mask_id = self.tokenizer.get_command('MASK').Id
            source_text = example.text_a
            target_text = example.meta['answer'].strip()
            question = example.meta['question'].strip()
            source_tokens = self.tokenizer.EncodeAsIds(
                source_text.rstrip()).tokenization
            question_tokens = self.tokenizer.EncodeAsIds('问题：' + question +
                                                         '答案：').tokenization
            max_src_length = self.max_src_length - len(question_tokens) - 2
            if max_src_length <= 0:
                print(question)
                question_tokens = question_tokens[self.max_src_length // 4]
            source_tokens = [cls_id] + question_tokens + [
                mask_id
            ] + source_tokens[:max_src_length]
        else:
            raise NotImplementedError
        if len(source_tokens) < self.max_src_length:
            source_tokens = source_tokens + [pad_id] * (self.max_src_length -
                                                        len(source_tokens))
        sep = len(source_tokens)
        position_ids = list(range(len(source_tokens)))
        block_position_ids = [0] * len(source_tokens)
        mask_pos = source_tokens.index(mask_id)
        if self.split == 'train':
            target_tokens = self.tokenizer.EncodeAsIds(
                ' ' + target_text).tokenization
            target_tokens = target_tokens + [eop_id]
            if len(target_tokens) > self.max_tgt_length:
                target_tokens = target_tokens[:self.max_tgt_length]
                target_truncated = True
            loss_mask = [1] * len(target_tokens)
            if len(target_tokens) < self.max_tgt_length:
                loss_mask += [0] * (self.max_tgt_length - len(target_tokens))
                target_tokens += [pad_id] * (self.max_tgt_length -
                                             len(target_tokens))
            tokens = source_tokens + [sop_id] + target_tokens[:-1]
            loss_mask = [0] * len(source_tokens) + loss_mask
            target_ids = [0] * len(source_tokens) + target_tokens
            position_ids += [mask_pos] * len(target_tokens)
            if self.args.no_block_position:
                block_position_ids += [1] * len(target_tokens)
            else:
                block_position_ids += list(range(1, len(target_tokens) + 1))
            position_ids = [position_ids, block_position_ids]
            import pdb
            pdb.set_trace()
            sample = {
                'text': np.array(tokens, dtype=np.int64),
                'target': np.array(target_ids, dtype=np.int64),
                'attention_mask': np.array(sep, dtype=np.int64),
                'loss_mask': np.array(loss_mask, dtype=np.int64),
                'position_id': np.array(position_ids, dtype=np.int64),
                'uid': example.guid
            }
        else:
            tokens = source_tokens + [sop_id]
            position_ids = position_ids + [mask_pos]
            block_position_ids = block_position_ids + [1]
            position_ids = [position_ids, block_position_ids]
            sample = {
                'text': np.array(tokens, dtype=np.int64),
                'attention_mask': np.array(sep, dtype=np.int64),
                'position_id': np.array(position_ids, dtype=np.int64),
                'uid': example.guid
            }
        return sample
