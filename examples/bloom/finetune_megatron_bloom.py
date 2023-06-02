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

from functools import partial

import torch

from megatron import get_args
from megatron.initialize import initialize_megatron
from megatron.utils import (average_losses_across_data_parallel_group,
                            get_ltor_masks_and_position_ids)
from megatron_patch.data.finetune_dataset import BloomDataset
from megatron_patch.finetune_utils import finetune
from megatron_patch.model.bloom.gpt_model import GPTModel
from megatron_patch.tokenizer import build_tokenizer, get_tokenizer


def get_tasks_args(parser):
    group = parser.add_argument_group(title='bloom')

    group.add_argument('--transformer-type',
                       type=str,
                       default='megatron',
                       help='transformer-type')

    group.add_argument('--max-padding-length',
                       type=int,
                       default=None,
                       help='max-padding-length')

    group.add_argument('--dataset', type=str, default=None, help='task')

    group.add_argument('--pretrained-checkpoint',
                       type=str,
                       default=None,
                       help='Pretrained checkpoint used for finetunning.')

    group.add_argument('--epochs',
                       type=int,
                       default=None,
                       help='Number of finetunning epochs. Zero results in '
                       'evaluation only.')

    group.add_argument('--embed-layernorm',
                       action='store_true',
                       help='use layernorm for embedding')

    group.add_argument('--extra-vocab-size',
                       type=int,
                       default=1,
                       help='--extra-vocab-size')

    group.add_argument('--keep-last',
                       action='store_true',
                       help='Keep the last batch (maybe incomplete) in'
                       'the data loader')

    group.add_argument('--data-dir', default=None, help='data-dir')

    group.add_argument('--train-data',
                       default=None,
                       help='Whitespace separated paths or corpora names '
                       'for training.')

    group.add_argument('--valid-data',
                       default=None,
                       help='path(s) to the validation data.')

    group.add_argument('--position-embedding-type',
                       type=str,
                       default='absolute',
                       help='Define position embedding type '
                       '("absolute"|"rotary"|"alibi"). "absolute" by default.')

    group.add_argument('--glu-activation',
                       type=str,
                       help='GLU activations to use.')

    group.add_argument('--patch-tokenizer-type',
                       type=str,
                       help='patch-tokenizer-type')

    return parser


def model_provider(pre_process=True, post_process=True):
    model = GPTModel(num_tokentypes=0,
                     parallel_output=True,
                     pre_process=pre_process,
                     post_process=post_process)
    return model


def train_valid_datasets_provider():
    args = get_args()
    tokenizer = build_tokenizer(args)
    train_dataset = BloomDataset(args.train_data, tokenizer,
                                 args.max_padding_length)
    valid_dataset = BloomDataset(args.valid_data, tokenizer,
                                 args.max_padding_length)
    return train_dataset, valid_dataset


def forward_step(data_iterator, model):
    args = get_args()
    tokenizer = get_tokenizer()
    try:
        data_iterator = next(data_iterator)
    except BaseException:
        data_iterator = data_iterator

    tokens_ = data_iterator['input_ids'].long().cuda()

    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    attention_mask, loss_mask, position_ids = \
        get_ltor_masks_and_position_ids(tokens,
                                        tokenizer.eod,
                                        args.reset_position_ids,
                                        args.reset_attention_mask,
                                        args.eod_mask_loss)

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    def loss_func(loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        averaged_loss = average_losses_across_data_parallel_group([loss])
        return loss, {'lm loss': averaged_loss[0]}

    return output_tensor, partial(loss_func, loss_mask)


if __name__ == '__main__':

    initialize_megatron(extra_args_provider=get_tasks_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

    finetune(train_valid_datasets_provider=train_valid_datasets_provider,
             model_provider=model_provider,
             forward_step=forward_step)
