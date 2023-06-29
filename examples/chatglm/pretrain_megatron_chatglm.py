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
import os

from megatron import get_args
from megatron.core import tensor_parallel
from megatron.utils import (average_losses_across_data_parallel_group,
                            get_ltor_masks_and_position_ids)
from megatron_patch.data.finetune_dataset import ChatGLMDataset
from megatron_patch.model.chatglm.gpt_model import GPTModel
from megatron_patch.tokenizer import build_tokenizer, get_tokenizer
from megatron_patch.training import pretrain

try:
    from megatron.model import ModelType
except ImportError:
    from megatron.core.enums import ModelType


def get_tasks_args(parser):
    group = parser.add_argument_group(title='llama')

    parser.add_argument('--local-rank', type=int, default=None,
                        help='local rank passed from distributed launcher')

    group.add_argument('--transformer-type',
                       type=str,
                       default='megatron',
                       help='transformer-type')

    group.add_argument('--pretrained-checkpoint',
                       type=str,
                       default=None,
                       help='Pretrained checkpoint used for finetunning.')

    group.add_argument('--epochs',
                       type=int,
                       default=None,
                       help='Number of finetunning epochs. Zero results in '
                            'evaluation only.')

    group.add_argument('--intermediate-size',
                       type=int,
                       default=None,
                       help='--intermediate-size')

    group.add_argument('--keep-last',
                       action='store_true',
                       help='Keep the last batch (maybe incomplete) in'
                            'the data loader')

    group.add_argument('--train-data',
                       nargs='+',
                       default=None,
                       help='Whitespace separated paths or corpora names '
                            'for training.')

    group.add_argument('--valid-data',
                       nargs='*',
                       default=None,
                       help='path(s) to the validation data.')

    group.add_argument('--extra-vocab-size',
                       type=int,
                       default=1,
                       help='--extra-vocab-size')

    group.add_argument('--max-padding-length',
                       type=int,
                       default=None,
                       help='max-padding-length')

    group.add_argument('--position-embedding-type',
                       type=str,
                       default='absolute',
                       help='Define position embedding type '
                            '("absolute"|"rotary"|"alibi"). "absolute" by default.')

    group.add_argument('--patch-tokenizer-type',
                       type=str,
                       help='patch-tokenizer-type')

    group.add_argument('--position-encoding-2d',
                       action='store_true',
                       help='position-encoding-2d')

    group.add_argument('--source-seq-len',
                       type=int,
                       default=None,
                       help='source-seq-len')

    group.add_argument('--target-seq-len',
                       type=int,
                       default=None,
                       help='target-seq-len')

    return parser


def model_provider(pre_process=True, post_process=True):
    args = get_args()
    build_tokenizer(args)
    model = GPTModel(num_tokentypes=0,
                     parallel_output=True,
                     pre_process=pre_process,
                     post_process=post_process)
    return model


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train and validation dataset."""
    args = get_args()
    tokenizer = get_tokenizer()
    train_dataset = ChatGLMDataset(args.train_data, tokenizer,
                                   args.source_seq_len, args.target_seq_len)
    valid_dataset = ChatGLMDataset(args.valid_data, tokenizer,
                                   args.source_seq_len, args.target_seq_len)
    test_dataset = ChatGLMDataset(args.valid_data, tokenizer,
                                  args.source_seq_len, args.target_seq_len)
    return train_dataset, valid_dataset, test_dataset


def forward_step(data_iterator, model):
    """Forward step."""

    try:
        data_iterator = next(data_iterator)
    except BaseException:
        data_iterator = data_iterator
    input_ids = data_iterator['input_ids'].long().cuda()
    labels = data_iterator['labels'].long().cuda()
    lm_logits = model(input_ids=input_ids)

    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def loss_func(shift_logits):
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
        averaged_loss = average_losses_across_data_parallel_group([loss])
        return loss, {'lm loss': averaged_loss[0]}

    return shift_logits, partial(loss_func)

if __name__ == '__main__':
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=get_tasks_args)
