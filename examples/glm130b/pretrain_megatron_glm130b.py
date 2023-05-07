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
from megatron.core import tensor_parallel
from megatron.utils import average_losses_across_data_parallel_group
from megatron_patch.data.pretrain_dataset import \
    build_pretrain_glm130b_datasets
from megatron_patch.model.glm130b.gpt_model import GPTModel
from megatron_patch.tokenizer import build_tokenizer, get_tokenizer
from megatron_patch.training import pretrain

try:
    from megatron.model import ModelType
except ImportError:
    from megatron.core.enums import ModelType


def get_tasks_args(parser):
    group = parser.add_argument_group(title='glm')

    group.add_argument('--transformer-type',
                       type=str,
                       default='megatron',
                       help='transformer-type')

    group.add_argument('--generation-length',
                       type=int,
                       default=None,
                       help='--generation-length')

    group.add_argument('--task', type=str, default=None, help='task')

    group.add_argument('--pretrained-checkpoint',
                       type=str,
                       default=None,
                       help='Pretrained checkpoint used for finetunning.')

    group.add_argument('--epochs',
                       type=int,
                       default=None,
                       help='Number of finetunning epochs. Zero results in '
                       'evaluation only.')

    group.add_argument('--keep-last',
                       action='store_true',
                       help='Keep the last batch (maybe incomplete) in'
                       'the data loader')

    group.add_argument('--data-dir', default=None, help='data-dir')

    group.add_argument('--geglu', action='store_true',
                       help='Use gated linear units and SiLU activation instead of default gelu')

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

    group.add_argument('--patch-tokenizer-type',
                       type=str,
                       help='patch-tokenizer-type')

    group.add_argument('--position-encoding-2d',
                       action='store_true',
                       help='position-encoding-2d')

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
    args = get_args()

    train_ds, valid_ds, test_ds = \
        build_pretrain_glm130b_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            max_seq_length=args.seq_length,
            generation_length=args.generation_length,
            short_seq_prob=args.short_seq_prob,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup))

    return train_ds, valid_ds, test_ds


def forward_step(data_iterator, model):

    keys = [
        'tokens', 'targets', 'position_ids', 'attention_mask', 'loss_mask'
    ]
    datatype = torch.int64
    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    batch = tensor_parallel.broadcast_data(keys, data, datatype)

    tokens = batch['tokens'].long().cuda().contiguous()
    labels = batch['targets'].long().cuda().contiguous()
    attention_mask = batch['attention_mask'].long().cuda().contiguous()
    loss_mask = batch['loss_mask'].long().cuda().contiguous()
    position_ids = batch['position_ids'].long().cuda().contiguous()
    attention_mask = attention_mask < 0.5
    attention_mask = attention_mask.to(torch.bool).unsqueeze(1)
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    def loss_func(loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        averaged_loss = average_losses_across_data_parallel_group([loss])
        return loss, {'lm loss': averaged_loss[0]}

    return output_tensor, partial(loss_func, loss_mask)

if __name__ == '__main__':
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=get_tasks_args)
