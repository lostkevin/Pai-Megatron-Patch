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

from megatron import get_args
from megatron.initialize import initialize_megatron
from megatron_patch.data.finetune_dataset import AlpacaDataset
from megatron_patch.finetune_utils import finetune
from megatron_patch.tokenizer import build_tokenizer, get_tokenizer
from transformers import AutoModelForCausalLM


def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='chatglm')

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

    group.add_argument('--patch-tokenizer-type',
                       type=str,
                       help='patch-tokenizer-type')

    group.add_argument('--cache-dir',
                       type=str,
                       help='cache-dir')

    return parser


def model_provider(pre_process=True, post_process=True):
    args = get_args()
    tokenizer = get_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(args.load, trust_remote_code=True)
    num_new_tokens = len(tokenizer) - tokenizer.vocab_size

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    return model


def train_valid_datasets_provider():
    """Build train and validation dataset."""
    args = get_args()
    tokenizer = build_tokenizer(args)
    train_dataset = AlpacaDataset(args.train_data, tokenizer)
    valid_dataset = AlpacaDataset(args.valid_data, tokenizer)
    return train_dataset, valid_dataset


def forward_step(data_iterator, model):
    try:
        data_iterator = next(data_iterator)
    except BaseException:
        data_iterator = data_iterator

    tokens = data_iterator['input_ids'].long().cuda()
    labels = data_iterator['labels'].long().cuda()
    attention_mask = data_iterator['attention_mask'].long().cuda()
    output_tensor = model(input_ids=tokens, labels=labels, attention_mask=attention_mask)
    return output_tensor.loss


if __name__ == '__main__':
    initialize_megatron(extra_args_provider=get_tasks_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

    finetune(train_valid_datasets_provider=train_valid_datasets_provider,
             model_provider=model_provider,
             forward_step=forward_step)
