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
from megatron_patch.generation.gpt_predictor import GPTPredictor
from megatron_patch.model.bloom.gpt_model import GPTModel

try:
    from megatron.model import ModelType
except ImportError:
    from megatron.core.enums import ModelType


def get_tasks_args(parser):
    group = parser.add_argument_group(title='text generation')

    group.add_argument('--local-rank', type=int, default=None,
                        help='local rank passed from distributed launcher')

    group.add_argument('--text-generate-input-file', type=str, default='')
    group.add_argument('--text-generate-output-file', type=str, default='')
    group.add_argument('--text-generate-gt-file', type=str, default='')
    group.add_argument('--time',
                       action='store_true',
                       help='measure end to end text generation average time')
    group.add_argument('--eval-dev', action='store_true')
    group.add_argument(
        '--input-len',
        type=int,
        default=1,
        help='input lenth for measure end to end text generation average time')
    group.add_argument('--patch-tokenizer-type',
                       type=str,
                       help='patch-tokenizer-type')
    group.add_argument('--top-p',
                       type=float,
                       default=0.0,
                       help='Top p sampling.')
    group.add_argument('--top-k', type=int, default=0, help='Top k sampling.')

    group.add_argument('--out-seq-length',
                       type=int,
                       default=1024,
                       help='Size of the output generated text.')

    group.add_argument('--temperature',
                       type=float,
                       default=1.2,
                       help='Sampling temperature.')
    group.add_argument('--repetition-penalty',
                       type=float,
                       default=1.2,
                       help='Repetition_penalty.')
    group.add_argument('--embed-layernorm',
                       action='store_true',
                       help='use layernorm for embedding')

    group.add_argument('--position-embedding-type',
                       type=str,
                       default='absolute',
                       help='Define position embedding type '
                       '("absolute"|"rotary"|"alibi"). "absolute" by default.')

    group.add_argument('--glu-activation',
                       type=str,
                       help='GLU activations to use.')
    return parser


class MegatronGPTPredictor(GPTPredictor):
    def model_provider(self, pre_process=True, post_process=True):
        args = get_args()
        args.model_type = ModelType.encoder_or_decoder
        model = GPTModel(num_tokentypes=0,
                         parallel_output=False,
                         pre_process=pre_process,
                         post_process=post_process)

        return model


if __name__ == '__main__':
    initialize_megatron(extra_args_provider=get_tasks_args)
    predictor = MegatronGPTPredictor()
    predictor.predict()
