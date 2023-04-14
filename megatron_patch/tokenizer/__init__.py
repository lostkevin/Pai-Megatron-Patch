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

from megatron.tokenizer.tokenizer import _vocab_size_with_padding

_GLOBAL_TOKENIZER = None


def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.patch_tokenizer_type))

    # Select and instantiate the tokenizer.
    if args.patch_tokenizer_type == 'JiebaBPETokenizer':
        from .jiebabpe_tokenizer import JiebaBPETokenizer
        tokenizer = JiebaBPETokenizer(args.patch_vocab_file)
    elif args.patch_tokenizer_type == 'BloomTokenizerFromHF':
        from transformers import BloomTokenizerFast as BloomTokenizer
        tokenizer = BloomTokenizer.from_pretrained('bigscience/bloom-560m')
        tokenizer.eod = tokenizer.eos_token_id
        args.padded_vocab_size = 250880
    elif args.patch_tokenizer_type == 'ChatGLMTokenizerFromHF':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b',
                                                  trust_remote_code=True)
        args.padded_vocab_size = 130528
    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(
                                      args.patch_tokenizer_type))

    if args.patch_tokenizer_type !=\
            'BloomTokenizerFromHF' and args.patch_tokenizer_type !=\
            'ChatGLMTokenizerFromHF':
        args.padded_vocab_size = _vocab_size_with_padding(
            tokenizer.vocab_size, args)
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = tokenizer
    return _GLOBAL_TOKENIZER


def get_tokenizer():
    """Return tokenizer."""
    return _GLOBAL_TOKENIZER
