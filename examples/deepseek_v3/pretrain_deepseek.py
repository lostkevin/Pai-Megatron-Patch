# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
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

import inspect
from typing import Union
from contextlib import nullcontext
import torch
import torch._dynamo

import megatron.legacy.model
from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.models.gpt import GPTModel

from megatron_patch.model.deepseek_v3.layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron_patch.arguments import get_patch_args
from megatron_patch.tokenizer import build_tokenizer, get_tokenizer
from megatron_patch.data import train_valid_test_datasets_provider


torch._dynamo.config.suppress_errors = True


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    build_tokenizer(args)
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')

    config = core_transformer_config_from_args(args)

    if args.num_experts:
        # Define the decoder block spec
        transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te)
    else:
        # Define the decoder layer spec
        if use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                args.num_experts, args.moe_grouped_gemm,
                args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(
                args.num_experts, args.moe_grouped_gemm,
                args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)

    build_model_context = nullcontext
    build_model_context_args = {}
    if args.fp8_param_gather:
        try:
            from transformer_engine.pytorch import fp8_model_init

            build_model_context = fp8_model_init
            build_model_context_args["enabled"] = True

            # Check if fp8_model_init supports preserve_high_precision_init_val
            if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                build_model_context_args["preserve_high_precision_init_val"] = True
        except:
            raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

    with build_model_context(**build_model_context_args):
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling
        )

    return model


if __name__ == "__main__":
    from megatron_patch.template.helper import forward_step
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=get_patch_args,
    )
