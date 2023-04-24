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

import os
import random
import sys

import numpy as np
import torch

from megatron import update_num_microbatches
from megatron.checkpointing import (_transpose_first_dim,
                                    find_checkpoint_rank_0,
                                    get_checkpoint_names,
                                    get_checkpoint_tracker_filename,
                                    get_checkpoint_version, read_metadata,
                                    set_checkpoint_version)
from megatron.core import mpu, tensor_parallel
from megatron.global_vars import get_args
from megatron.utils import print_rank_0, unwrap_model


def check_checkpoint_args(checkpoint_args):
    """Ensure fixed arguments for a model are the same for the input
    arguments and the one retrieved from checkpoint."""
    args = get_args()

    def _compare(arg_name, old_arg_name=None):
        if old_arg_name is not None:
            checkpoint_value = getattr(checkpoint_args, old_arg_name)
        else:
            checkpoint_value = getattr(checkpoint_args, arg_name)
        args_value = getattr(args, arg_name)
        error_message = '{} value from checkpoint ({}) is not equal to the ' \
                        'input argument' \
                        'value ({}).'.format(arg_name,
                                             checkpoint_value,
                                             args_value)
        assert checkpoint_value == args_value, error_message

    _compare('num_layers')
    _compare('hidden_size')
    _compare('num_attention_heads')
    """
    if args.vocab_file:
        _compare('max_position_embeddings')
        _compare('make_vocab_size_divisible_by')
        _compare('padded_vocab_size')
        _compare('tokenizer_type')
    """
    if args.data_parallel_random_init:
        _compare('data_parallel_random_init')
    if get_checkpoint_version() < 3.0:
        _compare('tensor_model_parallel_size',
                 old_arg_name='model_parallel_size')
    if get_checkpoint_version() >= 3.0:
        _compare('tensor_model_parallel_size')
        _compare('pipeline_model_parallel_size')


def fix_query_key_value_ordering(model, checkpoint_version):
    """Fix up query/key/value matrix ordering if checkpoint
    version is smaller than 2.0
    """
    if checkpoint_version < 2.0:
        if isinstance(model, list):
            assert len(model) == 1
            model = model[0]
        for name, param in model.named_parameters():
            tmp1 = '.query_key_value.weight'
            tmp2 = '.query_key_value.bias'
            if name.endswith((tmp1, tmp2)):
                if checkpoint_version == 0:
                    fixed_param = _transpose_first_dim(param.data, 3, True,
                                                       model)
                elif checkpoint_version == 1.0:
                    fixed_param = _transpose_first_dim(param.data, 3, False,
                                                       model)
                else:
                    print_rank_0(
                        f'Invalid checkpoint version {checkpoint_version}.')
                    sys.exit()
                param.data.copy_(fixed_param)
            if name.endswith(('.key_value.weight', '.key_value.bias')):
                if checkpoint_version == 0:
                    fixed_param = _transpose_first_dim(param.data, 2, True,
                                                       model)
                elif checkpoint_version == 1.0:
                    fixed_param = _transpose_first_dim(param.data, 2, False,
                                                       model)
                else:
                    print_rank_0(
                        f'Invalid checkpoint version {checkpoint_version}.')
                    sys.exit()
                param.data.copy_(fixed_param)
        print_rank_0(' succesfully fixed query-key-values ordering for'
                     ' checkpoint version {}'.format(checkpoint_version))


def _load_base_checkpoint(load_dir, use_distributed_optimizer, rank0=False):
    """ Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    """

    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_dir)

    # If no tracker file, return nothing
    if not os.path.isfile(tracker_filename):
        if not rank0:
            print_rank_0(
                'WARNING: could not find the metadata file {} '.format(
                    tracker_filename))
            print_rank_0(
                '    will not load any checkpoints and will start from '
                'random')
        return None, None, False

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration, release = read_metadata(tracker_filename)

    # Checkpoint.
    if rank0:
        checkpoint_names = find_checkpoint_rank_0(load_dir, iteration,
                                                  use_distributed_optimizer,
                                                  release)
    else:
        checkpoint_names = get_checkpoint_names(load_dir, iteration,
                                                use_distributed_optimizer,
                                                release)
        if release:
            print_rank_0(f' loading release checkpoint from {load_dir}')
        else:
            print_rank_0(
                f' loading checkpoint from {load_dir} at iteration {iteration}'
            )

    model_checkpoint_name, optim_checkpoint_name = checkpoint_names
    # Load the checkpoint.
    args = get_args()
    try:
        model_state_dict = torch.load(model_checkpoint_name,
                                      map_location='cpu')
        if not args.no_load_optim:
            if use_distributed_optimizer:
                optim_state_dict = torch.load(optim_checkpoint_name,
                                              map_location='cpu')
            else:
                optim_state_dict = model_state_dict
        else:
            optim_state_dict = None
    except ModuleNotFoundError:
        # For backward compatibility.
        if not rank0:
            print_rank_0(' > deserializing using the old code structure ...')
        sys.modules['fp16.loss_scaler'] = sys.modules[
            'megatron.fp16_deprecated.loss_scaler']
        sys.modules['megatron.fp16.loss_scaler'] = sys.modules[
            'megatron.fp16_deprecated.loss_scaler']
        model_state_dict = torch.load(model_checkpoint_name,
                                      map_location='cpu')
        optim_state_dict = torch.load(optim_checkpoint_name,
                                      map_location='cpu')
        sys.modules.pop('fp16.loss_scaler', None)
        sys.modules.pop('megatron.fp16.loss_scaler', None)
    except BaseException as e:
        print_rank_0('could not load the checkpoint')
        print_rank_0(e)
        sys.exit()

    return model_state_dict, optim_state_dict, release


def load_checkpoint(model,
                    optimizer,
                    opt_param_scheduler,
                    load_arg='load',
                    strict=True):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    args = get_args()
    load_dir = getattr(args, load_arg)

    model = unwrap_model(model)
    model_state_dict, optim_state_dict, release = \
        _load_base_checkpoint(
            load_dir,
            use_distributed_optimizer=args.use_distributed_optimizer,
            rank0=False)

    if model_state_dict is None:
        return 0

    # set checkpoint version
    set_checkpoint_version(model_state_dict.get('checkpoint_version', 0))

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = model_state_dict['iteration']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = model_state_dict['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but unable to load '
                             'iteration from checkpoint {}')
                sys.exit()

    # Check arguments.
    assert args.consumed_train_samples == 0
    assert args.consumed_valid_samples == 0
    if 'args' in model_state_dict:
        checkpoint_args = model_state_dict['args']
        # check_checkpoint_args(checkpoint_args)
        args.consumed_train_samples = getattr(checkpoint_args,
                                              'consumed_train_samples', 0)
        update_num_microbatches(consumed_samples=args.consumed_train_samples)
        args.consumed_valid_samples = getattr(checkpoint_args,
                                              'consumed_valid_samples', 0)
    else:
        print_rank_0('could not find arguments in the checkpoint ...')

    # Model.
    if len(model) == 1:
        model[0].load_state_dict(model_state_dict['model'], strict=strict)
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            model[i].load_state_dict(model_state_dict['model%d' % i],
                                     strict=strict)

    # Fix up query/key/value matrix ordering if needed
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f' checkpoint version {checkpoint_version}')
    fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            if optimizer is not None:
                optimizer.load_state_dict(optim_state_dict['optimizer'])
            if opt_param_scheduler is not None:
                if 'lr_scheduler' in optim_state_dict:  # backward compatbility
                    opt_param_scheduler.load_state_dict(
                        optim_state_dict['lr_scheduler'])
                else:
                    opt_param_scheduler.load_state_dict(
                        optim_state_dict['opt_param_scheduler'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint. '
                         'Specify --no-load-optim or --finetune to prevent '
                         'attempting to load the optimizer state, '
                         'exiting ...')
            sys.exit()

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            if 'rng_state' in model_state_dict:
                # access rng_state for data parallel rank
                if args.data_parallel_random_init:

                    rng_state = model_state_dict['rng_state'][
                        mpu.get_data_parallel_rank()]
                else:
                    rng_state = model_state_dict['rng_state'][0]
                random.setstate(rng_state['random_rng_state'])
                np.random.set_state(rng_state['np_rng_state'])
                torch.set_rng_state(rng_state['torch_rng_state'])
                torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
                # Check for empty states array
                if not rng_state['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    rng_state['rng_tracker_states'])
            else:  # backward compatability
                random.setstate(model_state_dict['random_rng_state'])
                np.random.set_state(model_state_dict['np_rng_state'])
                torch.set_rng_state(model_state_dict['torch_rng_state'])
                torch.cuda.set_rng_state(model_state_dict['cuda_rng_state'])
                # Check for empty states array
                if not model_state_dict['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    model_state_dict['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load rng state from checkpoint. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the rng state, '
                         'exiting ...')
            sys.exit()

    # Some utilities want to load a checkpoint
    # without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(f'  successfully loaded checkpoint from {args.load} '
                 f'at iteration {iteration}')

    return iteration
