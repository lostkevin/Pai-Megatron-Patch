# Copyright (c) 2021 Alibaba PAI Team.
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

import timeit

import pynvml
import torch
from megatron import get_args, get_timers

from megatron_patch.checkpointing import load_checkpoint
try:
    from megatron.model import ModelType
except ImportError:
    from megatron.core.enums import ModelType
from megatron.text_generation_server import MegatronServer
from megatron_patch.generation.api import generate_and_post_process
from megatron_patch.generation.api import beam_search_and_post_process
from megatron_patch.tokenizer import build_tokenizer, get_tokenizer
from megatron_patch.training import setup_model_and_optimizer

class GPTPredictor():
    """A Predictor for model."""
    def __init__(self):
        super().__init__()

    def predict(self):
        """Run predict process """

        args = get_args()
        tokenizer = build_tokenizer(args)
        timers = get_timers()
        
        args.train_iters = 1
        
        # Model, optimizer, and learning rate.
        timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
        model, _, _ = setup_model_and_optimizer(self.model_optimizer_lr_scheduler_provider, ModelType.encoder_or_decoder)
        timers('model-and-optimizer-setup').stop()
        torch.distributed.barrier()

        assert args.load is not None
        timers = get_timers()
        timers('load-checkpoint', log_level=0).start(barrier=True)
        # _ = load_checkpoint(model, None, None)

        timers('load-checkpoint').stop()
        timers.log(['load-checkpoint'])

        timers.log(['model-and-optimizer-setup'])

        if not isinstance(model, list):
            model = [model]

        assert len(model) == 1, 'Above condition should have caught this'
        model = model[0]
        if args.fp16:
            model = model.module.module
        else:
            model = model.module

        if args.text_generate_input_file != '':
            num_examples = len(open(args.text_generate_input_file).readlines())
            prompts = []
            pred_outputs = []
            with open(args.text_generate_input_file,
                      encoding='utf-8') as reader,\
                    open(args.text_generate_output_file,
                         'w', encoding='utf-8') as writer:
                buffer = []

                for idx, line in enumerate(reader):
                    line = line.strip()
                    line = line[:args.seq_length]
                    prompts.append(line)
                    if len(buffer) < args.micro_batch_size:
                        buffer.append(line)

                    if len(
                            buffer
                    ) == args.micro_batch_size or idx == num_examples - 1:
                        sl = args.out_seq_length
                        tk = args.top_k
                        tp = args.top_p
                        temperature = args.temperature
                        prompts_plus_generations, _, _, _ = \
                            generate_and_post_process(model,
                                                      prompts=buffer,
                                                      tokens_to_generate=sl,
                                                      top_k_sampling=tk,
                                                      temperature=0.1,
                                                      top_p_sampling=tp)

                        for prompt, p_and_g in zip(buffer,
                                                   prompts_plus_generations):
                            generation = p_and_g.replace('<|endoftext|>', '')
                            print(p_and_g)
                            writer.write(generation + '\n')
                            pred_outputs.append(generation)
                        buffer.clear()

                    if idx % args.micro_batch_size == 0:
                        print('processed {} examples'.format(idx))

            # if args.text_generate_gt_file:
            #     gt_outputs = []
            #     with open(args.text_generate_gt_file,
            #               encoding='utf-8') as reader:
            #         for line in reader:
            #             gt_outputs.append(line.strip())

            #     bleu = calc_bleu(pred_outputs, gt_outputs)
            #     parent = calc_parent(pred_outputs, gt_outputs, prompts)
            #     avg = (bleu + parent) / 2
            #     print('bleu {}, parent {}, agv {}'.format(bleu, parent, avg))

        elif args.time:
            if args.input_len == 1:
                input = '我'
            elif args.input_len == 20:
                input = '有关人士建议，这些特殊类型考生最好不要在提前批志愿和'

            iterations = 10
            for i in range(iterations):
                generate_and_post_process(
                    model,
                    prompts=[input],
                    tokens_to_generate=args.out_seq_length,
                    top_k_sampling=args.top_k,
                    top_p_sampling=args.top_p,
                    print_result=False)

            pynvml.nvmlInit()
            # 这里的0是GPU id
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

            time_start_gene = timeit.default_timer()
            inference_times = 0
            for i in range(iterations):
                _, _, _, inference_time = generate_and_post_process(
                    model,
                    prompts=[input],
                    tokens_to_generate=args.out_seq_length,
                    top_k_sampling=args.top_k,
                    top_p_sampling=args.top_p,
                    print_result=False)
                inference_times += inference_time
            time_elapsed = timeit.default_timer() - time_start_gene

            print('[INFO] GPT e2e time costs: {:.2f} ms'.format(
                time_elapsed * 1000 / iterations))
            print('[INFO] GPT t2t time costs: {:.2f} ms'.format(
                inference_times * 1000 / iterations))
            print('[INFO] GPU Memory Usage:', meminfo.used / 1024 / 1024)

        else:
            if torch.cuda.current_device() == 0:
                server = MegatronServer(model)
                server.run('0.0.0.0')

            while True:
                choice = torch.zeros(1).cuda()
                torch.distributed.broadcast(choice, 0)
                if choice[0].item() == 0:
                    try:
                        generate_and_post_process(model)
                    except ValueError:
                        pass
                elif choice[0].item() == 1:
                    try:
                        beam_search_and_post_process(model)
                    except ValueError:
                        pass
