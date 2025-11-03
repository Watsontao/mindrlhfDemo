# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" infer main """
import os
import argparse
import time

import numpy as np
import mindspore as ms
from mindspore.communication import get_rank
from mindrlhf.trainer.spmd.grpo_trainer import GRPOTrainer
from mindrlhf.worker.infer_worker import InferWorker
from mindrlhf.configs.grpo_configs import VllmMode


tokens = [
    [28084, 3867, 264, 2618, 23988, 65408, 304, 806, 12534, 13],
    [28084, 3867, 264, 2618, 23988, 65408, 304, 806, 12534, 13],
]


class GRPOInferTest(GRPOTrainer):
    """Infercase"""

    def __init__(self, args):
        self.args = args
        self._init_grpo_configs(args)
        self._init_reward_fn()
        seed = self.grpo_config.rl_config.seed
        ms.set_deterministic(True)
        ms.set_seed(seed)
        self._update_args(args)
        self.grpo_config.generate_config.model_config = args.model_config
        self.inferwork = InferWorker(grpo_config=self.grpo_config, args=self.args)
        self.inferwork.generate_strategy(None)
        ms.mint.distributed.barrier()

        save_strategy_dir = self.grpo_config.rl_config.save_strategy_dir
        dst_merged_stra = os.path.join(save_strategy_dir, "merge_strategy", "infer_policy_merged_strategy.ckpt")
        if get_rank() == 0:
            ms.merge_pipeline_strategys(os.path.join(save_strategy_dir, "infer_policy_strategy"), dst_merged_stra)
        else:
            time.sleep(10)
        ms.mint.distributed.barrier()

        self.inferwork.load_checkpoint()
        self.infer_dp = self.inferwork.get_infer_dp()

    def _update_args(self, args):
        """update args"""
        self.grpo_config.generate_config.parallel_config.data_parallel = args.data_parallel
        self.grpo_config.generate_config.parallel_config.param_dict["data_parallel"] = args.data_parallel
        self.grpo_config.generate_config.parallel_config.model_parallel = args.model_parallel
        self.grpo_config.generate_config.parallel_config.param_dict["model_parallel"] = args.model_parallel
        self.grpo_config.generate_config.sampling_config.temperature = args.temperature
        self.grpo_config.generate_config.sampling_config.repetition_penalty = args.repetition_penalty
        self.grpo_config.generate_config.sampling_config.top_p = args.top_p
        self.grpo_config.generate_config.sampling_config.top_k = args.top_k
        self.grpo_config.generate_config.use_vllm = args.use_vllm
        self.grpo_config.generate_config.load = args.tokenizer_dir

    def infer_(self):
        """infer entrance"""
        input_ids_numpy = np.array(tokens, dtype=np.int32)
        num_rollouts = self.grpo_config.rl_config.num_rollouts
        num_generations = self.grpo_config.rl_config.num_generations
        n_questions = 1
        max_tokens = self.grpo_config.generate_config.sampling_config.max_tokens
        results = []
        for _ in range(2):
            if self.inferwork.use_vllm == VllmMode.ORIGIN:
                results = []
                input_bs = n_questions // self.infer_dp
                for idx in range(num_rollouts * num_generations):
                    result = self.inferwork.generate(
                        input_ids_numpy[idx * input_bs : (idx + 1) * input_bs, :], max_tokens=max_tokens
                    )
                    for res_idx in range(len(result)):
                        if len(results) == len(result):
                            results[res_idx] = np.concatenate((results[res_idx], result[res_idx]))
                        else:
                            results.append(result[res_idx])
            else:

                results = self.inferwork.generate(input_ids_numpy, max_tokens)

        right_padding_responses, _, left_padding_prompts, _ = self.inferwork.post_process_infer_outputs(results)
        pad_token_id = self.grpo_config.generate_config.sampling_config.pad_token_id
        no_padding_prompts = self._remove_left_padding(left_padding_prompts, padding_token=pad_token_id)
        no_padding_responses = self._remove_right_padding(right_padding_responses, padding_token=pad_token_id)
        prompts = self.tokenizer.decode(no_padding_prompts, skip_special_tokens=True)
        completions = self.tokenizer.decode(no_padding_responses, skip_special_tokens=True)

        print(f"prompts: \n {prompts}")
        print(f"completions: \n {completions}")


def get_args():
    """get args"""
    parser = argparse.ArgumentParser(description="grpo inference test")
    parser.add_argument("--config", type=str, default=None, help="configs path", required=True)
    parser.add_argument("--model_name", type=str, default="qwen", help="custom model name")
    parser.add_argument("--dataset_file", type=str, default=None, help="dataset file for training")
    parser.add_argument("--tokenizer_type", type=str, default="qwen", help="custom tokenizer type")
    parser.add_argument("--tokenizer_dir", type=str, default=None, help="the directory contain hf tokenizer files")
    parser.add_argument("--actor_checkpoint_path", type=str, default=None, help="the actor model file path for loading")
    parser.add_argument(
        "--ref_checkpoint_path", type=str, default=None, help="the reference model file path for loading"
    )
    parser.add_argument(
        "--generate_checkpoint_path", type=str, default=None, help="the generate model file path for loading"
    )
    parser.add_argument("--verifier_function", type=str, default=None, help="verifier funcs")
    parser.add_argument("--verifier_weight", type=str, default=None, help="verifier weights")
    parser.add_argument("--tensorboard", type=str, default=None, help="enable tensorboard")
    parser.add_argument("--save_checkpoint_dir", type=str, default=None, help="save model path")
    parser.add_argument("--model_config", type=str, default=None, help="generate_config.model_config")
    parser.add_argument("--data_parallel", type=int, default=2, help="save model path")
    parser.add_argument("--model_parallel", type=int, default=4, help="save model path")
    parser.add_argument("--temperature", type=float, default=0.8, help="save model path")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="save model path")
    parser.add_argument("--top_p", type=float, default=0.8, help="save model path")
    parser.add_argument("--top_k", type=int, default=20, help="save model path")
    parser.add_argument("--use_vllm", type=int, default=1, help="save model path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args_ = get_args()
    infercase = GRPOInferTest(args_)
    infercase.infer_()
