# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test grpo trainer."""
import os
import sys
from dataclasses import dataclass
from unittest import mock

import mindspore as ms
import numpy as np
from mindspore import Tensor

no_patch_tensor_shape = Tensor.shape
import pytest

WORKDIR = os.path.dirname(os.path.abspath(__file__))
print(f"WORKDIR is {WORKDIR}")
MINDRLHF_PATH = os.path.join(WORKDIR, "../../")
MINDFORMERS_PATH = os.path.join(WORKDIR, "mindformers")
sys.path = [MINDRLHF_PATH, MINDFORMERS_PATH] + sys.path
print(f"sys.path is {sys.path}")


from mindrlhf.trainer.spmd.grpo_trainer import GRPOTrainer
from mindrlhf.trainer.spmd.grpo_experience_maker import GRPOExperienceMaker
from mindrlhf.worker.train_worker import TrainWorker


solution_ids = np.array(
    [
        20217,
        702,
        220,
        18,
        15,
        311,
        7664,
        13,
        11387,
        702,
        220,
        17,
        15,
        803,
        311,
        7664,
        1091,
        9354,
        1558,
        13,
        20445,
        702,
        10917,
        438,
        1657,
        311,
        7664,
        438,
        11387,
        1558,
        13,
        2585,
        1657,
        311,
        7664,
        1558,
        20445,
        614,
        30,
        36634,
        702,
        220,
        18,
        15,
        10,
        17,
        15,
        284,
        1115,
        18,
        15,
        10,
        17,
        15,
        28,
        20,
        15,
        2452,
        20,
        15,
        311,
        7664,
        198,
        53083,
        702,
        220,
        20,
        15,
        9,
        17,
        284,
        1115,
        20,
        15,
        9,
        17,
        28,
        16,
        15,
        15,
        2452,
        16,
        15,
        15,
        311,
        7664,
        198,
        2,
        2,
        2,
        2,
        220,
        16,
        15,
        15,
    ],
    np.int32,
)
responses = np.array(
    [2014, 11625, 279, 3491, 11, 582, 646, 1438, 432, 1495, 1119, 264, 2421, 4285, 7354, 1447, 16, 13, 29901, 279],
    np.int32,
)
prompts = np.array(
    [
        20217,
        702,
        220,
        18,
        15,
        311,
        7664,
        13,
        11387,
        702,
        220,
        17,
        15,
        803,
        311,
        7664,
        1091,
        9354,
        1558,
        13,
        20445,
        702,
        10917,
        438,
        1657,
        311,
        7664,
        438,
        11387,
        1558,
        13,
        2585,
        1657,
        311,
        7664,
        1558,
        20445,
        614,
        30,
    ],
    np.int32,
)


@dataclass
class GRPOTrainerArgs:
    """Fake GRPOTrainerArgs."""

    config: str = os.path.join(WORKDIR, "qwen2_5/grpo_config_st.yaml")
    model_name: str = "qwen"
    dataset_file: str = ""
    resume_training: bool = False
    tokenizer_type: str = "qwen"
    tokenizer_dir: str = os.path.join(WORKDIR, "qwen2_5_vllm")
    actor_checkpoint_path: str = ""
    ref_checkpoint_path: str = ""
    generate_checkpoint_path: str = ""
    verifier_function: str = "accuracy_reward,format_reward"
    verifier_weight: str = "1.0,1.0"
    tensorboard: str = None
    save_checkpoint_dir: str = None


class TestGRPOTrainer:
    """
    Unittest of GRPOTrainer.
    """

    def __init__(self):
        self.qwen2_5_args = GRPOTrainerArgs()
        self.num_rollouts = 2
        self.num_generations = 4
        self.dp = 1
        self.pack_num = 1
        self.ref_model_batch_size = 1
        self.seq_length = 100
        self.expect_pack_group = 8
        self.pack_num_pre_group = 1
        self.valid_len = responses.shape[0] + prompts.shape[0]
        print(self.qwen2_5_args)

    def get_mock_data(
        self, num_rollouts, num_generations, dp, ref_model_batch_size, max_prompt_length, seq_length, pad_token_id
    ):
        """Get data for mock modules."""
        ref_per_token_logps = ms.Tensor(np.random.uniform(-1, 1, (dp * ref_model_batch_size, seq_length)), ms.bfloat16)
        total_bs = num_rollouts * num_generations * dp
        solution_ids_base = np.ones((num_generations, 4096), np.int32) * pad_token_id
        prompts_tensor = np.ones((num_rollouts * dp, 4096), np.int32)
        right_padding_responses = np.ones((total_bs, max_prompt_length), np.int32) * pad_token_id
        responses_mask_gather = np.zeros((total_bs, max_prompt_length), np.int32)
        left_padding_prompts = np.ones((total_bs, max_prompt_length), np.int32) * pad_token_id
        prompts_mask_gather = np.zeros((total_bs, max_prompt_length), np.int32)
        for i in range(num_generations):
            solution_ids_base[i, : solution_ids.shape[0]] = solution_ids
        for i in range(total_bs):
            right_padding_responses[i, : responses.shape[0]] = responses
            responses_mask_gather[i, : responses.shape[0]] = 1
            left_padding_prompts[i, -prompts.shape[0] :] = prompts
            prompts_mask_gather[i, -prompts.shape[0] :] = 1
        return (
            [prompts_tensor, solution_ids_base],
            right_padding_responses,
            responses_mask_gather,
            left_padding_prompts,
            prompts_mask_gather,
            ref_per_token_logps,
        )

    @mock.patch.object(GRPOTrainer, "rename_safetensors_weights")
    @mock.patch.object(GRPOTrainer, "_load_checkpoint")
    @mock.patch.object(GRPOTrainer, "_compile")
    @mock.patch.object(TrainWorker, "_init_grpo_network_and_optimizer")
    @mock.patch.object(GRPOTrainer, "_init_net_parameters")
    @mock.patch.object(TrainWorker, "set_optimizer")
    @mock.patch.object(GRPOExperienceMaker, "_get_batch")
    @mock.patch.object(GRPOExperienceMaker, "_init_grpo_experience_dataset")
    @mock.patch("mindrlhf.trainer.spmd.grpo_experience_maker.get_dp_rank")
    @mock.patch("mindrlhf.trainer.spmd.grpo_trainer.TransformWorker")
    @mock.patch("mindrlhf.trainer.spmd.grpo_trainer.InferWorker")
    @mock.patch("mindrlhf.trainer.spmd.grpo_trainer.get_ref_worker")
    @mock.patch("mindrlhf.worker.train_worker.get_rank")
    @mock.patch("mindrlhf.worker.train_worker.CausalLMHybrid")
    @mock.patch("mindrlhf.worker.train_worker.GRPOModelTrain")
    @mock.patch("mindrlhf.trainer.spmd.grpo_trainer.get_old_policy_worker")
    # pylint: disable=E1120
    def run_grpo_packing(
        self,
        mock_old_policy_worker,
        mock_train_grpo_model,
        mock_train_model,
        mock_train_get_rank,
        mock_ref_worker,
        mock_infer_worker,
        mock_transfer_worker,
        mock_get_dp_rank,
        mock_init_train_network_and_optimizer,
        mock_get_batch,
        mock_set_optimizer,
        mock_init_net_parameters,
        mock_init_infer_dataloader,
        mock_compile,
        mock_load_ckpt,
        mock_rename_sf_weights,
    ):
        """A unit test example for GRPO Trainer."""
        args = self.qwen2_5_args

        mock_ref_worker.return_value.get_ref_dp.return_value = self.dp
        mock_infer_worker.return_value.get_infer_dp.return_value = self.dp
        mock_get_dp_rank.return_value = 0
        mock_train_get_rank.return_value = 0
        mock_train_grpo_model.return_value = ms.nn.Cell()
        grpo_trainer = GRPOTrainer(no_patch_tensor_shape=no_patch_tensor_shape, args=args)
        grpo_trainer.grpo_config.rl_config.save_prompt_completions_data = False
        grpo_trainer.grpo_config.rl_config.pack_num = self.pack_num
        grpo_trainer.grpo_config.rl_config.seq_length = self.seq_length
        grpo_trainer.train.sft_model_config_train.seq_length = self.seq_length
        grpo_trainer.grpo_config.ref_config.ref_model_batch_size = self.ref_model_batch_size
        grpo_config = grpo_trainer.grpo_config
        data = self.get_mock_data(
            self.num_rollouts,
            self.num_generations,
            self.dp,
            self.ref_model_batch_size,
            50,
            grpo_config.rl_config.seq_length,
            grpo_config.generate_config.sampling_config.pad_token_id,
        )

        mock_get_batch.return_value = data[0]
        mock_infer_worker.return_value.post_process_infer_outputs.return_value = data[1:-1]
        mock_ref_worker.return_value.compute_ref_log_prob.return_value = data[-1]
        grpo_trainer.experience_maker.make_experience(self.num_rollouts, self.num_generations)
        store = grpo_trainer.train.store
        assert len(store) == self.expect_pack_group
        base_prompt_ids = np.concatenate((prompts, responses), axis=0)
        base_res_mask = np.concatenate(
            (np.zeros((len(prompts),), np.int32), np.ones((len(responses),), np.int32)), axis=0
        )
        last_group_len = (self.num_rollouts * self.num_generations * self.dp - 1) % self.pack_num_pre_group + 1
        for i in range(self.expect_pack_group):
            start_idx = 0
            prompt_ids = store[i].prompt_completion_ids
            res_mask = store[i].responses_mask
            actual_sequence_length = store[i].actual_sequence_length
            assert actual_sequence_length[-1] == self.seq_length
            for j in range(self.pack_num_pre_group):
                if i == self.expect_pack_group - 1 and j >= last_group_len - 1:
                    break
                if j != self.pack_num_pre_group - 1 or (i == self.expect_pack_group - 1 and j != last_group_len - 1):
                    assert actual_sequence_length[j] == start_idx + self.valid_len + 1, actual_sequence_length
                assert (prompt_ids[start_idx : start_idx + self.valid_len] == base_prompt_ids).all()
                assert (res_mask[start_idx : start_idx + self.valid_len - 1] == base_res_mask[1:]).all()
                start_idx += self.valid_len + 1


grpo_trainer_tester = TestGRPOTrainer()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.parametrize(
    "num_rollouts, num_generations, dp, pack_num, ref_model_batch_size, seq_length, "
    "expect_pack_group, pack_num_pre_group",
    [(2, 4, 1, 1, 1, 128, 8, 1)],
)
def test_grpo_packing(
    num_rollouts, num_generations, dp, pack_num, ref_model_batch_size, seq_length, expect_pack_group, pack_num_pre_group
):
    """
    Feature: change num_rollouts, num_generations, dp, pack_num... and  check packed data.
    Description: Test qwen grpo packing.
    Expectation: success.
    """
    grpo_trainer_tester.num_rollouts = num_rollouts
    grpo_trainer_tester.num_generations = num_generations
    grpo_trainer_tester.dp = dp
    grpo_trainer_tester.pack_num = pack_num
    grpo_trainer_tester.ref_model_batch_size = ref_model_batch_size
    grpo_trainer_tester.seq_length = seq_length
    grpo_trainer_tester.expect_pack_group = expect_pack_group
    grpo_trainer_tester.pack_num_pre_group = pack_num_pre_group
    grpo_trainer_tester.run_grpo_packing()
