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
"""test_config.py"""

import os
import sys
import pytest

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(root_path, 'mindformers'))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_config():
    """
    Feature: Merge yaml.
    Description: Load a fake yaml where the parameters and initial values are different, and match the expected values.
    Expectation: All parameters in yaml are right.
    """
    from mindrlhf.configs.grpo_configs import GRPOConfig
    grpo_config_empty = GRPOConfig(f"{root_path}/qwen2_5/grpo_config_empty.yaml")
    grpo_config_fake = GRPOConfig(f"{root_path}/qwen2_5/grpo_config_fake.yaml")

    assert grpo_config_empty.actor_config.enable_alltoall != grpo_config_fake.actor_config.enable_alltoall == True
    assert grpo_config_empty.actor_config.lr_schedule.lr != grpo_config_fake.actor_config.lr_schedule.lr == 6e-07
    assert grpo_config_empty.actor_config.optimizer.type != grpo_config_fake.actor_config.optimizer.type == 'adam'
    assert grpo_config_empty.actor_config.parallel_config.data_parallel != \
           grpo_config_fake.actor_config.parallel_config.data_parallel == 10
    assert grpo_config_empty.actor_config.recompute_config.recompute != \
           grpo_config_fake.actor_config.recompute_config.recompute == True
    assert grpo_config_empty.context.device_id != grpo_config_fake.context.device_id == 1
    assert grpo_config_empty.context.param_dict['mode'] != grpo_config_fake.context.param_dict['mode'] == 1
    assert grpo_config_empty.generate_config.block_size != grpo_config_fake.generate_config.block_size == 160
    assert grpo_config_empty.generate_config.parallel_config.data_parallel != \
           grpo_config_fake.generate_config.parallel_config.data_parallel == 80
    assert grpo_config_empty.generate_config.sampling_config.temperature != \
           grpo_config_fake.generate_config.sampling_config.temperature == 0.08
    assert grpo_config_empty.ref_config.parallel_config.data_parallel != \
           grpo_config_fake.ref_config.parallel_config.data_parallel == 40
    assert grpo_config_empty.ref_config.recompute_config.recompute != \
           grpo_config_fake.ref_config.recompute_config.recompute == True
    assert grpo_config_empty.reward_config.verifier_function != \
           grpo_config_fake.reward_config.verifier_function == ['accuracy_reward']
    assert grpo_config_empty.rl_config.align_type != grpo_config_fake.rl_config.align_type == "train_stages"
