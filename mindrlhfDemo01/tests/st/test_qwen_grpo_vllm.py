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
"""test_qwen_grpo_vllm.py"""

import os
import pytest
from tests.mark_utils import arg_mark
from tests.st.utils import check_log

root_path = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_qwen_grpo_vllm():
    """
    Feature: vllm and cp
    Description: generate use vllm, train open context parallel
    Expectation: The training process is successful and the checkpoints are saved.
    """
    os.system(f"bash {root_path}/run_qwen_grpo_vllm_test.sh")

    log_path = f"{root_path}/qwen2_vllm_log/worker_0.log"
    check_pair = {"Save checkpoints in": 1}
    check_values = [
        "generate parallel_config:{'data_parallel': 4, 'model_parallel': 2, 'pipeline_stage': 1}",
        "actor parallel_config:{'data_parallel': 1, 'model_parallel': 4, 'pipeline_stage': 2,"
        " 'use_seq_parallel': False, 'micro_batch_num': 2, 'vocab_emb_dp': False}",
        "parallel_config:{'data_parallel': 4, 'model_parallel': 2, 'pipeline_stage': 1}",
        # recompute config
        "grpo_config.ref_config.recompute_config:{'recompute': False, 'select_recompute': False, "
        "'select_comm_recompute': False, 'parallel_optimizer_comm_recompute': False, 'mp_comm_recompute': True, "
        "'recompute_slice_activation': False}",
        "total_ref_batch_size: ref_model_batch_size * ref_dp, 2 * 4 = 8",
    ]

    device_memory = [62370, 32089, 43500]
    host_memory = [24, 230]
    check_log(log_path, check_pair, check_values, device_memory, host_memory)



@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_qwen_grpo_vllm_temporary():
    """
    Feature: vllm and cp
    Description: generate use vllm, train open context parallel
    Expectation: The training process is successful and the checkpoints are saved.
    """
    os.system(f"bash {root_path}/run_qwen_grpo_vllm_test.sh")

    log_path = f"{root_path}/qwen2_vllm_log/worker_0.log"
    check_pair = {"Save checkpoints in": 1}
    check_values = [
        "generate parallel_config:{'data_parallel': 4, 'model_parallel': 2, 'pipeline_stage': 1}",
        "actor parallel_config:{'data_parallel': 1, 'model_parallel': 4, 'pipeline_stage': 2,"
        " 'use_seq_parallel': False, 'micro_batch_num': 2, 'vocab_emb_dp': False}",
        "parallel_config:{'data_parallel': 4, 'model_parallel': 2, 'pipeline_stage': 1}",
        # recompute config
        "grpo_config.ref_config.recompute_config:{'recompute': False, 'select_recompute': False, "
        "'select_comm_recompute': False, 'parallel_optimizer_comm_recompute': False, 'mp_comm_recompute': True, "
        "'recompute_slice_activation': False}",
        "total_ref_batch_size: ref_model_batch_size * ref_dp, 2 * 4 = 8",
    ]

    check_log(log_path, check_pair, check_values)
