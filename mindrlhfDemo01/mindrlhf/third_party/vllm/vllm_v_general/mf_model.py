# Copyright 2025 Huawei Technologies Co., Ltd
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
from typing import Iterable, Set, Tuple, List, Union, Optional

import mindspore as ms
from mindspore import Tensor
from mindspore.common.api import _pynative_executor
from mindformers.tools.register.config import MindFormerConfig
from mindformers.core.context import build_mf_context
from mindformers.core.parallel_config import build_parallel_config
from research.qwen2_5.infer.qwen2_5 import (
    ParallelQwenForCausalLM as ParallelQwenForCausalLM_MF,
)

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.distributed import get_tensor_model_parallel_world_size

# TODO
from vllm_mindspore.model_executor.models.attention_mask import LowerTriangularMask
from vllm_mindspore.model_executor.models.mf_models.mf_model_base import MfModelBase

from mindrlhf.utils import get_infer_dp_size


def mf_model_base_init(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
    super(MfModelBase, self).__init__(vllm_config=vllm_config, prefix=prefix)
    # Diff: delete set_cpu_affinity
    self.mf_config = MindFormerConfig(os.getenv("MINDFORMERS_MODEL_CONFIG"))
    build_mf_context(self.mf_config)
    build_parallel_config(self.mf_config)
    self.mf_config.model.model_config.parallel_config = self.mf_config.parallel_config
    self.mf_config.model.model_config.parallel_config.model_parallel = get_tensor_model_parallel_world_size()
    self.mf_config.model.model_config.parallel_config.data_parallel = get_infer_dp_size()
    self.mf_config.model.model_config.parallel_config.pipeline_stage = 1
    self._generate_model_config()
    self.casual_mask = LowerTriangularMask(dtype=self.mf_model_config.compute_dtype,
                                           max_model_len=self.model_config.max_model_len)
    self.network, self.lm_head = self._create_network()
    self._set_dynamic_inputs()
