#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
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
"""Qwen2 Sample Patch."""

import os
from typing import Optional, Iterable, Set, Tuple
import numpy as np

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size

import mindspore as ms
from mindspore import Tensor, runtime, JitConfig
from mindspore.common.api import _pynative_executor

from mindformers.tools.register.config import MindFormerConfig
from mindformers.core.context import build_mf_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.models.llama import LlamaConfig as LlamaConfig_MF
from mindformers import logger
from research.qwen2_5.infer.qwen2_5 import (
    ParallelQwenForCausalLM as ParallelQwenForCausalLM_MF,
)


def sample(
    cls,
    logits: Tensor,
    sampling_metadata: SamplingMetadata,
) -> Optional[SamplerOutput]:
    _pynative_executor.sync()
    runtime.synchronize()
    next_tokens = cls.sampler(logits, sampling_metadata)
    _pynative_executor.sync()
    runtime.synchronize()
    return next_tokens


# pylint: disable=W0613
def load_weights(cls, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
    """When resume training, skip vllm loading infer model weights"""
    pass
