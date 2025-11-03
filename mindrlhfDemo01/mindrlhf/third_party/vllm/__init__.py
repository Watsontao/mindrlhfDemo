# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from importlib.metadata import version
from vllm_mindspore.model_executor.models.mf_models.qwen2 import Qwen2ForCausalLM
from packaging import version as vs

from .qwen2 import sample
Qwen2ForCausalLM.sample = sample

package_version = version("vllm")

if package_version.startswith("0.8"):
    from vllm import LLM
else:
    raise ValueError(f"Not support vllm version: {package_version}")

import vllm
from .ascend import check_and_update_config

vllm.config.current_platform.check_and_update_config = check_and_update_config
vllm.platforms.current_platform.check_and_update_config = check_and_update_config
vllm.utils.current_platform.check_and_update_config = check_and_update_config