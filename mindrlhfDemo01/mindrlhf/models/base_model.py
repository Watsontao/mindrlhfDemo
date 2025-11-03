# Copyright 2024 Huawei Technologies Co., Ltd
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
# limitations under the License
# ============================================================================
"""
MindRLHF Base Model
"""
import mindspore.nn as nn
from mindformers import LlamaForCausalLM
from research.qwen2_5.infer.qwen2_5 import ParallelQwenForCausalLM
from research.deepseek3.deepseek3_model_infer import InferenceDeepseekV3ForCausalLM
from research.deepseek3.deepseek3_model_train import TrainingDeepseekV3ForCausalLM



class BaseModel(nn.Cell):
    """BaseModel"""

    def __init__(self):
        super(BaseModel, self).__init__()
        self._model_list = [
            "llama",
            "deepseek_infer",
            "deepseek_training"
        ]

    def select_actor_model(self, model_config):
        """
        select actor model
        """
        self.model_type = None
        if not model_config.model_name:
            raise NotImplementedError("model_name in actor/reference model is None")
        for model in self._model_list:
            if model in model_config.model_name:
                self.model_type = model
        if not self.model_type:
            raise NotImplementedError(
                "only support {}".format(" ".join(self._model_list))
            )
        if self.model_type == "llama":
            if model_config.use_past:
                self.model = ParallelQwenForCausalLM(model_config)
            else:
                self.model = LlamaForCausalLM(model_config)
            self.backbone = self.model.model
            self.lm_head = self.model.lm_head
        elif "deepseek" in self.model_type:
            if model_config.use_past:
                self.model = InferenceDeepseekV3ForCausalLM(model_config)
            else:
                self.model = TrainingDeepseekV3ForCausalLM(model_config)
            self.backbone = self.model.model
            self.lm_head = self.model.lm_head

    def select_critic_model(self, model_config):
        """
        select critic model
        """
        self.model_type = None
        if not model_config.model_name:
            raise NotImplementedError("model_name in critic model is None")
        for model in self._model_list:
            if model in model_config.model_name:
                self.model_type = model
        if not self.model_type:
            raise NotImplementedError(
                "only support {}".format(" ".join(self._model_list))
            )
        if self.model_type == "llama":
            if model_config.use_past:
                self.model = ParallelQwenForCausalLM(model_config)
            else:
                self.model = LlamaForCausalLM(model_config)
            self.backbone = self.model.model
        elif "deepseek" in self.model_type:
            if model_config.use_past:
                self.model = InferenceDeepseekV3ForCausalLM(model_config)
            else:
                self.model = TrainingDeepseekV3ForCausalLM(model_config)
            self.backbone = self.model.model
            self.lm_head = self.model.lm_head

    def select_reward_model(self, model_config):
        """
        select reward model
        """
        self.model_type = None
        if not model_config.model_name:
            raise NotImplementedError("model_name in reward model is None")
        for model in self._model_list:
            if model in model_config.model_name:
                self.model_type = model
        if not self.model_type:
            raise NotImplementedError(
                "only support {}".format(" ".join(self._model_list))
            )
        if self.model_type == "llama":
            if model_config.use_past:
                self.model = ParallelQwenForCausalLM(model_config)
            else:
                self.model = LlamaForCausalLM(model_config)
            self.backbone = self.model.model
        elif "deepseek" in self.model_type:
            if model_config.use_past:
                self.model = InferenceDeepseekV3ForCausalLM(model_config)
            else:
                self.model = TrainingDeepseekV3ForCausalLM(model_config)
            self.backbone = self.model.model
            self.lm_head = self.model.lm_head
