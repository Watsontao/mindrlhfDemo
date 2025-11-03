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
import gc
import vllm_mindspore
import vllm
import torch
from .parallel_state import init_model_parallel_group, init_group_coordinator, initialize_parallel_state
from .mf_model import mf_model_base_init

vllm.distributed.parallel_state.init_model_parallel_group = init_model_parallel_group
vllm.distributed.parallel_state.GroupCoordinator.__init__ = init_group_coordinator
vllm_mindspore.model_executor.models.mf_models.mf_model_base.MfModelBase.__init__ = mf_model_base_init

_initialize_kv_cache = vllm.v1.worker.gpu_model_runner.GPUModelRunner.initialize_kv_cache
def initialize_kv_cache(self, kv_cache_config):
    self.kv_cache_config = kv_cache_config
    _initialize_kv_cache(self, kv_cache_config)
vllm.v1.worker.gpu_model_runner.GPUModelRunner.initialize_kv_cache = initialize_kv_cache

def init_cache_engine(self):
    model_runner = self.llm_engine.model_executor.driver_worker.worker.model_runner
    model = model_runner.model
    for fake_attention in model.kv_caches:
        fake_attention.kv_cache = [
            (
                torch.zeros(fake_attention.kv_shape, dtype=torch.bfloat16, device="Ascend"),
                torch.zeros(fake_attention.kv_shape, dtype=torch.bfloat16, device="Ascend"),
            )
            for _ in range(1) # FIXME: not support pp now.
        ]
    model_runner.initialize_kv_cache(model_runner.kv_cache_config)
    model.mf_kvcaches_init = False  # mindformers model


def free_cache_engine(self):
    model_runner = self.llm_engine.model_executor.driver_worker.worker.model_runner
    model = model_runner.model
    model_runner.kv_caches = []
    for fake_attention in model.kv_caches:
        fake_attention.kv_cache = None
    gc.collect()


def pre_process_inputs(self, prompt, valid_length_each_example):
    batch_size = len(prompt)
    token_ids = []
    for i in range(batch_size):
        max_id = valid_length_each_example[i]
        token_ids.append(prompt[i][:max_id])
    return token_ids


vllm.LLM.init_cache_engine = init_cache_engine
vllm.LLM.free_cache_engine = free_cache_engine
vllm.LLM.pre_process_inputs = pre_process_inputs


def post_process_outputs(request_outputs):
    output_token_ids = []
    for request_output in request_outputs:  # List[RequestOutput]
        outputs = request_output.outputs
        for output in outputs:  # List[CompletionOutput], usually len == 1
            output_token_ids.append(torch.tensor(output.token_ids))

    return output_token_ids


def set_device(device):
    # Avoid duplicate settings
    pass


torch.cuda.set_device = set_device
