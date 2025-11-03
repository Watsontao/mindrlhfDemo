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
""" test Qwen GRPO inference  """
import os
import re

import pytest

infer_path = os.path.dirname(__file__)
st_path = os.path.abspath(os.path.join(infer_path, "../"))
mindrlhf_path = os.path.abspath(os.path.join(st_path, "../../"))
model_config = os.path.join(mindrlhf_path, "model_configs/qwen_grpo/qwen2_5_7b/predict_qwen2_5_7b_instruct.yaml")
grpo_config = os.path.join(mindrlhf_path, "examples/grpo/qwen_grpo_tutorial/grpo_config.yaml")
qwen2_5_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-7B-Instruct"
mindformers_path = os.path.join(st_path, "mindformers")

pre_cmd = f"""
export PYTHONPATH={mindrlhf_path}:{mindformers_path}:$PYTHONPATH
export vLLM_MODEL_BACKEND=MindFormers
export vLLM_MODEL_MEMORY_USE_GB=40

export MS_DEV_RUNTIME_CONF="memory_statistics:True"
msrun --worker_num=8 --local_worker_num=8 \
--master_addr=127.0.0.1 --master_port=9887 \
--join=True --bind_core=True --log_dir=%s \
infer_main.py --config {grpo_config} \
--model_config {model_config} \
--tokenizer_dir {qwen2_5_path} %s > log 2>&1\
"""


def run(log_dir, params, check_value):
    """run entrance"""
    log_path = os.path.join(infer_path, log_dir)
    complete_cmd = pre_cmd % (log_path, params)
    print(f"run cmd:\n{complete_cmd}")
    os.system(complete_cmd)
    log_file = os.path.join(log_path, "worker_0.log")
    with open(log_file, "r") as f:
        content = f.read()
        try:
            perf = re.findall(r"Generating elapsed time: (\d+\.\d+)", content)[-1]
            print(f"Generating elapsed time: {perf}")
            used_memory = re.search(r"MindSpore Used memory size: (\d+)M", content).group(1)
            print(f"MindSpore Used memory size: {used_memory}M")
        except (IndexError, AttributeError):
            assert False, f"please check errors in log"

        assert check_value in content, f"{check_value} is not in log, config may be not set right"


@pytest.mark.parametrize("dp,mp", [(4, 2), (8, 1)])
def test_generate_parallel(dp, mp):
    """
    Feature: infer case
    Description: set different data_parallel and model_parallel to test inference,
    check inference performance and memory.
    Expectation: Run success or failure.
    """
    log_dir = f"log_{dp}_{mp}"
    params = f"--data_parallel {dp} --model_parallel {mp}"
    check_value = "generate parallel_config:{'data_parallel': %d, 'model_parallel': %d" % (dp, mp)
    run(log_dir, params, check_value)


@pytest.mark.parametrize(
    "temperature, repetition_penalty, top_p, top_k",
    [(1.0, 1.05, 0.8, 20), (0.8, 1.0, 0.8, 20), (0.8, 1.05, 1.0, 20), (0.8, 1.05, 0.8, -1)],
)
def test_generate_sampling(temperature, repetition_penalty, top_p, top_k):
    """
    Feature: infer case
    Description: set different generate sampling config to test inference, check inference performance and memory.
    Expectation: Run success or failure.
    """
    log_dir = f"log_{temperature}_{repetition_penalty}_{top_p}_{top_k}"
    params = (
        f"--temperature {temperature} " f"--repetition_penalty {repetition_penalty} " f"--top_p {top_p} --top_k {top_k}"
    )
    check_value = (
        f"temperature={temperature}, repetition_penalty={repetition_penalty}, " f"top_p={top_p}, top_k={top_k}"
    )
    run(log_dir, params, check_value)


@pytest.mark.parametrize("use_vllm", [0, 1])
def test_use_vllm(use_vllm):
    """
    Feature: infer case
    Description: set use_vllm to 0 or 1 to test inference, check inference performance and memory.
    Expectation: Run success or failure.
    """
    log_dir = f"log_vllm_{use_vllm}"
    params = f"--use_vllm {use_vllm}"
    check_value = f"use_vllm={use_vllm}"
    run(log_dir, params, check_value)
