#!/bin/bash
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

export GLOG_v=2
export PYTHONPATH=/path/to/mindrlhf/:$PYTHONPATH
export PYTHONPATH=/path/to/mindformers/:$PYTHONPATH
# use for vllm
export vLLM_MODEL_BACKEND=MindFormers
export MINDFORMERS_MODEL_CONFIG="path/to/predict_deepseek3_671b.yaml"
# 性能优化
export MS_DEV_RUNTIME_CONF="parallel_dispatch_kernel:True"
export MS_ALLOC_CONF="enable_vmm:true"
export ENABLE_LAZY_INLINE_NO_PIPELINE=1
export MS_DEV_RUNTIME_CONF="multi_stream:true"
#ray组网相关
export GLOO_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
# 关闭多机lccl
export MS_ENABLE_LCCL=off
export MS_DEV_HOST_BLOCKING_RUN=1

master_ip=127.0.0.1
node_rank=$1

msrun --worker_num=512 --local_worker_num=16 \
--master_addr=$master_ip --node_rank=$node_rank --master_port=9887 \
--join=True --log_dir=./prof_vllm_log \
examples/grpo/deepseekv3_grpo_tutorial/main.py \
--config examples/grpo/deepseekv3_grpo_tutorial/grpo_config.yaml \
--tokenizer_dir /path/to/tokenizer.json  \
--dataset_file /path/to/dkv3.mindrecord \
--save_checkpoint_dir /path/to/save/ckpt \
--actor_checkpoint_path /path/to/train/ckpt \
--ref_checkpoint_path /path/to/train/ckpt \
--generate_checkpoint_path /path/to/infer/ckpt \
--verifier_function "accuracy_reward,format_reward" \
--verifier_weight "1.0,1.0" > vllm.log 2>&1 &
