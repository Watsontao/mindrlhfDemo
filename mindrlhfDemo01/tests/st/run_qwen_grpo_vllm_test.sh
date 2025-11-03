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

export MS_ENABLE_LCCL=off
export GLOG_v=2

WORKDIR="$(realpath "$(dirname "$0")")"
echo "WORKDIR is $WORKDIR"
cd $WORKDIR
export MINDRLHF_PATH=$WORKDIR/../../
export MINDFORMERS_PATH=$WORKDIR/mindformers/
export PYTHONPATH=$MINDRLHF_PATH:$MINDFORMERS_PATH:$PYTHONPATH
echo "PYTHONPATH is $PYTHONPATH"

# vllm config
export vLLM_MODEL_BACKEND=MindFormers
export HCCL_EXEC_TIMEOUT=7200
export MS_JIT_MODULES=vllm_mindspore,research
export MS_MEMORY_STATISTIC=1
jsonl_path="$WORKDIR/qwen2_5/mini_gsm8k.jsonl"
vocab_path="$WORKDIR/qwen2_5_vllm/vocab.json"
merges_path="$WORKDIR/qwen2_5_vllm/merges.txt"
mkdir -p $WORKDIR/dataset/
data_path="$WORKDIR/dataset/mini_gsm8k.mindrecord"

python ./qwen2_5/rlhf_data.py \
--vocab_path $vocab_path \
--merges_file_path $merges_path \
--file_path $jsonl_path \
--output_path $data_path > $WORKDIR/data_preprocess.log

mkdir -p $WORKDIR/grpo_data

msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 \
--master_port=9191 --join=True --log_dir=$WORKDIR/qwen2_vllm_log \
./qwen2_5/grpo_train.py \
--config ./qwen2_5_vllm/grpo_config_st.yaml \
--dataset_file $data_path \
--save_checkpoint_dir $WORKDIR/ckpt/train \
--tokenizer_dir "$WORKDIR/qwen2_5_vllm" \
--vllm_test \
--actor_checkpoint_path "" \
--ref_checkpoint_path "" \
--generate_checkpoint_path "$WORKDIR/qwen2_5_vllm"
