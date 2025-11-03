#!/bin/bash

export MS_ALLOC_CONF="memory_tracker:False,enable_vmm:True"
export GLOG_v=2
export MS_DEV_DUMP_IR_PASSES="hwopt_d_after_stream_assign,valid,graph_build"
export MS_DEV_RUNTIME_CONF="memory_statistics:False"
export HCCL_IF_BASEE_PORT=60009

root_path="$(realpath "$(dirname "$0")")"
root_path=$root_path/../../../
cd $root_path
WORKDIR=$PWD/../
export PYTHONPATH=$root_path:$PYTHONPATH  # define mindrlhf path
export MINDFORMERS_PATH=$WORKDIR/mindformers/
export PYTHONPATH=$MINDFORMERS_PATH:$PYTHONPATH  # define mindformers path

noderank=$1
master_addr=$2
echo $noderank
msrun --worker_num=32 --local_worker_num=8 --node_rank=$noderank --master_addr=$master_addr \
--master_port=9190 --join=False --log_dir=./qwen2_one_log_$noderank \
examples/grpo/qwen_grpo_tutorial/main.py \
--config examples/grpo/qwen_grpo_tutorial/grpo_config.yaml \
--tokenizer_dir /path/to/ \
--dataset_file /path/to/grpo_data/cvalues_one.mindrecord \
--save_checkpoint_dir /path/to/ckpt/train \
--ref_checkpoint_path "/path/to/ckpt/infer" \
--actor_checkpoint_path "/path/to/ckpt/train/" \
--generate_checkpoint_path "/path/to/ckpt/infer" \
--verifier_function "accuracy_reward,format_reward" \
--verifier_weight "1.0,1.0"
