#!/bin/bash
# Note: Only training network can be dryrun, inference process cannot be dryrun

export PYTHONPATH=/path/to/mindrlhf:/path/to/mindformers/:$PYTHONPATH
export GLOG_v=2
export MS_SIMULATION_LEVEL=1

msrun --worker_num=8 --local_worker_num=8 \
--master_addr=127.0.0.1 --master_port=9887 \
--join=True --log_dir=./prof_vllm_log \
--sim_level=$MS_SIMULATION_LEVEL --sim_rank_id=0 \
examples/grpo/qwen_grpo_tutorial/main.py \
--config examples/grpo/qwen_grpo_tutorial/grpo_config.yaml \
--tokenizer_dir /path/to/configs/ \
--dataset_file /path/to/limr_template_qwenr1.mindrecord \
--save_checkpoint_dir /path/to/train \
--actor_checkpoint_path "/path/to/qwen25_7b" \
--ref_checkpoint_path "/path/to/qwen25_7b" \
--generate_checkpoint_path "/path/to/qwen25_7b" > dryrun.log 2>&1 &
