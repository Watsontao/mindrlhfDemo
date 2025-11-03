#!/bin/bash

export MS_ALLOC_CONF="memory_tracker:False,enable_vmm:True"
export GLOG_v=2

export vLLM_MODEL_BACKEND=MindFormers
export HCCL_EXEC_TIMEOUT=7200
export MS_JIT_MODULES=vllm_mindspore,research
export MS_ENABLE_LCCL=off

root_path="$(realpath "$(dirname "$0")")"
root_path=$root_path/../../../
cd $root_path
export PYTHONPATH=$root_path:$PYTHONPATH  # define mindrlhf path

export ASCEND_HOME_PATH=/path/to/ascend # need modify
export MINDFORMERS_PATH=/path/to/mindformers # need modify
export MSADAPTER_PATH=/path/to/msadapter # need modify (msadapter lib path)
export QWEN_MODEL_PATH=/path/to/qwen25_7b # need modify (only support huggingface model path)
export DATASET_FILE=/path/to/limr_template_qwenr1.mindrecord # need modify
export SAVE_CHECKPOINT_DIR=/path/to/train # need modify

export PYTHONPATH=$MSADAPTER_PATH:$MINDFORMERS_PATH:$PYTHONPATH

msrun --worker_num=8 --local_worker_num=8 \
--master_addr=127.0.0.1 --master_port=9887 \
--join=True --log_dir=./prof_vllm_log \
examples/grpo/qwen_grpo_tutorial/main.py \
--config examples/grpo/qwen_grpo_tutorial/grpo_config.yaml \
--tokenizer_dir $QWEN_MODEL_PATH \
--dataset_file $DATASET_FILE \
--save_checkpoint_dir $SAVE_CHECKPOINT_DIR \
--actor_checkpoint_path $QWEN_MODEL_PATH \
--ref_checkpoint_path $QWEN_MODEL_PATH \
--generate_checkpoint_path $QWEN_MODEL_PATH \
--verifier_function "qwen_accuracy_reward,format_reward" \
--verifier_weight "1.0,1.0" > vllm.log 2>&1 &
