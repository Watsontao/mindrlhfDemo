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
# limitations under the License.
# ============================================================================
"""
MindRLHF config
"""
__all__ = []

from dataclasses import asdict, make_dataclass

import mindspore
import mindspore.nn as nn
from mindspore.dataset import GeneratorDataset, MindDataset
from mindspore.dataset.transforms import TypeCast
from mindspore.nn import PipelineCell, MicroBatchInterleaved
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindrlhf.utils.adam import AdamWeightDecayOp
from mindrlhf.utils.dataset import GRPOIteratorStore
from mindrlhf.utils.utils import LearningRate, FP32StateAdamWeightDecay
from mindrlhf.wrapper import TrainOneStepWithLossScaleGRPO, TrainPipelineWithLossScaleCellGRPO


def set_weight_decay(params, is_use_other_params=True):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    """

    def decay_filter(x):
        return "layernorm" not in x.name.lower() and "bias" not in x.name.lower()

    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    if is_use_other_params:
        group_params = [
            {"params": decay_params, "weight_decay": 1e-1},
            {"params": other_params, "weight_decay": 0.0},
            {"order_params": params},
        ]
    else:
        # use for deepseek
        group_params = [{"params": decay_params, "weight_decay": 1e-1}, {"order_params": params}]
    return group_params


def combine_grpo_config(grpo_config, model_config):
    """
    Combine grpo config and model config.

    Args:
        grpo_config: Configuration of the GRPO algorithm.
        model_config: model_config.

    Returns:
        Configure after combination.
    """
    config_temp = asdict(grpo_config)
    for k, v in model_config.to_dict().items():
        if k not in config_temp:
            config_temp[k] = v
    config_temp["max_prompt_length"] = config_temp["seq_length"] - config_temp["max_decode_length"]
    grpo_config_ = make_dataclass("GRPOConfig", [(key, type(value)) for key, value in config_temp.items()])
    return grpo_config_(**config_temp)


def init_grpo_dataset(trainer):
    """
    Init grpo dataset.
    """
    grpo_config = trainer.grpo_config
    sft_model_config = trainer.sft_model_config_train
    column_names = [
        "prompt_completion_ids",
        "responses_mask",
        "ref_per_token_logps",
        "advantages",
        "actual_sequence_length",
        "sample_index",
        "sample_valid_length",
    ]
    if not trainer.store:
        dataset = MindDataset(dataset_files=grpo_config.save_data_file, shuffle=False)
        dataset = dataset.project(columns=column_names)
    else:
        pipeline = GRPOIteratorStore(trainer.store)
        dataset = GeneratorDataset(pipeline, column_names=column_names)
    type_cast_op_int32 = TypeCast(mindspore.int32)
    type_cast_op_fp16 = TypeCast(mindspore.float16)
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="prompt_completion_ids")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="responses_mask")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="ref_per_token_logps")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="advantages")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="actual_sequence_length")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="sample_index")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="sample_valid_length")
    micro_batch_num = 1
    if sft_model_config.parallel_config.pipeline_stage > 1:
        micro_batch_num = sft_model_config.parallel_config.micro_batch_num
    dataset = dataset.batch(
        batch_size=grpo_config.batch_size * sft_model_config.parallel_config.data_parallel * micro_batch_num
    )
    return dataset


def init_grpo_network_and_optimizer(trainer):
    """init grpo network and optimizer"""
    sft_model_config = trainer.sft_model_config_train
    grpo_config = trainer.grpo_config
    if sft_model_config.parallel_config.pipeline_stage > 1:
        grpo_with_loss_net = PipelineCell(
            MicroBatchInterleaved(trainer.grpo_model_train, grpo_config.micro_batch_interleaved),
            sft_model_config.parallel_config.micro_batch_num,
        )
    else:
        grpo_with_loss_net = trainer.grpo_model_train
    grpo_with_loss = _VirtualDatasetCell(grpo_with_loss_net)
    lr = LearningRate(
        learning_rate=grpo_config.start_lr,
        end_learning_rate=grpo_config.end_lr,
        warmup_steps=grpo_config.warmup_step,
        decay_steps=grpo_config.decay_steps,
    )
    params = grpo_with_loss.trainable_params()
    if trainer.sft_model_config_train.model_name == "deepseek_training":
        group_params = set_weight_decay(params, is_use_other_params=False)
    else:
        group_params = set_weight_decay(params)

    if grpo_config.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    elif grpo_config.opt_offload:
        optimizer = AdamWeightDecayOp(
            group_params,
            learning_rate=lr,
            eps=grpo_config.eps,
            beta1=grpo_config.beta1,
            beta2=grpo_config.beta2,
            param_init_type=sft_model_config.param_init_type,
        )
    else:
        optimizer = FP32StateAdamWeightDecay(
            group_params, learning_rate=lr, beta1=grpo_config.beta1, beta2=grpo_config.beta2, eps=grpo_config.eps
        )

    loss_scale_value = grpo_config.actor_config.loss_scale_value
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value, scale_factor=2, scale_window=1000)

    if sft_model_config.parallel_config.pipeline_stage > 1:
        grpo_with_grad = TrainPipelineWithLossScaleCellGRPO(
            grpo_with_loss,
            optimizer=optimizer,
            scale_sense=update_cell,
            micro_batch_num=sft_model_config.parallel_config.micro_batch_num
        )
    else:
        grpo_with_grad = TrainOneStepWithLossScaleGRPO(
            grpo_with_loss,
            optimizer=optimizer,
            scale_sense=update_cell,
            use_clip_grad=True
        )
    return grpo_with_grad
