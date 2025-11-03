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
"""MindRLHF utils."""
__all__ = [
    "AdamWeightDecayOp",
    "TransformParametersD2D",
    "TransformParametersD2DForDSv3",
    "MetricData",
    "GRPOIteratorStore",
    "GeneratorMixin",
    "set_pipeline_parallel_context",
    "is_last_stage",
    "is_first_stage",
    "FP32StateAdamWeightDecay",
    "TimePoint",
    "LearningRate",
    "GlobalNorm",
    "ClipByGlobalNorm",
    "transfer_from_str_to_bool",
    "ckpt_transfer_for_generate",
    "yaml_to_dataclass",
    "set_perf_stats",
    "_get_pipeline_group",
    "convert_index_json_total",
    "save_prompt_completions_data",
    "add_metrics_to_tensorboard",
    "get_dp_rank",
    "get_checkpoint_name",
    "ensure_total_ckpt_is_less_than_limit",
    "load_param_to_net",
    "record_last_ckpt_to_json",
    "TimeConsumingCollector",
    "profiler_start",
    "profiler_step",
    "mstx_timer_decorator",
    "set_pack_level",
    "set_infer_dp_size",
    "get_infer_dp_size"
]

from .dataset import GRPOIteratorStore
from .generator import GeneratorMixin
from .utils import (
    _get_pipeline_group,
    GlobalNorm,
    ClipByGlobalNorm,
    yaml_to_dataclass,
    set_perf_stats,
    convert_index_json_total,
    save_prompt_completions_data,
    add_metrics_to_tensorboard,
    get_dp_rank,
    get_checkpoint_name,
    ensure_total_ckpt_is_less_than_limit,
    load_param_to_net,
    record_last_ckpt_to_json,
    TimeConsumingCollector,
    transfer_from_str_to_bool,
    ckpt_transfer_for_generate,
    TimePoint,
    LearningRate,
    FP32StateAdamWeightDecay,
    set_pipeline_parallel_context,
    is_last_stage,
    is_first_stage,
    profiler_start,
    profiler_step,
    mstx_timer_decorator,
    set_pack_level,
    set_infer_dp_size,
    get_infer_dp_size
)
from .adam import AdamWeightDecayOp
from .transform_param import TransformParametersD2D, TransformParametersD2DForDSv3
from .metrcis import MetricData
