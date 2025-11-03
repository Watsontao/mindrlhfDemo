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
    "mstx_timer_decorator",
    "set_infer_dp_size",
    "get_infer_dp_size",
    "MsProbe",
]

import os
import json
import re
import time
import hashlib
import copy
import stat
from glob import glob
from typing import Dict
from functools import wraps

import yaml
from safetensors import safe_open
import numpy as np
import mindspore.nn as nn
import mindspore as ms
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.context import ParallelMode
from mindspore.communication.management import get_rank, get_group_size, create_group
from mindspore.nn import AdamWeightDecay
from mindspore.common import Parameter, ParameterTuple
from mindspore.common.initializer import initializer
import mindspore.communication.management as D
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore import context
import mindspore.common.dtype as mstype
from mindspore.common.api import _pynative_executor

from mindformers.tools.ckpt_transform import TransformCkpt
from mindformers import logger

import mindrlhf.utils.reshard_optimizer as reshard_optimizer
from mindrlhf.configs.grpo_configs import GRPOConfig


class MsProbe:
    r"""
    Centralized Debugging Toolkit for MindSpore Applications

    This class provides a unified interface for diagnostic operations during training,
    particularly in distributed environments. It serves as a wrapper for:
    - Configuration saving
    - Tensor data serialization
    - String list handling
    - Execution step marking

    Key differences from standard debugging tools:
    - Enforces a whitelist root (`config.dump_path`) for all file I/O
    - Implements retry-safe directory creation and robust filename sanitization
    - Minimizes performance impact in production environments
    - Compatible with FP16/FP32 mixed-precision workflows

    Usage:
        Configures via config_init(), then call save methods during training:
        >>> MsProbe.config_init(msprobe_config)
        >>> MsProbe.save_data("activation", layer_output)
        >>> MsProbe.save_string_list("responses", completion_texts)
        >>> MsProbe.step()  # Mark training step boundaries
    """

    config = None
    enabled = False

    @classmethod
    def config_init(cls, msprobe_config):
        """
        Initialize the MsProbe debugging toolkit with configuration

        Args:
            msprobe_config: Configuration object containing:
                - msprobe: Global enable flag
                - dump_path: Base directory for debug outputs
                - configurations_dump: Enable config saving
                - key_data_dump: Enable tensor data saving

        Behavior:
            - Sets global enabled state based on configuration
            - Stores configuration for future operations
            - Ensures the whitelist root directory exists
            - Prints activation status when enabled
        """
        if not msprobe_config.msprobe:
            return
        cls.config = msprobe_config
        cls.enabled = True

        dangerous_paths = ["/etc", "/usr", "/bin", "/sbin", "/boot", "/proc", "/sys"]
        if cls.config.dump_path:
            for dangerous_path in dangerous_paths:
                if dangerous_path in cls.config.dump_path:
                    raise PermissionError(
                        f"Security violation: Cannot use system directory '{dangerous_path}' "
                        f"as dump path. Attempted path: '{cls.config.dump_path}'"
                    )

            # Pre-create the dump root to avoid first-write errors
            os.makedirs(cls.config.dump_path, exist_ok=True)
        print("msprobe enabled")

    @classmethod
    def save_json(cls, data, path_in_dump_dir: str, indent: int = 4):
        """
        Safely write JSON under the whitelist root.

        Args:
            data: Configuration/data object (dict or serializable)
            path_in_dump_dir: Relative subpath under dump_path, e.g.:
                              "configurations.json" or "logs/run1/meta.json"
            indent: JSON indentation level

        Behavior:
            - Normalizes and validates target path is inside <dump_path>
            - Recursively converts non-serializable objects to strings
            - Creates parent directories as needed
            - Raises PermissionError on whitelist escape
        """
        if not cls.enabled:
            return

        resolved_path = cls._resolve_under_base(cls.config.dump_path, path_in_dump_dir)

        dir_path = os.path.dirname(resolved_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        converted = cls._json_convert(data)
        with open(resolved_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, indent=indent, ensure_ascii=False)

    @classmethod
    def save_configs(cls, data):
        """
        Save configuration data to JSON format

        Args:
            data: Configuration data (dict or serializable object)

        Behavior:
            - Saves to <dump_path>/configurations.json
            - Only executes if MsProbe is enabled and config saving is active
            - Provides error details on failure
        """
        if not cls.enabled:
            return
        if not cls.config.configurations_dump:
            return
        try:
            cls.save_json(data, "configurations.json")
        except Exception as e:  # pylint: disable=W0703
            print(f"Failed to save configurations. Error: {str(e)}")

    @classmethod
    def step(cls):
        """
        Mark an execution step for temporal debugging

        Behavior:
            - Creates a logical time marker in debug outputs
            - Synchronizes distributed execution points
            - Only executes if MsProbe is enabled
        """

        if not cls.enabled:
            return
        try:
            from msprobe.mindspore import step
        except ImportError:
            print("Failed to import step from msprobe.mindspore")
            return
        step()

    @classmethod
    def save_data(cls, name, data):
        """
        Save tensor data to NPY format

        Args:
            name: Identifier for the saved data
            data: MindSpore Tensor or convertible object

        Behavior:
            - Saves to <dump_path>/data/<name>.npy
            - Only executes if MsProbe is enabled and data saving is active
            - Provides detailed error messages on failure
        """

        if not cls.enabled:
            return
        if not cls.config.key_data_dump:
            return
        try:
            from msprobe.mindspore import save
        except ImportError:  # pylint: disable=W0703
            print("Failed to import save from msprobe.mindspore")
            return
        try:
            safe_name = cls._sanitize_filename(name)
            # Target directory: <dump_path>/data/<safe_name>
            rel_dir = os.path.join("data", safe_name)
            abs_dir = cls._resolve_under_base(cls.config.dump_path, rel_dir)

            os.makedirs(abs_dir, exist_ok=True)
            save(abs_dir, safe_name, data)
        except PermissionError as e:
            print(f"Security error: {e}")
        except Exception as e:  # pylint: disable=W0703
            print(f"Save {data} failed. Error: {str(e)}")

    @classmethod
    def save_string_list(cls, name, string_list):
        """
        Save a list of strings as serialized tensors

        Args:
            name: Base identifier for the string collection
            string_list: List of string elements to save

        Behavior:
            - Validates input is a list
            - Converts each string to UTF-8 byte tensor
            - Saves each string as a separate tensor
            - Skips non-string elements with warning
            - Provides error details on failure
        """

        if not cls.enabled:
            return
        if not cls.config.key_data_dump:
            return

        try:
            # Validate input type
            if not isinstance(string_list, list):
                print(f"Warning: {name} is not a list, skipping save")
                return

            # Process each string in the list
            for text in string_list:
                if not isinstance(text, str):
                    print(f"Warning: Skipping non-string item in {name}: {text}")
                    continue

                # Convert string to byte tensor
                encoded_bytes = np.array(list(text.encode("utf-8")), dtype=np.uint8)
                text_tensor = Tensor(encoded_bytes)
                # Save the tensor
                cls.save_data(name, text_tensor)

        except Exception as e:  # pylint: disable=W0703
            print(f"Failed to save string list {name}: {e}")

    # -----------------------------
    # Internal helpers (whitelisting)
    # -----------------------------
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """
        Sanitize a user-controlled filename component.

        Behavior:
            - Only allow letters/digits/._-
            - Replace other characters with underscores
            - Strip leading dots to avoid hidden files
            - Limit length to 255 characters
        """
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(name))
        safe = safe.lstrip(".")
        return safe[:255]

    @staticmethod
    def _resolve_under_base(base_dir: str, target_path: str) -> str:
        """
        Normalize `target_path` to an absolute real path and ensure it stays
        within the whitelist root `base_dir`. Prevents path traversal and symlink escapes.

        Args:
            base_dir: Whitelist root directory (e.g., MsProbe.config.dump_path)
            target_path: Relative or absolute path to resolve/validate

        Returns:
            Absolute, real path confined to the whitelist root

        Raises:
            ValueError: If base_dir is not configured
            PermissionError: If the path escapes the whitelist
        """
        if not base_dir:
            raise ValueError("Whitelist base_dir is not configured.")
        base = os.path.realpath(os.path.abspath(base_dir))
        candidate = target_path if os.path.isabs(target_path) else os.path.join(base, target_path)
        candidate = os.path.realpath(os.path.abspath(candidate))
        try:
            inside = os.path.commonpath([base, candidate]) == base
        except ValueError:
            # Different drives on Windows or invalid roots -> treat as escape
            inside = False
        if not inside:
            raise PermissionError(f"Path escapes whitelist: {target_path}")
        return candidate

    @staticmethod
    def _json_convert(obj):
        """Recursively convert objects to JSON-serializable formats"""
        if isinstance(obj, dict):
            # Process dictionaries
            return {k: MsProbe._json_convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            # Process lists and tuples
            return [MsProbe._json_convert(item) for item in obj]

        # Attempt direct serialization, convert to string on failure
        try:
            json.dumps(obj)
            return obj
        except Exception:  # pylint: disable=W0703
            return str(obj)


PERF_STATS = False
INFER_DP_SIZE = 1


def yaml_to_dataclass(file_path, dataclass_type):
    """
    Parse a YAML file into an instance of the specified dataclass.

    Parameters:
    file_path (str): Path to the YAML file.
    dataclass_type (type): The dataclass type to instantiate.

    Returns:
    dataclass_type: An instance of the dataclass.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
            return dataclass_type(**data)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except yaml.YAMLError as error:
        logger.error(f"Error parsing YAML file: {error}")
        return None
    except TypeError as error:
        logger.error(f"Data type mismatch: {error}")
        return None


def set_pipeline_parallel_context(ppo_config):
    """Set pipeline parallel context."""
    D.init()
    device_num = D.get_group_size()
    rank_id = D.get_rank()
    context.reset_auto_parallel_context()

    if hasattr(ppo_config.parallel_config, "optimizer_shard"):
        optimizer_shard = bool(ppo_config.parallel_config.get("optimizer_shard"))
    elif hasattr(ppo_config.parallel, "enable_parallel_optimizer"):
        optimizer_shard = bool(ppo_config.parallel.get("enable_parallel_optimizer"))
    else:
        optimizer_shard = True

    context.set_auto_parallel_context(
        parallel_mode=ppo_config.parallel_mode,
        gradients_mean=False,
        full_batch=bool(ppo_config.full_batch),
        loss_repeated_mean=True,
        device_num=device_num,
        enable_parallel_optimizer=optimizer_shard,
        pipeline_stages=ppo_config.parallel_config.get("pipeline_stage"),
        enable_alltoall=bool(ppo_config.enable_alltoall),
        strategy_ckpt_save_file="strategy.ckpt",
    )
    set_algo_parameters(elementwise_op_strategy_follow=True)
    _set_multi_subgraphs()
    return rank_id, device_num


def is_last_stage(pipeline_stage):
    """Check is last stage"""
    device_num = D.get_group_size()
    rank = D.get_rank()
    per_stage_num = int(device_num / pipeline_stage)
    return device_num - 1 - per_stage_num < rank <= device_num - 1


def is_first_stage(pipeline_stage):
    """Check is first stage"""
    device_num = D.get_group_size()
    rank = D.get_rank()
    per_stage_num = int(device_num / pipeline_stage)
    return rank < per_stage_num


class FP32StateAdamWeightDecay(AdamWeightDecay):
    r"""
    This class is almost same with the mindspore's AdamWeightDecay implements, the
    only difference is the optimizer's state will be always initialized with float32,
    where the original AdamWeightDecay will initialize the optimizer's state with float16,
    if the parameters are initialized with fp16.
    This setting will avoid overflow in training PanGu-Alpha model using fp16.
    """

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(FP32StateAdamWeightDecay, self).__init__(
            params, learning_rate=learning_rate, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay
        )

        self.moments1 = self.clone_state(self.parameters, prefix="adam_m", init="zeros")
        self.moments2 = self.clone_state(self.parameters, prefix="adam_v", init="zeros")

    def clone_state(self, parameter_tuple, prefix, init):
        r"""
        parameter_tuple: ParameterTuple. The parameters of the network
        prefix: str. The prefix name of the parameters
        init: str. The initialization method
        """
        new = []
        for old_param in parameter_tuple:
            new_state = Parameter(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.param_info = old_param.param_info.clone()
            if hasattr(old_param.param_info, "cloned_obj"):
                old_param.param_info.cloned_obj.append(new_state)
            else:
                old_param.param_info.cloned_obj = [new_state]
            new_state.is_init = False
            new_state.set_data(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.name = prefix + "." + new_state.name
            new.append(new_state)
        return ParameterTuple(new)


get_square_sum = C.MultitypeFuncGraph("get_square_sum")


@get_square_sum.register("Tensor", "Number")
def _get_square_sum(grad, value):
    norm = P.ReduceSum(False)(F.square(grad), ()) / value
    norm = F.expand_dims(F.cast(norm, mstype.float32), 0)
    return norm


apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")


@apply_global_norm.register("Bool", "Tensor", "Tensor", "Tensor")
def _apply_global_norm(enable_grad_fp16, clip_norm, global_norm, grad):
    if enable_grad_fp16:
        grad = P.Cast()(grad * clip_norm / global_norm, mstype.float16)
    else:
        grad = grad * clip_norm / global_norm
    return grad


def _get_model_parallel_group(mp):
    """

    Calculate the communication group of model parallel dim in one pipeline stage

    """
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    stage_id = rank // per_stage_device_nums
    local_stage_rank_id = rank % per_stage_device_nums
    index = local_stage_rank_id // mp
    group = range(0, mp)
    rank_str_list = [str(x + index * mp + stage_id * per_stage_device_nums) for x in group]
    rank_list_str = "-".join(rank_str_list)
    rank_list = [x + index * mp + stage_id * per_stage_device_nums for x in group]
    return rank_list, rank_list_str


def _get_pipeline_group():
    """

    Calculate the communication group between all pipeline stages

    """
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    local_stage_rank_id = rank % per_stage_device_nums
    group = range(0, stage_nums)
    rank_list = [local_stage_rank_id + x * per_stage_device_nums for x in group]
    rank_str_list = [str(local_stage_rank_id + x * per_stage_device_nums) for x in group]
    rank_list_str = "-".join(rank_str_list)
    return rank_list, rank_list_str


class GlobalNorm(nn.Cell):
    """
    Calculate the global norm value of given tensors
    """

    def __init__(self, params, config):
        super(GlobalNorm, self).__init__()
        self.norm = nn.Norm()
        self.hyper_map = C.HyperMap()
        self.is_pipeline = context.get_auto_parallel_context("pipeline_stages") > 1
        self.is_data_parallel = context.get_auto_parallel_context("parallel_mode") == ParallelMode.DATA_PARALLEL
        self.config = config
        self.group_size = 1
        if self.is_data_parallel:
            self.merge_op = P.identity()
        else:
            self.merge_op = P.AllReduce()
        if self.is_pipeline:
            if context.get_auto_parallel_context("enable_parallel_optimizer"):
                self.group_size = get_group_size() // config.parallel_config.pipeline_stage
            else:
                self.group_size = config.parallel_config.model_parallel
            group_list, group_name = _get_model_parallel_group(self.group_size)
            # In avoid of the group name too long
            sha256 = hashlib.sha256(group_name.encode()).hexdigest()  # sha256 len is 64
            logger.info(f"Creating hash value for the group_name hash({group_name})={sha256}")
            group_name = str(sha256)
            create_group(group_name, group_list)
            self.allreduce = P.AllReduce(group=group_name)
            pipeline_group_list, pipeline_group_name = _get_pipeline_group()
            sha256 = hashlib.sha256(pipeline_group_name.encode()).hexdigest()  # sha256 len is 64
            logger.info(f"Creating hash value for the group_name hash({pipeline_group_name})={sha256}")
            pipeline_group_name = str(sha256)
            create_group(pipeline_group_name, pipeline_group_list)
            self.allreduce2 = P.AllReduce(group=pipeline_group_name)
        else:
            self.group_size = get_group_size()
        self.allreduce_group_size = ()
        if self.is_data_parallel:
            self.allreduce_group_size = (1,) * len(params)
        else:
            self.allreduce_group_size = self._get_scale_for_gradient_norm(params)

    def construct(self, grads):
        """Calculate global norm construct"""
        square_sum = self.hyper_map(get_square_sum, grads, self.allreduce_group_size)
        square_reduce_sum = F.addn(square_sum)
        if self.is_pipeline:
            stage_square_reduce_sum = self.allreduce(square_reduce_sum)
            global_square_reduce_sum = self.allreduce2(stage_square_reduce_sum)
            global_norms = F.sqrt(global_square_reduce_sum)
        else:
            global_norms = F.sqrt(self.merge_op(square_reduce_sum))
        return grads, global_norms

    def _get_scale_for_gradient_norm(self, params):
        """
        get scale for gradient norm
        """
        allreduce_group_size = ()
        for x in params:
            if "projection.bias" not in x.name and "layernorm" not in x.name and "embedding_table" not in x.name:
                allreduce_group_size = allreduce_group_size + (1.0,)
            elif "embedding_table" not in x.name:
                allreduce_group_size = allreduce_group_size + (self.group_size * 1.0,)
            else:
                if (
                    not self.config.parallel_config.vocab_emb_dp
                    and "position_embedding.embedding_table" not in x.name
                    and "top_query_embedding_table" not in x.name
                ):
                    allreduce_group_size = allreduce_group_size + (self.config.parallel_config.data_parallel * 1.0,)
                else:
                    allreduce_group_size = allreduce_group_size + (self.group_size * 1.0,)
        return allreduce_group_size


class ClipByGlobalNorm(nn.Cell):
    """

    Clip grads by global norm

    """

    def __init__(self, params, config, clip_norm=1.0):
        super(ClipByGlobalNorm, self).__init__()
        self.global_norm = GlobalNorm(params, config)
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()
        if config.param_init_type == mstype.float16:
            self.enable_grad_fp16 = True
        else:
            self.enable_grad_fp16 = False

    def construct(self, grads):
        """Clip grads by global norm construct"""
        grads, global_norm_value = self.global_norm(grads)
        cond = P.GreaterEqual()(global_norm_value, self.clip_norm)
        global_norm = F.select(cond, global_norm_value, self.clip_norm)
        grads = self.hyper_map(F.partial(apply_global_norm, self.enable_grad_fp16, self.clip_norm, global_norm), grads)
        return grads, global_norm_value


class LearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for PanguAlpha network.
    """

    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power=1.0, use_cosine=True):
        super(LearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.cosine_decay_lr = CosineDecayLR(end_learning_rate, learning_rate, decay_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.use_cosine = use_cosine

    def construct(self, global_step):
        """dynamic learning rate"""
        if not self.use_cosine:
            decay_lr = self.decay_lr(global_step)
        else:
            decay_lr = self.cosine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


class TimePoint:
    """A helper function for recording the time spend."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def set_start(self):
        """Set the start time point"""
        self.start_time = time.time()

    def set_end(self):
        """Set the end time point"""
        self.end_time = time.time()

    def get_spend_time(self):
        """Get the time spend between end and start"""
        return self.end_time - self.start_time


def get_testing_dataset_path(dataset_name):
    """Get dataset path for testing."""
    dataset_dict = {"cvalues_1024": "/path/cvalues_1024.mindrecord", "cvalues_2048": "/path/cvalues_2048.mindrecord"}
    dataset = dataset_dict.get(dataset_name)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    return dataset


def get_valid_length_each_example(input_ids, pad_token_id):
    """get valid length and max length in a batch"""
    batch_size = input_ids.shape[0]
    valid_length_each_example = []
    for i in range(batch_size):
        # As the nonzero returns the index and we need length
        valid_length_each_example.append(np.max(np.argwhere(input_ids[i] != pad_token_id)) + 1)
    valid_length_each_example = np.array(valid_length_each_example)
    max_length = np.max(valid_length_each_example)
    return valid_length_each_example, max_length


def transfer_from_str_to_bool(input_str):
    """transfer from string to bool"""
    if isinstance(input_str, bool):
        return input_str
    if input_str.lower() == "true":
        return True
    if input_str.lower() == "false":
        return False
    raise ValueError(f"input_str {input_str} is not supported.")


def get_strategy(startegy_path, rank_id=None):
    """Merge strategy if strategy path is dir

    Args:
        startegy_path (str): The path of stategy.
        rank_id (int): The rank id of device.

    Returns:
        None or strategy path
    """
    if not startegy_path or startegy_path == "None":
        return None

    if not os.path.exists(startegy_path):
        raise ValueError(f"{startegy_path} not found!")

    if os.path.isfile(startegy_path):
        return startegy_path

    if os.path.isdir(startegy_path):
        if rank_id:
            merge_path = os.path.join(startegy_path, f"merged_ckpt_strategy_{rank_id}.ckpt")
        else:
            merge_path = os.path.join(startegy_path, f"merged_ckpt_strategy.ckpt")

        if os.path.exists(merge_path):
            os.remove(merge_path)

        ms.merge_pipeline_strategys(startegy_path, merge_path)
        return merge_path

    return None


def ckpt_transfer_for_generate(load_sft_checkpoint):
    """ckpt transfer for generate"""
    transform_ckpt = TransformCkpt(
        rank_id=get_rank(), world_size=get_group_size(), transform_process_num=get_group_size()
    )

    generate_ckpt = "./generate_ckpt"
    return transform_ckpt(
        src_checkpoint=load_sft_checkpoint,
        dst_checkpoint_dir=generate_ckpt,
        src_strategy="./train_policy_strategy/",
        dst_strategy="./generate_policy_strategy/",
        prefix="./generate_policy_",
    )


def set_perf_stats(grpo_config: GRPOConfig):
    """set performance stats"""
    global PERF_STATS
    if grpo_config.rl_config.performance_stats:
        logger.info("grpo performance statistics is on")
        PERF_STATS = True


def convert_index_json_total(load_checkpoint, converted_dir, convert_map_dict_lst, is_qkv_concat):
    """convert mapping file if exists"""
    index_path = os.path.join(load_checkpoint, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        logger.warning(f"The given path contains no 'model.safetensors.index.json' file.")
        return
    with open(index_path, "r") as f:
        data = json.load(f)
    weight_map = data.get("weight_map")
    total_weight_map = {}
    for convert_map_dict_func in convert_map_dict_lst:
        new_weight_map = copy.deepcopy(weight_map)
        converted_weight_map = convert_map_dict_func(new_weight_map, qkv_concat=is_qkv_concat)
        total_weight_map.update(converted_weight_map)
    flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(os.path.join(converted_dir, "param_name_map.json"), flags_, 0o750), "w") as f:
        json.dump(total_weight_map, f, indent=2)
        logger.info(f"Converted file param_name_map.json")


def get_unique_filename(save_file_path):
    """get the unique name of the save file."""
    root, ext = os.path.splitext(save_file_path)
    counter = 1
    while True:
        if not os.path.exists(save_file_path):
            return save_file_path
        save_file_path = f"{root}-{counter}{ext}"
        counter += 1


def _is_empty(arr):
    """Whether the array is empty or not."""
    if arr is None:
        return True
    if isinstance(arr, list) and not arr:
        return True
    if isinstance(arr, np.ndarray) and arr.size == 0:
        return True
    return len(arr) == 0


def save_prompt_completions_data(save_file_dir, global_step, **kwargs):
    """Save prompt completions data."""
    if get_rank() == 0:
        if not os.path.exists(save_file_dir):
            os.makedirs(save_file_dir)
        file_path = os.path.join(save_file_dir, f"prompt_completions_step_{global_step}.json")
        file_path = get_unique_filename(file_path)
        data_dict = []
        first_key = next(iter(kwargs))
        data_length = len(kwargs[first_key])

        for i in range(data_length):
            new_row = {}
            for key, values in kwargs.items():
                if _is_empty(values):
                    new_row[key] = None
                else:
                    value = values[i]
                    if value is None:
                        new_row[key] = None
                    elif hasattr(value, "item") and callable(getattr(value, "item", None)):
                        new_row[key] = value.item()
                    else:
                        new_row[key] = value
            data_dict.append(new_row)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved data to {file_path}")
        except OSError as e:
            logger.warning(f"Save File Error: {e}")


def get_dp_rank(data_parallel):
    """Get Data Parallel Rank ID"""
    rank_id = get_rank()
    if reshard_optimizer.OPT_COMMUNICATION_GROUPS:
        for group in reshard_optimizer.OPT_COMMUNICATION_GROUPS["dp"]:
            if rank_id in group:
                dp_rank_id = group.index(rank_id)
                break

        if dp_rank_id is None:
            raise ValueError(
                f"Rank {rank_id} not found in any DP group: " f"{reshard_optimizer.OPT_COMMUNICATION_GROUPS}"
            )
    else:
        dp_rank_id = rank_id // (get_group_size() // data_parallel)
    return dp_rank_id


def add_metrics_to_tensorboard(tensor_writer, metrics, global_step):
    """add training metrics to tensorboard."""
    for key, value in metrics.items():
        tensor_writer.add_scalar(key, value, global_step=global_step)


def get_checkpoint_name(
    ckpt_path, prefix: str = "network", epoch_num: int = None, step_num: int = None, formats: str = "ckpt"
):
    """
    Get checkpoint file name of model.
    The layout of the ckpt_path will be like:
    ckpt_path/
    ├── rank_0
    │   ├── network_rank_0-0_1.ckpt
    │   └── network_rank_0-0_2.ckpt
    """
    rank = get_rank()
    # ensure ckpt path exist
    ckpt_path = os.path.normpath(os.path.abspath(ckpt_path))
    ckpt_local_path = os.path.join(ckpt_path, f"rank_{rank}")
    os.makedirs(ckpt_local_path, exist_ok=True)
    ckpt_file = os.path.join(ckpt_local_path, f"{prefix}_rank_{rank}-{epoch_num}_{step_num}.{formats}")
    return ckpt_file


def ensure_total_ckpt_is_less_than_limit(ckpt_path: str, limit: int = 5, formats: str = "ckpt"):
    """
    make sure the provided path contain less than limited number of checkpoint file
    Args:
        ckpt_path (str): Checkpoint file path.
        limit (int): limited number of checkpoint file. Default: 5
        formats (str): checkpoint format. Default: 'ckpt'
    """
    ckpt_list = [checkpoint for checkpoint in os.listdir(ckpt_path) if checkpoint.endswith(f".{formats}")]
    # ckpt_list: [oldest, ..., newest]
    ckpt_list = sorted(ckpt_list, key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)))
    ckpt_num = len(ckpt_list)
    if ckpt_num >= limit:
        for rm_ckpt_name in ckpt_list[: (ckpt_num - limit)]:
            logger.warning(f"Current checkpoint file exceed keep_checkpoint_max, " f"removing {rm_ckpt_name}")
            rm_ckpt_path = os.path.join(ckpt_path, rm_ckpt_name)
            os.remove(rm_ckpt_path)


def load_param_to_net(net, param_dict):
    """load param to net."""
    param_not_load, ckpt_not_load = ms.load_param_into_net(net, param_dict)
    if param_not_load:
        logger.warning(f"When loading ckpt into the net, param_not_load:{param_not_load}")
    if ckpt_not_load:
        logger.warning(f"When loading ckpt into the net, ckpt_not_load:{ckpt_not_load}")


def record_last_ckpt_to_json(epoch: int, step: int, ckpt_file: str, meta_json: str):
    """record last ckpt info to json"""
    dangerous_paths = ["/etc", "/usr", "/bin", "/sbin", "/boot", "/proc", "/sys"]

    for dangerous_path in dangerous_paths:
        if dangerous_path in meta_json:
            raise PermissionError(
                f"Security violation: Cannot use system directory '{dangerous_path}' "
                f"as dump path. Attempted path: '{meta_json}'"
            )

    meta_data = {"last_epoch": epoch, "last_step": step, "last_ckpt_file": ckpt_file}
    flags = os.O_WRONLY | os.O_CREAT
    mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(meta_json, flags, mode), "w", encoding="utf-8") as fp:
        json.dump(meta_data, fp)


def load_safetensors(safetensors_path, load_ckpt_format, network, grpo_model, prefix, strategy_path):
    """
    load_safetensors
    Args:
        safetensors_path (str): safetensors path.
        load_ckpt_format (int): hf_safetensors, ms_safetensors.
        network (cell): network
        grpo_model : grpo model
        prefix: model prefix.
        strategy_path: strategy file path.
    """
    try:
        load_checkpoint_files = glob(os.path.join(safetensors_path, f"*.safetensors"))
        load_checkpoint_files.sort()
        if load_ckpt_format == "hf_safetensors":
            name_map = network.obtain_name_map(load_checkpoint_files)
            name_map = {f"{prefix}{key}": value for key, value in name_map.items()}
        else:
            name_map = dict()
            param_name_map_dict = dict()
            for checkpoint_file in load_checkpoint_files:
                with safe_open(checkpoint_file, framework="np") as f:
                    for k in f.keys():
                        name_map.update({f"{prefix}{k}": k})
                        param_name_map_dict[f"{prefix}{k}"] = os.path.basename(checkpoint_file)
            if get_rank() == 0:
                with open(os.path.join(safetensors_path, "param_name_map.json"), "w", encoding="utf-8") as f:
                    json.dump(param_name_map_dict, f, ensure_ascii=False, indent=4)
            else:
                # wait for rank 0 to finish
                time.sleep(10)
            ms.mint.distributed.barrier()
    except Exception as e:
        raise TypeError(f"Please complete abstract function obtain_name_map. Details: {e}") from e

    ms.load_distributed_checkpoint(
        network=grpo_model,
        predict_strategy=strategy_path,
        unified_safetensors_dir=safetensors_path,
        format="safetensors",
        name_map=name_map,
    )


def enable_pynative_async(func):
    """enable pynative async and disable when finish executing func"""

    def wrapper(*args, **kwargs):
        _pynative_executor.set_async_for_graph(True)
        result = func(*args, **kwargs)
        _pynative_executor.set_async_for_graph(False)
        return result

    return wrapper


class TimeConsumingCollector:
    """
    Collect code line performance context mgr.

    Args:
        func_name (str): To be collected function or stage name.
        timer (Dict[str, list]): Timer to collect multi-steps statistics.
    """

    def __init__(self, func_name, timer: Dict[str, list] = None):
        self.func_name = func_name
        self.start_timestamp = None
        self.timer = timer
        self.duration = 0
        self.mstx_id = None

    def __enter__(self):
        self.start_timestamp = time.perf_counter()
        self.mstx_id = ms.profiler.mstx.range_start(self.func_name)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        global PERF_STATS
        self.duration = time.perf_counter() - self.start_timestamp
        if self.mstx_id is not None:
            ms.profiler.mstx.range_end(self.mstx_id)
        if PERF_STATS:
            logger.info(f"Performance Statistics: {self.func_name}: duration={self.duration}")


def get_grpo_profiler(profiler_config, role: str = None):
    """
    Get GRPO profiler instance.

    Args:
        profiler_config: Profiler configuration object
        role (str, optional): Role name for profiling. Defaults to None.

    Returns:
        Profiler instance or None if profiling is disabled or not for this rank
    """
    args = profiler_config
    if not args or not args.profile:
        return None

    profiler_this_rank = False
    if args.profile_ranks == "all":
        profiler_this_rank = True
    else:
        try:
            ranks = list(args.profile_ranks)
        except (TypeError, ValueError):
            ranks = [0]
        if ms.communication.get_rank() in ranks:
            profiler_this_rank = True
    if not profiler_this_rank:
        return None

    if args.profile_level == "levelNone":
        profiler_level = ms.profiler.ProfilerLevel.LevelNone
    elif args.profile_level == "level0":
        profiler_level = ms.profiler.ProfilerLevel.Level0
    elif args.profile_level == "level1":
        profiler_level = ms.profiler.ProfilerLevel.Level1
    elif args.profile_level == "level2":
        profiler_level = ms.profiler.ProfilerLevel.Level2
    else:
        raise ValueError(
            f"profiler_level only supports level0," f" 1, 2, and level_none, but gets {args.profile_level}"
        )

    base_path = args.profile_save_path
    if role:
        profile_save_path = os.path.join(base_path, role)
    else:
        profile_save_path = base_path

    # pylint: disable=protected-access
    experimental_config = ms.profiler._ExperimentalConfig(
        aic_metrics=ms.profiler.AicoreMetrics.PipeUtilization,
        profiler_level=profiler_level,
        data_simplification=True,
        mstx=args.mstx,
    )

    if args.stage == "all":
        skip_first = args.profile_step_start - 1
        active = args.profile_step_end - args.profile_step_start
    else:
        skip_first = 0
        active = 1

    activities = []
    if args.profile_with_npu:
        activities.append(ms.profiler.ProfilerActivity.NPU)
    if args.profile_with_cpu:
        activities.append(ms.profiler.ProfilerActivity.CPU)

    prof = ms.profiler.profile(
        start_profile=False,
        with_stack=args.profile_with_stack,
        profile_memory=args.profile_with_memory,
        activities=activities,
        schedule=ms.profiler.schedule(wait=0, warmup=0, active=active, repeat=1, skip_first=skip_first),
        on_trace_ready=ms.profiler.tensorboard_trace_handler(profile_save_path, analyse_flag=args.profile_analysis),
        experimental_config=experimental_config,
    )

    return prof


def profiler_start(profiler_config, role="profiler", profiler_iteration=None):
    """
    Start profiler based on configuration.

    根据用户配置（profiler_config），判断是否需要启动性能分析器（Profiler），如果需要则创建并启动它；否则返回 None。

    Args:
        profiler_config: Profiler configuration object
        role (str): Role name for profiling. Defaults to "profiler".
        profiler_iteration (int, optional): Current iteration number. Defaults to None.

    Returns:
        Started profiler instance or None if profiling is not needed
    """
    if not profiler_config:
        return None
    if profiler_iteration is not None and (
        profiler_iteration < profiler_config.profile_step_start
        or profiler_iteration >= profiler_config.profile_step_end
    ):
        return None
    if profiler_config.stage != "all":
        if isinstance(profiler_config.stage, str):
            if role != profiler_config.stage:
                return None
        else:
            try:
                stages = list(profiler_config.stage)
            except (TypeError, ValueError):
                stages = []
            if role not in stages:
                return None
    if profiler_config.stage == "all" and role != "grpo_all_stage":
        return None
    profiler = get_grpo_profiler(profiler_config, role)
    if not profiler:
        return None
    profiler.start()
    return profiler


def profiler_step(profiler):
    """Execute a profiler step if profiler exists."""
    if profiler:
        profiler.step()


def mstx_timer_decorator(func):
    """Decorator for MindSpore mstx timer profiling."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        mstx_id = ms.profiler.mstx.range_start(func.__qualname__)
        result = func(*args, **kwargs)
        ms.profiler.mstx.range_end(mstx_id)
        return result

    return wrapper


def set_infer_dp_size(grpo_config):
    global INFER_DP_SIZE
    INFER_DP_SIZE = grpo_config.generate_config.parallel_config.data_parallel


def get_infer_dp_size():
    global INFER_DP_SIZE
    return INFER_DP_SIZE


def set_pack_level(grpo_config):
    """set min and max pack level that are compatible with train model pp micro batch num"""
    pipeline_stage = grpo_config.actor_config.parallel_config.pipeline_stage
    if pipeline_stage <= 1:
        grpo_config.rl_config.max_pack_level = 0
        grpo_config.rl_config.min_pack_level = 0
        logger.warning(f"Adaptive packing is set to False as pipeline parallel in train model is not enabled.")
        return
    micro_batch_num = grpo_config.actor_config.parallel_config.micro_batch_num
    import math

    max_allowed_pack_level = math.floor(math.log2(micro_batch_num / pipeline_stage))
    if grpo_config.rl_config.max_pack_level > max_allowed_pack_level:
        grpo_config.rl_config.max_pack_level = max_allowed_pack_level
        logger.warning(f"Reset max_pack_level to max allowed pack level: {max_allowed_pack_level}")
    if grpo_config.rl_config.min_pack_level > max_allowed_pack_level:
        grpo_config.rl_config.min_pack_level = max_allowed_pack_level
        logger.warning(f"Reset min_pack_level to max allowed pack level: {max_allowed_pack_level}")
    return
