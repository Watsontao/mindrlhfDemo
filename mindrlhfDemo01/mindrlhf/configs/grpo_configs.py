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
# limitations under the License
# ============================================================================
"""GRPO model"""
from dataclasses import dataclass
from enum import Enum
import yaml


class VllmMode(Enum):
    ORIGIN = 0
    VLLM = 1
    DEBUG = 2  # DEBUG mode: init model with vllm, but generate with mindformers


@dataclass
class ParallelConfig:
    """parallel config"""

    data_parallel: int = 1
    model_parallel: int = 4
    pipeline_stage: int = 2
    expert_parallel: int = 1
    use_seq_parallel: bool = True
    micro_batch_num: int = 4
    vocab_emb_dp: bool = False
    context_parallel: int = 1

    param_dict = {
        "data_parallel": data_parallel,
        "model_parallel": model_parallel,
        "pipeline_stage": pipeline_stage,
        "expert_parallel": expert_parallel,
        "use_seq_parallel": use_seq_parallel,
        "micro_batch_num": micro_batch_num,
        "vocab_emb_dp": vocab_emb_dp,
        "context_parallel": context_parallel,
    }


@dataclass
class RecomputeConfig:
    """recompute config"""

    recompute: bool = False
    select_recompute: bool = False
    select_comm_recompute: bool = False
    parallel_optimizer_comm_recompute: bool = False
    mp_comm_recompute: bool = True
    recompute_slice_activation: bool = False

    param_dict = {
        "recompute": recompute,
        "select_recompute": select_recompute,
        "select_comm_recompute": select_comm_recompute,
        "parallel_optimizer_comm_recompute": parallel_optimizer_comm_recompute,
        "mp_comm_recompute": mp_comm_recompute,
        "recompute_slice_activation": recompute_slice_activation,
    }


@dataclass
class Optimizer:
    """optimizer"""

    type: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    eps: float = 1.0e-8
    weight_decay: float = 0.01
    opt_offload: bool = False


@dataclass
class LRSchedule:
    """lr schedule"""

    lr: float = 5.0e-7
    min_lr: float = 1.0e-10
    warmup_step: int = 10
    decay_steps: int = 200000
    lr_decay_style: str = "cosine"


@dataclass
class ActorConfig:
    """actor model config"""

    load: str = "/path/"
    save: str = "/tmp/"
    model_config: str = "/path/finetune.yaml"
    parallel_config: ParallelConfig = ParallelConfig
    recompute_config: RecomputeConfig = RecomputeConfig
    offset: int = 0
    enable_parallel_optimizer: bool = True
    enable_alltoall: bool = False
    use_eod_attn_mask_compression: bool = True
    loss_scale_value: int = 1
    optimizer: Optimizer = Optimizer
    lr_schedule: LRSchedule = LRSchedule


@dataclass
class RefConfig:
    """reference model config"""

    model_config: str = "/path/finetune.yaml"
    load: str = "/path/"
    ref_model_batch_size: int = 2
    # Whether to synchronize the reference model with the policy model every `ref_model_sync_steps`
    sync_ref_model: bool = False
    ref_model_sync_steps: int = 50
    parallel_config: ParallelConfig = ParallelConfig
    recompute_config: RecomputeConfig = RecomputeConfig
    offset: int = 0
    use_eod_attn_mask_compression: bool = True


@dataclass
class SamplingConfig:
    """sampling config"""

    temperature: float = 0.8
    repetition_penalty: float = 1.05
    top_p: float = 0.8
    top_k: int = 20
    bos_token_id: int = 153643
    eos_token_id: list = None
    pad_token_id: int = 153643
    detokenize: bool = False
    logprobs: float = 1
    max_tokens: int = 512
    min_tokens: int = 2
    min_p: float = 0.01


@dataclass
class GenerateConfig:
    """generate model config"""

    model_config: str = "/path/predict.yaml"
    load: str = "/path/"
    infer_model_batch_size: int = 2
    parallel_config: ParallelConfig = ParallelConfig
    offset: int = 0
    use_eod_attn_mask_compression: bool = True
    # generate config
    use_vllm: int = 1  # 0--MindFormers; 1--VLLM; 2--DEBUG mode: init model with vllm, but generate with mindformers
    hf_config_path: str = "config.json"  # vllm config path
    block_size: int = 16
    max_model_len: int = 25536
    max_num_batched_tokens: int = 25536
    max_num_seqs: int = 1024
    max_prompt_length: int = 2048
    num_scheduler_steps: int = 32
    gpu_memory_utilization: float = 0.8
    trust_remote_code: bool = True
    sampling_config: SamplingConfig = SamplingConfig


@dataclass
class RewardConfig:
    """grpo reward config"""

    verifier_function: list = None
    verifier_weight: list = None


@dataclass
class JitConfig:
    """jit config"""

    jit_level: str = "O0"


@dataclass
class AscendConfig:
    """Ascend config"""

    precision_mode: str = "must_keep_origin_dtype"


@dataclass
class Context:
    """context"""

    mode: int = 0  # 0--Graph Mode; 1--Pynative Mode
    device_target: str = "Ascend"
    max_call_depth: int = 10000
    max_device_memory: str = "55GB"
    save_graphs: bool = False
    save_graphs_path: str = "./graph"
    device_id: int = 0
    jit_config: JitConfig = JitConfig
    memory_optimize_level: str = "O0"
    ascend_config: AscendConfig = AscendConfig

    param_dict = {
        "mode": mode,
        "device_target": device_target,
        "max_call_depth": max_call_depth,
        "max_device_memory": max_device_memory,
        "save_graphs": save_graphs,
        "save_graphs_path": save_graphs_path,
        "device_id": device_id,
        "jit_config": {"jit_level": jit_config.jit_level},
        "memory_optimize_level": memory_optimize_level,
        "ascend_config": {"precision_mode": ascend_config.precision_mode},
    }


@dataclass
class MonitorConfig:
    host_monitor_interval: float = -1.0
    host_monitor_steps: list = None
    host_memory_protection: bool = False
    host_max_memory_threshold: float = 0.95


@dataclass
class RLConfig:
    """rl config"""

    model_name: str = "qwen2.5"
    deterministic: str = "OFF"
    align_type: str = "rlhf_stages"
    dataset_file: str = "/path/train.mindrecord"
    tokenizer_type: str = "qwen"
    tokenizer_dir: str = "/path/"
    epochs: int = 10
    batch_size: int = 1
    sink_size: int = 2
    seq_length: int = 4096
    use_parallel: bool = True
    load_ckpt_format: str = "hf_safetensors"
    parallel_mode: str = "semi_auto_parallel"
    enable_compile_cache: bool = False
    save_strategy_dir: str = "./strategy/"
    save_data_file: str = "/tmp/"

    packing: bool = True
    pack_num: int = 1

    save_prompt_completions_data: bool = True
    save_prompt_completions_interval: int = 1
    save_prompt_completions_dir: str = "/tmp/"

    # 0: do not optimize mem during resharding
    # 1: offload all src and dst param during resharding
    reshard_mem_opt_level: int = 0
    # 0: run reshard in PYNATIVE mode
    # 1: run reshard in GRAPH mode
    # 2: run reshard in HYBRID mode
    reshard_mode: int = 0
    save_ckpt_interval: int = 1
    save_max_ckpt_num: int = 5
    save_ckpt_format: str = "safetensors"  # format support safetensors/ckpt
    enable_reshard_optimizer: bool = False

    tensorboard: bool = False
    tensorboard_dir: str = "/tmp/"
    tensorboard_queue_size: int = 10
    enable_full_monitor: bool = False
    calculate_entropy: bool = False

    save_checkpoint_dir: str = "/tmp/"
    performance_stats: bool = False
    micro_batch_interleaved: int = 1

    beta: float = 0.01  # KL coefficient
    num_generations: int = 8
    num_rollouts: int = 4
    chunk_size: int = 1
    gen_experience_kwargs: bool = False
    # clip higher
    num_iterations: int = 1
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    enable_oldpolicy: bool = True
    seed: int = None

    dynamic_pack_level: bool = False
    max_pack_level: int = 0
    min_pack_level: int = 0

    # support "token-mean", "seq-mean-token-mean"
    loss_agg_mode: str = "token-mean"


@dataclass
class ProfilerConfig:
    """profiler config"""

    profile: bool = False
    mstx: bool = False
    stage: str = "all"
    profile_save_path: str = "./profiler_data"
    profile_level: str = "level1"
    profile_with_memory: bool = False
    profile_with_cpu: bool = True
    profile_with_npu: bool = True
    profile_with_stack: bool = False
    profile_step_start: int = 0
    profile_step_end: int = 1
    profile_analysis: bool = False
    profile_ranks: str = "all"


@dataclass
class MsprobeConfig:
    """msprobe config"""

    msprobe: bool = False
    dump_path: str = "./msprobe_dump"
    key_data_dump: bool = False
    configurations_dump: bool = False


@dataclass
class GRPOConfig:
    """
    GRPO config class which defines the model size
    """

    actor_config = ActorConfig
    ref_config = RefConfig
    reward_config = RewardConfig
    generate_config = GenerateConfig
    rl_config = RLConfig
    monitor_config = MonitorConfig
    profiler_config = ProfilerConfig
    context = Context
    msprobe_config = MsprobeConfig

    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        for key in data:
            setattr(self, key, getattr(self, key)(**data[key]))
        self._update_sub_config(data)

    def _set_config(self, data, name, sub_name, config_class):
        if name in data and sub_name in data[name]:
            module = getattr(self, name)
            setattr(module, sub_name, config_class(**data[name][sub_name]))

    def _update_sub_config(self, data):
        """update sub config"""
        self._set_config(data, "actor_config", "parallel_config", ParallelConfig)
        self._set_config(data, "ref_config", "parallel_config", ParallelConfig)
        self._set_config(data, "generate_config", "parallel_config", ParallelConfig)
        self._set_config(data, "actor_config", "recompute_config", RecomputeConfig)
        self._set_config(data, "ref_config", "recompute_config", RecomputeConfig)
        self._set_config(data, "context", "jit_config", JitConfig)
        self._set_config(data, "context", "ascend_config", AscendConfig)
        self._set_config(data, "actor_config", "optimizer", Optimizer)
        self._set_config(data, "actor_config", "lr_schedule", LRSchedule)
        self._set_config(data, "generate_config", "sampling_config", SamplingConfig)
        if "actor_config" in data and "parallel_config" in data["actor_config"]:
            self.actor_config.parallel_config.param_dict = data["actor_config"]["parallel_config"]
        if "actor_config" in data and "recompute_config" in data["actor_config"]:
            self.actor_config.recompute_config.param_dict = data["actor_config"]["recompute_config"]
        if "ref_config" in data and "parallel_config" in data["ref_config"]:
            self.ref_config.parallel_config.param_dict = data["ref_config"]["parallel_config"]
        if "generate_config" in data and "parallel_config" in data["generate_config"]:
            self.generate_config.parallel_config.param_dict = data["generate_config"]["parallel_config"]
        if "context" in data:
            self.context.param_dict = data["context"]
