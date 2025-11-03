# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reference Worker."""
import os
import numpy as np

import mindspore as ms
from mindspore import context, ops
from mindspore.communication.management import get_rank, create_group

from mindformers import MindFormerConfig
from mindformers import logger
from mindformers import LlamaConfig
from mindformers.trainer.utils import load_distributed_checkpoint
from mindformers.tools.resume_ckpt import get_resume_checkpoint_by_meta
from research.deepseek3.deepseek3_config import DeepseekV3Config

from mindrlhf.models.grpo_models import CausalLMHybrid
from mindrlhf.worker.worker import Worker
from mindrlhf.utils import TimeConsumingCollector, _get_pipeline_group
from mindrlhf.configs.grpo_configs import GRPOConfig
from mindrlhf.utils.utils import (
    load_param_to_net,
    record_last_ckpt_to_json,
    get_checkpoint_name,
    ensure_total_ckpt_is_less_than_limit,
    load_safetensors,
)


class RefWorker(Worker):
    """
    This class generates responses.
    """

    def __init__(self, grpo_config: GRPOConfig, args):
        super().__init__()
        logger.info("init RefWorker")
        self.args = args
        self.use_parallel = grpo_config.rl_config.use_parallel
        self.load_ckpt_format = grpo_config.rl_config.load_ckpt_format
        ref_config = MindFormerConfig(grpo_config.ref_config.model_config)
        ref_config.model.model_config.seq_length = grpo_config.rl_config.seq_length
        ref_config.use_parallel = self.use_parallel
        ref_config.parallel_config = MindFormerConfig(**grpo_config.ref_config.parallel_config.param_dict)
        logger.info(f"ref parallel_config:{ref_config.parallel_config}")
        logger.info(f"grpo_config.ref_config.recompute_config:{grpo_config.ref_config.recompute_config.param_dict}")
        ref_config.recompute_config = grpo_config.ref_config.recompute_config.param_dict
        ref_config.model.model_config.offset = grpo_config.ref_config.offset
        ref_config.model.model_config.parallel_config = ref_config.parallel_config
        ref_config.model.model_config.parallel_config.recompute = ref_config.recompute_config
        self.ref_pp_stage = ref_config.parallel_config.pipeline_stage
        ref_config.model.model_config.use_past = False
        if args.model_name in ["qwen", "llama"]:
            ref_config.model.model_config.use_eod_attn_mask_compression = (
                grpo_config.ref_config.use_eod_attn_mask_compression
            )
            ref_model_config = LlamaConfig(**ref_config.model.model_config)
            ref_model_config.model_name = "llama"
        elif args.model_name == "deepseek":
            ref_config.model.model_config.moe_config = ref_config.moe_config
            ref_model_config = DeepseekV3Config(**ref_config.model.model_config)
            ref_model_config.model_name = "deepseek_training"
        else:
            raise ValueError(f"model_name should in ['qwen', 'llama','deepseek'], but get {args.model_name}")

        # set pipeline stage
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)
        context.set_auto_parallel_context(pipeline_stages=self.ref_pp_stage, enable_parallel_optimizer=False)
        logger.info(f"ref_model_config:{ref_model_config}")
        # set allreduce
        rank_list, pipeline_group_name = _get_pipeline_group()
        pipeline_group_name = "ref_pipeline" + pipeline_group_name
        logger.info(f"start create pipeline {pipeline_group_name}")
        create_group(pipeline_group_name, rank_list)
        logger.info(f"end create pipeline {pipeline_group_name}")
        self.all_reduce = ops.AllReduce(group=pipeline_group_name)
        ref_model_config.checkpoint_name_or_path = grpo_config.ref_config.load
        self.ref_model_config = ref_model_config
        self.ref_ckpt_path = self.get_ref_ckpt_path
        ref_model_config.checkpoint_name_or_path = None
        self.ref_model = CausalLMHybrid(ref_model_config, grpo_config)
        self.ref_model.model.set_train(False)
        for name, param in self.ref_model.parameters_and_names():
            param.name = name
        self.on_device = True
        self.grpo_config = grpo_config
        if grpo_config.actor_config.save and get_rank() == 0:
            ref_save_dir = os.path.join(self.grpo_config.actor_config.save, "ref")
            if not os.path.exists(ref_save_dir):
                os.makedirs(ref_save_dir)
        self.ref_pp_stage = ref_config.parallel_config.pipeline_stage or 1
        self.ref_dp = ref_config.parallel_config.data_parallel
        self.save_strategy_dir = grpo_config.rl_config.save_strategy_dir

    def model(self):
        """Return ref model."""
        return self.ref_model

    @property
    def get_ref_ckpt_path(self):
        """get ref ckpt path"""
        return self.ref_model_config.checkpoint_name_or_path

    def get_ref_dp(self):
        """Get reference model data parallel size"""
        return self.ref_dp

    def compile(self):
        """compile and save strategy"""
        self.ref_model.model.set_train(False)
        context.set_auto_parallel_context(pipeline_stages=self.ref_pp_stage, enable_parallel_optimizer=False)
        total_ref_model_batch_size = self.grpo_config.ref_config.ref_model_batch_size * self.ref_dp
        fake_data = ms.Tensor(shape=(total_ref_model_batch_size, self.grpo_config.rl_config.seq_length), dtype=ms.int32)
        actual_seq_data = ms.Tensor(
            shape=(total_ref_model_batch_size, self.grpo_config.rl_config.pack_num), dtype=ms.int32
        )
        stage_name = "infer"
        strategy_path = self.grpo_config.rl_config.save_strategy_dir
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file": f"{strategy_path}/{stage_name}_ref_strategy/strategy_{get_rank()}.ckpt",
                "only_trainable_params": False,
            }
        )
        # To avoid mindspore compiler's unpacking bug and prevent duplicate compilation,
        # use positional arguments instead of keyword arguments
        self.ref_model.compile(
            fake_data, fake_data, samples=fake_data, actual_sequence_length=actual_seq_data, is_ref=False
        )
        stage_name = "other"
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file": f"{strategy_path}/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt",
                "only_trainable_params": True,
            }
        )

    def compute_ref_log_prob(
        self, prompt_completion_ids_tensor, attention_mask_tensor, samples, actual_sequence_length
    ):
        """
        compute ref log prob
        """
        np.set_printoptions(threshold=1024)
        context.set_auto_parallel_context(pipeline_stages=self.ref_pp_stage, enable_parallel_optimizer=False)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)
        ref_per_token_logps = self.ref_model(
            prompt_completion_ids_tensor,
            attention_mask_tensor,
            samples=samples,
            actual_sequence_length=actual_sequence_length,
        )
        if self.ref_pp_stage > 1:
            ref_per_token_logps = self.all_reduce(ref_per_token_logps)
        return ref_per_token_logps

    def offload(self):
        """offload ref model"""
        if self.on_device is False:
            return
        logger.info(f"before offload ref model {ms.hal.memory_stats()}")
        with TimeConsumingCollector("offload ref model"):
            for param in self.ref_model.get_parameters(expand=True):
                # pylint: disable=W0212
                param._offload()
        logger.info(f"after offload ref model {ms.hal.memory_stats()}")
        self.on_device = False

    def load(self):
        """load ref model"""
        if self.on_device:
            return
        logger.info(f"before load ref model {ms.hal.memory_stats()}")
        with TimeConsumingCollector("load ref model"):
            for param in self.ref_model.get_parameters(expand=True):
                # pylint: disable=W0212
                param._load()
        logger.info(f"after load ref model {ms.hal.memory_stats()}")
        self.on_device = True

    def save_checkpoints(self, epochs=0, steps=0, start_epoch=0, start_step=0, formats="ckpt"):
        """save checkpoint"""
        if epochs == start_epoch and steps == start_step:
            return
        if self.grpo_config.actor_config.save:
            if self.grpo_config.rl_config.save_ckpt_format == "safetensors":
                formats = "safetensors"
            logger.info("Save checkpoints in {}".format(self.grpo_config.actor_config.save))
            ref_save_dir = os.path.join(self.grpo_config.actor_config.save, "ref")
            rank_path = os.path.join(ref_save_dir, f"rank_{get_rank()}")
            ckpt_file = get_checkpoint_name(
                ref_save_dir, prefix="ref", epoch_num=epochs, step_num=steps, formats=formats
            )
            self.load()
            ms.save_checkpoint(self.ref_model, ckpt_file, integrated_save=False, format=formats)
            self.offload()
            ensure_total_ckpt_is_less_than_limit(
                ckpt_path=rank_path, limit=self.grpo_config.rl_config.save_max_ckpt_num, formats=formats
            )
            record_last_ckpt_to_json(
                epoch=epochs,
                step=steps,
                ckpt_file=os.path.basename(ckpt_file),
                meta_json=os.path.join(rank_path, "meta.json"),
            )

    def reload_ckpt(self, formats="ckpt"):
        """reload checkpoint for resume training"""
        if self.ref_ckpt_path:
            if self.grpo_config.rl_config.save_ckpt_format == "safetensors":
                formats = "safetensors"
            src_ckpt_file = os.path.join(self.ref_ckpt_path, f"rank_{get_rank()}")
            if not os.path.isdir(src_ckpt_file):
                raise ValueError(f"There is no *.{formats} in {src_ckpt_file}, load failed.")
            logger.info(f"Loading latest ref checkpoint: {src_ckpt_file}, this may take a while.")
            meta_path = os.path.join(src_ckpt_file, "meta.json")
            if not os.path.exists(meta_path):
                raise ValueError(f"Could not find meta.json in directory {src_ckpt_file} {meta_path}")
            os.environ["LOCAL_DEFAULT_PATH"] = self.ref_ckpt_path
            resume_ckpt = get_resume_checkpoint_by_meta(self.ref_ckpt_path, formats)
            ckpt_path = os.path.join(src_ckpt_file, resume_ckpt)
            param_dict = ms.load_checkpoint(ckpt_path, format=formats)
            load_param_to_net(self.ref_model, param_dict)
            self.on_device = True

    def load_checkpoint(self):
        """load_checkpoint"""
        logger.info(f"ref_ckpt_path:{self.ref_ckpt_path}")
        if not self.ref_ckpt_path:
            return

        if not os.path.exists(self.ref_ckpt_path):
            raise ValueError(f"ref model checkpoint path: {self.ref_ckpt_path} not exists")

        if self.ref_ckpt_path and self.load_ckpt_format in ["ms_safetensors", "hf_safetensors"]:
            self.on_device = True
            strategy_path = os.path.join(self.save_strategy_dir, "merge_strategy", "infer_ref_merged_strategy.ckpt")
            prefix = "model."
            load_safetensors(
                self.ref_ckpt_path, self.load_ckpt_format, self.ref_model.model, self.ref_model, prefix, strategy_path
            )
            return
        load_ckpt_func = load_distributed_checkpoint if self.use_parallel else ms.load_checkpoint
        logger.info(f"use_parallel is {self.use_parallel} {load_ckpt_func}")
        if self.ref_ckpt_path:
            self.on_device = True
            param_dict = load_ckpt_func(self.ref_ckpt_path)
            new_param_dict = {"model." + k: v for k, v in param_dict.items()}
            # ===========================================================================
            logger.info(f"begin to load ref model from: {self.ref_ckpt_path}")
            for _, param in self.ref_model.parameters_and_names():
                logger.info(f"ref model para names:   {param.name}")
            param_not_load, ckpt_not_load = ms.load_param_into_net(self.ref_model, new_param_dict)
            logger.info(f"param not load: {param_not_load}")
            logger.info(f"ckpt not load: {ckpt_not_load}")

    def convert_map_dict(self, source_dict, **kwargs):
        """convert_map_dict"""
        network = self.ref_model.model
        prefix = "model."
        weight_dict = network.convert_map_dict(source_dict, **kwargs)
        new_weight_dict = {f"{prefix}{key}": value for key, value in weight_dict.items()}
        return new_weight_dict


class DummyRefWorker(Worker):
    # pylint: disable=W0231,W0613
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        def method(*args, **kwargs):
            pass

        return method


def get_ref_worker(*args, **kwargs):
    grpo_config = kwargs.get("grpo_config", None)
    if grpo_config and grpo_config.rl_config.enable_ref:
        logger.info(f"enable_ref is True, RefWorker is enalbled")
        return RefWorker(*args, **kwargs)
    logger.info(f"enable_ref is False, RefWorker is not enalbled")
    return DummyRefWorker(*args, **kwargs)
