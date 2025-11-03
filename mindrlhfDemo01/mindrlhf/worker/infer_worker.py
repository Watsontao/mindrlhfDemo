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
"""Inference Worker."""

import os
import time
import numpy as np

import mindspore as ms
from mindspore import Tensor
from mindspore.communication import GlobalComm, get_rank, create_group
from mindspore import context
from mindspore import communication as D

from mindformers import LlamaConfig
from mindformers import MindFormerConfig
from mindformers.trainer.utils import load_distributed_checkpoint
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.parallel_core.inference.utils import generate_state_dict
from mindformers import logger
from research.deepseek3.deepseek3_config import DeepseekV3Config

from mindrlhf.utils import transfer_from_str_to_bool, TimeConsumingCollector
from mindrlhf.models.grpo_models import CausalLMHybrid, GRPOModelInfer
from mindrlhf.configs.grpo_configs import VllmMode
from mindrlhf.utils.utils import get_valid_length_each_example, get_dp_rank, load_safetensors, enable_pynative_async
from mindrlhf.worker.worker import Worker
from mindrlhf.utils.strategy_utils import save_strategy_file
import mindrlhf.utils.reshard_optimizer as reshard_optimizer
from mindrlhf.configs.grpo_configs import GRPOConfig


DTYPE_STR = {ms.float16: "float16", ms.bfloat16: "bfloat16", ms.float32: "float32", ms.int8: "int8", ms.uint8: "uint8"}


class InferWorker(Worker):
    """
    This class generates responses.
    """

    def __init__(self, grpo_config: GRPOConfig, args, **kwargs):
        """
        Init infer worker.

        Args:
            grpo_config (GRPOConfig): GRPO config instance.
            args: Args will be removed in next release version.
            **kwargs: Key word args.
        """
        super().__init__()
        logger.info("init InferWorker")
        self.args = args
        self.tokenizer = kwargs.get("tokenizer")
        self.load_ckpt_format = grpo_config.rl_config.load_ckpt_format
        sft_config_infer = MindFormerConfig(grpo_config.generate_config.model_config)
        sft_config_infer.model.model_config.seq_length = grpo_config.rl_config.seq_length
        sft_config_infer.use_parallel = grpo_config.rl_config.use_parallel
        sft_config_infer.parallel_config = MindFormerConfig(**grpo_config.generate_config.parallel_config.param_dict)
        logger.info(f"generate parallel_config:{sft_config_infer.parallel_config}")
        sft_config_infer.model.model_config.offset = grpo_config.generate_config.offset
        sft_config_infer.model.model_config.max_decode_length = grpo_config.generate_config.sampling_config.max_tokens
        sft_config_infer.model.model_config.min_decode_length = grpo_config.generate_config.sampling_config.min_tokens
        enable_compile_cache = transfer_from_str_to_bool(grpo_config.rl_config.enable_compile_cache)
        self.use_parallel = grpo_config.rl_config.use_parallel
        os.environ["RUN_MODE"] = sft_config_infer.run_mode

        # Reentrancy protection for distributed init.
        if not GlobalComm.INITED:
            logger.info(f"launch actor roll out sft_config_infer.use_parallel {sft_config_infer.use_parallel}")
            sft_config_infer.context = grpo_config.context.param_dict
            logger.info(f"sft_config_infer.context:{sft_config_infer.context}")
            deterministic = grpo_config.rl_config.deterministic.lower() == "on"
            logger.info(f"set deterministic: {deterministic}")
            if deterministic:
                ms.set_deterministic(True)
                os.environ["HCCL_DETERMINISTIC"] = "true"
                os.environ["TE_PARALLEL_COMPILER"] = "1"
                os.environ["CUSTOM_MATMUL_SHUFFLE"] = "off"
                os.environ["LCCL_DETERMINISTIC"] = "1"
            build_context(sft_config_infer)
        build_parallel_config(sft_config_infer)
        context.set_context(enable_compile_cache=enable_compile_cache, compile_cache_path="./generate_cache")

        # init sft infer model
        sft_config_infer.model.model_config.parallel_config = sft_config_infer.parallel_config
        logger.info(f"sft_config_infer:\n{sft_config_infer}")

        if args.model_name in ["qwen", "llama"]:
            sft_model_config_infer = LlamaConfig(**sft_config_infer.model.model_config)
            sft_model_config_infer.model_name = "llama"
        elif args.model_name == "deepseek":
            sft_config_infer.model.model_config.moe_config = sft_config_infer.moe_config
            sft_model_config_infer = DeepseekV3Config(**sft_config_infer.model.model_config)
            sft_model_config_infer.model_name = "deepseek_infer"
        else:
            raise ValueError(f"model_name should in ['qwen', 'llama','deepseek'], but get {args.model_name}")

        sft_model_config_infer.checkpoint_name_or_path = grpo_config.generate_config.load

        self.grpo_config = grpo_config
        self.sft_ckpt_path_infer = sft_model_config_infer.checkpoint_name_or_path
        # Must set this to None before building policy model.
        sft_model_config_infer.checkpoint_name_or_path = None
        self.sft_model_config_infer = sft_model_config_infer
        self.dp_rank_id = get_dp_rank(self.sft_model_config_infer.parallel_config.data_parallel)

        context.set_auto_parallel_context(parallel_mode="stand_alone", full_batch=False)
        sim_level = os.getenv("MS_SIMULATION_LEVEL")
        if sim_level:
            logger.warning(f"MS_SIMULATION_LEVEL is set to {sim_level}, will not use vllm")
            self.use_vllm = VllmMode.ORIGIN
        else:
            self.use_vllm = grpo_config.generate_config.use_vllm
        if self.use_vllm != VllmMode.ORIGIN:
            self.policy_model = self.__init_use_vllm()
            self.old_phase = self.policy_model.phase
            self.policy_model.dp = sft_model_config_infer.parallel_config.data_parallel
        else:
            # no vllm
            self.policy_model = CausalLMHybrid(sft_model_config_infer, self.grpo_config)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)
        self.grpo_model_infer = GRPOModelInfer(self.grpo_config, self.policy_model)
        self.grpo_model_infer.set_train(False)
        self.infer_pp_stage = sft_model_config_infer.parallel_config.pipeline_stage or 1
        if self.use_vllm == VllmMode.ORIGIN:
            self.grpo_model_infer.grpo_model.policy_model.model.add_flags_recursive(is_first_iteration=True)
            self.grpo_model_infer.grpo_model.policy_model.model.set_train(False)
        else:
            self.grpo_model_infer.grpo_model.policy_model.add_flags_recursive(is_first_iteration=True)
            self.grpo_model_infer.grpo_model.policy_model.set_train(False)
        self.on_device = True
        self.save_strategy_dir = grpo_config.rl_config.save_strategy_dir
        self.dp_group = dict()
        group_name, group_list = self._get_dp_group(self.get_infer_dp())
        self.dp_group[group_name] = group_list

    def _get_dp_group(self, dp):
        """
        get dp group for allgather
        """
        world_size = D.get_group_size()
        rank_id = D.get_rank()
        str_dp_rank_list = '-'.join([str(rank) for rank in range(rank_id%(world_size//dp), world_size, world_size//dp)])
        dp_rank_list = [rank for rank in range(rank_id%(world_size//dp), world_size, world_size//dp)]
        dp_group = f"dp_group-{str_dp_rank_list}"
        create_group(dp_group, dp_rank_list)
        return dp_group, dp_rank_list

    def _get_dp_group_name(self, dp):
        """
        get dp group name
        """
        world_size = D.get_group_size()
        rank_id = D.get_rank()
        str_dp_rank_list = '-'.join([str(rank) for rank in range(rank_id%(world_size//dp), world_size, world_size//dp)])
        dp_group = f"dp_group-{str_dp_rank_list}"
        return dp_group

    def refresh_policy_model_phase(self):
        """
        refresh policy model phase
        """
        if self.use_vllm != VllmMode.ORIGIN:
            self.policy_model.phase = self.old_phase

    @enable_pynative_async
    def __init_use_vllm(self):
        """
        init_vllm
        """
        # pylint: disable=W0611
        from mindrlhf.third_party.vllm import package_version, LLM
        from vllm import SamplingParams

        if self.args.resume_training:
            logger.warning("Enable resume_training, skip loading infer model weights")
            from vllm_mindspore.model_executor.models.mf_models.qwen2 import Qwen2ForCausalLM
            from mindrlhf.third_party.vllm.qwen2 import load_weights

            Qwen2ForCausalLM.load_weights = load_weights
            self.sft_ckpt_path_infer = self.grpo_config.rl_config.tokenizer_dir

        self.tokenizer.max_token_id = max(self.tokenizer.get_vocab().values())
        # 初始化vllm
        logger.info(
            f"init LLM, block_size: {self.grpo_config.generate_config.block_size}, "
            f"max_model_len = {self.grpo_config.generate_config.max_model_len}, "
            f"max_num_batched_tokens: {self.grpo_config.generate_config.max_num_batched_tokens}, "
            f"max_num_seqs: {self.grpo_config.generate_config.max_num_seqs}, "
            f"num_scheduler_steps: {self.grpo_config.generate_config.num_scheduler_steps}, "
            f"gpu_memory_utilization: {self.grpo_config.generate_config.gpu_memory_utilization}, "
            f"seed: {self.dp_rank_id}"
        )
        vllm_start_time = time.time()
        if package_version.startswith("0.8"):
            from mindrlhf.third_party.vllm.vllm_v_general import initialize_parallel_state

            initialize_parallel_state(
                tensor_model_parallel_size=self.sft_model_config_infer.parallel_config.model_parallel
            )
            vllm_start_time = time.time()
            if self.load_ckpt_format not in ["hf_safetensors", "ms_safetensors"]:
                raise ValueError(
                    f"For vllm {package_version}, "
                    f"the infer model load_ckpt_format must be hf_safetensors or ms_safetensors."
                )
            if self.grpo_config.generate_config.num_scheduler_steps > 1:
                logger.warning(f"For VLLM V1, num_scheduler_steps > 1 is not supported, set it to 1.")
                self.grpo_config.generate_config.num_scheduler_steps = 1
            self.inference_engine = LLM(
                model=self.sft_ckpt_path_infer,  # path to hf model
                tensor_parallel_size=self.sft_model_config_infer.parallel_config.model_parallel,
                distributed_executor_backend="external_launcher",
                dtype="bfloat16",
                block_size=self.grpo_config.generate_config.block_size,
                skip_tokenizer_init=False,
                max_model_len=self.grpo_config.generate_config.max_model_len,
                max_num_batched_tokens=self.grpo_config.generate_config.max_num_batched_tokens,
                max_num_seqs=self.grpo_config.generate_config.max_num_seqs,
                num_scheduler_steps=self.grpo_config.generate_config.num_scheduler_steps,
                gpu_memory_utilization=self.grpo_config.generate_config.gpu_memory_utilization,
                trust_remote_code=self.grpo_config.generate_config.trust_remote_code,
                enable_prefix_caching=False,  # FIXME: not support prefix caching now.
                seed=self.dp_rank_id,
            )
            logger.info(f"init LLM end, cost time: {time.time() - vllm_start_time}")
            model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
            policy_model = model_runner.model.network
        else:
            raise ValueError(f"Not support vllm version: {package_version}")
        logger.info(f"init LLM end, cost time: {time.time() - vllm_start_time}")
        max_decode_len = (
            self.grpo_config.generate_config.sampling_config.max_tokens - 2**self.grpo_config.rl_config.max_pack_level
        )

        logger.info(
            f"temperature: {self.grpo_config.generate_config.sampling_config.temperature}, "
            f"repetition_penalty: {self.grpo_config.generate_config.sampling_config.repetition_penalty}, "
            f"top_p: {self.grpo_config.generate_config.sampling_config.top_p}, "
            f"top_k: {self.grpo_config.generate_config.sampling_config.top_k}, "
            f"stop_token_ids: {self.grpo_config.generate_config.sampling_config.eos_token_id}, "
            f"max_tokens: {max_decode_len}, "
            f"detokenize: {self.grpo_config.generate_config.sampling_config.detokenize}"
        )
        vllm_start_time = time.time()
        self.sampling_params = SamplingParams(
            repetition_penalty=self.grpo_config.generate_config.sampling_config.repetition_penalty,
            temperature=self.grpo_config.generate_config.sampling_config.temperature,
            top_p=self.grpo_config.generate_config.sampling_config.top_p,
            top_k=self.grpo_config.generate_config.sampling_config.top_k,
            stop_token_ids=self.grpo_config.generate_config.sampling_config.eos_token_id,
            max_tokens=max_decode_len,
            min_tokens=self.grpo_config.generate_config.sampling_config.min_tokens,
            detokenize=self.grpo_config.generate_config.sampling_config.detokenize,
        )
        logger.info(f"init SamplingParams end, cost time: {time.time() - vllm_start_time}")
        return policy_model

    def model(self):
        """Return model"""
        return self.grpo_model_infer

    def get_updated_grpo_config(self):
        """Get updated grpo config."""
        return self.grpo_config

    def get_infer_dp(self):
        """Get infer dp."""
        return self.sft_model_config_infer.parallel_config.data_parallel

    def _allgather_data(self, batch_input, data_parallel_size, padding_length=128):
        """
        allgather_data
        """
        lengths = []
        padded_arrays = []
        local_bs = len(batch_input)
        for array in batch_input:
            lengths.append(len(array))
            padded_array = [0] * padding_length
            padded_array[: len(array)] = array
            padded_arrays.append(padded_array)
        padded_arrays = Tensor(padded_arrays).astype(ms.int32)
        lengths = Tensor(lengths).astype(ms.int32)
        dp_group = self._get_dp_group_name(self.get_infer_dp())
        dp_rank_list = self.dp_group[dp_group]
        all_padded_arrays, _ = D.comm_func.all_gather_into_tensor(padded_arrays, group=dp_group)
        all_lengths, _ = D.comm_func.all_gather_into_tensor(lengths, group=dp_group)
        all_lengths = all_lengths.asnumpy()
        all_padded_arrays = all_padded_arrays.asnumpy()
        world_size = len(dp_rank_list)
        all_other_group_size = world_size // data_parallel_size
        output_batch = []
        if reshard_optimizer.OPT_COMMUNICATION_GROUPS:
            collect_range = [_ * local_bs for _ in reshard_optimizer.OPT_COMMUNICATION_GROUPS["dp"][0]]
        else:
            collect_range = range(0, world_size * local_bs, all_other_group_size * local_bs)

        for i in collect_range:
            for k in range(local_bs):
                global_idx = i + k
                output_batch.append(list(all_padded_arrays[global_idx][: all_lengths[global_idx]]))

        return output_batch

    def post_process_infer_outputs(self, results):
        """post_process_infer_outputs"""
        with TimeConsumingCollector("post process infer outputs"):
            right_padding_responses, responses_mask, left_padding_prompts, prompts_mask = results
            max_tokens = self.grpo_config.generate_config.sampling_config.max_tokens
            # allgather data
            right_padding_responses_batch = self._allgather_data(
                right_padding_responses,
                self.sft_model_config_infer.parallel_config.data_parallel,
                padding_length=max_tokens,
            )
            responses_mask_batch = self._allgather_data(
                responses_mask, self.sft_model_config_infer.parallel_config.data_parallel, padding_length=max_tokens
            )
            left_padding_prompts_batch = self._allgather_data(
                left_padding_prompts,
                self.sft_model_config_infer.parallel_config.data_parallel,
                padding_length=self.grpo_config.rl_config.seq_length - max_tokens,
            )
            prompts_mask_batch = self._allgather_data(
                prompts_mask,
                self.sft_model_config_infer.parallel_config.data_parallel,
                padding_length=self.grpo_config.rl_config.seq_length - max_tokens,
            )
            right_padding_responses = np.array(right_padding_responses_batch).astype(np.int32)
            responses_mask = np.array(responses_mask_batch).astype(np.int32)
            left_padding_prompts = np.array(left_padding_prompts_batch).astype(np.int32)
            prompts_mask = np.array(prompts_mask_batch).astype(np.int32)
            return right_padding_responses, responses_mask, left_padding_prompts, prompts_mask

    # For SPMD, developer could call 'post_process_infer_outputs' to process data.
    # For MPMD, data should be collected to driver process and dispatch to other ray actors.
    def generate(self, input_ids_numpy, max_tokens=0):
        """Policy model generates responses for a batch of prompts."""
        ms.set_context(jit_config={"infer_boost": "on"})
        context.set_auto_parallel_context(
            pipeline_stages=self.infer_pp_stage, parallel_mode="stand_alone", full_batch=False
        )

        logger.info(f"input_ids shape {input_ids_numpy.shape}")

        valid_length_each_example, max_valid_length = get_valid_length_each_example(
            input_ids_numpy, self.grpo_model_infer.grpo_model.pad_token_id
        )  # get valid length and max length in a batch

        generate_begin_time = time.time()
        if max_tokens == 0:
            max_tokens = self.grpo_config.generate_config.sampling_config.max_tokens
        min_tokens = self.grpo_config.generate_config.sampling_config.min_tokens
        logger.info(f"max_tokens {max_tokens}")
        logger.info(f"min_tokens {min_tokens}")
        if self.use_vllm == VllmMode.DEBUG:
            # use vllm model
            logger.info("infer without vllm, use vllm model")
            outputs = self.grpo_model_infer.grpo_model.policy_model.model.generate(
                input_ids_numpy[:, :max_valid_length],
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                do_sample=True,
                seed=self.dp_rank_id,
            )
            logger.info("infer without vllm end, use vllm model")
        elif self.use_vllm == VllmMode.ORIGIN:
            logger.info("infer without vllm, not use vllm model")
            outputs = self.grpo_model_infer.grpo_model.policy_model.model.generate(
                input_ids_numpy[:, :max_valid_length],
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                do_sample=True,
                seed=self.dp_rank_id,
            )
            logger.info("infer without vllm end, not use vllm model")
        else:
            logger.info("start vllm")
            prompt = input_ids_numpy[:, :max_valid_length]
            vllm_prompt = prompt.tolist()
            outputs = self._vllm_generate(vllm_prompt, valid_length_each_example)
            logger.info("end vllm")

        logger.info(f"Generating elapsed time: {time.time() - generate_begin_time}")

        input_ids_list = input_ids_numpy.tolist()
        num_sample = len(input_ids_list)
        max_prompt_length = self.grpo_config.generate_config.max_prompt_length
        max_tokens = self.grpo_config.generate_config.sampling_config.max_tokens
        pad_token_id = self.grpo_config.generate_config.sampling_config.pad_token_id
        # init prompt, max length is max_prompt_length
        left_padding_prompts = np.ones((num_sample, max_prompt_length)) * pad_token_id
        # init response, max length is max_decode_length
        right_padding_responses = np.ones((num_sample, max_tokens)) * pad_token_id
        # prompt length without padding token
        prompt_len = (np.array(input_ids_list) != pad_token_id).astype(int).sum(1)
        responses_mask = np.zeros((num_sample, max_tokens))

        for i in range(num_sample):
            # only response
            if self.use_vllm == VllmMode.ORIGIN:
                response = outputs[i][prompt_len[i] : prompt_len[i] + max_tokens]
            else:
                # vllm output without prompt
                response = outputs[i].outputs[0].token_ids
            right_padding_responses[i, : len(response)] = response
            responses_mask[i, : len(response)] = 1

            # right padding
            left_padding_prompts[i, max_prompt_length - prompt_len[i] :] = input_ids_list[i][: prompt_len[i]]

        responses_mask = responses_mask.astype(np.int32)
        prompts_mask = (left_padding_prompts != pad_token_id).astype(np.int32)

        ms.set_context(jit_config={"infer_boost": "off"})
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)
        return (
            right_padding_responses.astype(np.int32),
            responses_mask,
            left_padding_prompts.astype(np.int32),
            prompts_mask,
        )

    def generate_strategy(self, reshard_optimizer_):
        """generate strategy"""
        ms.set_context(jit_config={"infer_boost": "on"})
        context.set_auto_parallel_context(
            pipeline_stages=self.infer_pp_stage, parallel_mode="stand_alone", full_batch=False
        )
        stage_name = "infer"
        ms.mint.distributed.barrier()
        if self.use_vllm == VllmMode.ORIGIN:
            static_dict = generate_state_dict(self.grpo_model_infer.grpo_model.policy_model.model)
        else:
            static_dict = generate_state_dict(self.grpo_model_infer.grpo_model.policy_model)
        save_strategy_file(
            static_dict,
            reshard_optimizer_,
            f"{self.save_strategy_dir}/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt",
        )
        stage_name = "other"
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file": f"{self.save_strategy_dir}/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"
            }
        )
        ms.set_context(jit_config={"infer_boost": "off"})
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)

    def offload(self):
        """offload infer"""
        if not self.on_device:
            return
        logger.info(f"before offload infer {ms.hal.memory_stats()}")
        with TimeConsumingCollector("offload infer"):
            skip_kv_cache = False
            if self.use_vllm == VllmMode.VLLM:
                self.inference_engine.free_cache_engine()
                skip_kv_cache = True
            for param in self.grpo_model_infer.grpo_model.get_parameters(expand=True):
                if skip_kv_cache and "paged_attention_mgr" in param.name:
                    continue
                # pylint: disable=W0212
                param._offload()
        logger.info(f"after offload infer {ms.hal.memory_stats()}")
        self.on_device = False

    def load(self, skip_kv_cache=False):
        """load infer"""
        if self.on_device:
            return
        logger.info(f"before load infer {ms.hal.memory_stats()}")
        with TimeConsumingCollector("load infer"):
            if self.use_vllm == VllmMode.VLLM and not skip_kv_cache:
                self._init_cache_engine()
                skip_kv_cache = True
            for param in self.grpo_model_infer.grpo_model.get_parameters(expand=True):
                if skip_kv_cache and "paged_attention_mgr" in param.name:
                    continue
                # pylint: disable=W0212
                param._load()
        logger.info(f"after load infer {ms.hal.memory_stats()}")
        self.on_device = True

    def init_kvcache(self):
        """init infer kvcache"""
        logger.info(f"before init infer kv cache {ms.hal.memory_stats()}")
        with TimeConsumingCollector("init infer kvcache"):
            if self.use_vllm == VllmMode.VLLM:
                self._init_cache_engine()
        logger.info(f"after init  infer kv cache {ms.hal.memory_stats()}")
        self.on_device = True

    def load_checkpoint(self):
        """load_checkpoint"""
        logger.info(f"sft_ckpt_path_infer:{self.sft_ckpt_path_infer}")
        if not self.sft_ckpt_path_infer:
            return
        if not os.path.exists(self.sft_ckpt_path_infer):
            raise ValueError(f"infer model checkpoint path: {self.sft_ckpt_path_infer} not exists")

        if self.sft_ckpt_path_infer and self.load_ckpt_format in ["ms_safetensors", "hf_safetensors"]:
            self.on_device = True
            strategy_path = os.path.join(self.save_strategy_dir, "merge_strategy", "infer_policy_merged_strategy.ckpt")
            if self.use_vllm != VllmMode.ORIGIN:
                return
            network = self.grpo_model_infer.grpo_model.policy_model.model
            prefix = "grpo_model.policy_model.model."
            load_safetensors(
                self.sft_ckpt_path_infer,
                self.load_ckpt_format,
                network,
                self.grpo_model_infer.grpo_model.policy_model,
                prefix,
                strategy_path,
            )
            return
        load_ckpt_func = load_distributed_checkpoint if self.use_parallel else ms.load_checkpoint
        logger.info(f"use_parallel is {self.use_parallel} {load_ckpt_func}")
        if self.sft_ckpt_path_infer:
            self.on_device = True
            param_dict = load_ckpt_func(self.sft_ckpt_path_infer)
            if self.use_vllm == VllmMode.ORIGIN:
                new_param_dict = {"grpo_model.policy_model.model." + k: v for k, v in param_dict.items()}
            else:
                new_param_dict = {"grpo_model.policy_model." + k: v for k, v in param_dict.items()}

            logger.info(f"begin to load infer policy model from: {self.sft_ckpt_path_infer}")
            logger.info(new_param_dict.keys())
            for _, param in self.grpo_model_infer.grpo_model.policy_model.parameters_and_names():
                logger.info(f"infer model para names:   {param.name}")
            param_not_load, ckpt_not_load = ms.load_param_into_net(
                self.grpo_model_infer.grpo_model.policy_model, new_param_dict
            )
            logger.info(f"param not load: {param_not_load}")
            logger.info(f"ckpt not load: {ckpt_not_load}")

    def convert_map_dict(self, source_dict, **kwargs):
        """convert_map_dict"""
        if self.use_vllm != VllmMode.ORIGIN:
            network = self.grpo_model_infer.grpo_model.policy_model
            prefix = "grpo_model.policy_model."
        else:
            network = self.grpo_model_infer.grpo_model.policy_model.model
            prefix = "grpo_model.policy_model.model."
        weight_dict = network.convert_map_dict(source_dict, **kwargs)
        new_weight_dict = {f"{prefix}{key}": value for key, value in weight_dict.items()}
        return new_weight_dict

    @enable_pynative_async
    def _vllm_generate(self, vllm_prompt, valid_length_each_example):
        token_ids = self.inference_engine.pre_process_inputs(vllm_prompt, valid_length_each_example)
        outputs = self.inference_engine.generate(
            prompts=None, sampling_params=self.sampling_params, prompt_token_ids=token_ids, use_tqdm=False
        )
        return outputs

    @enable_pynative_async
    def _init_cache_engine(self):
        self.inference_engine.init_cache_engine()
