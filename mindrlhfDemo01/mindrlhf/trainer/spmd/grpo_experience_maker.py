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
""" GRPO ExperienceMaker """

# python
import time
from dataclasses import asdict
from copy import deepcopy
import numpy as np

# mindspore
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore.dataset import MindDataset
from mindspore import Tensor, mint
from mindspore.communication import get_rank
from mindspore.mindrecord import FileWriter

# mindformers
from mindformers import logger

# mindrlhf
from mindrlhf.utils import (
    TimeConsumingCollector,
    save_prompt_completions_data,
    MetricData,
    get_dp_rank,
    add_metrics_to_tensorboard,
    profiler_start,
    profiler_step,
)

from mindrlhf.reward.reward_fn import accuracy_reward, format_reward, qwen_accuracy_reward
from mindrlhf.reward.kk_reward_fn import kk_reward
from mindrlhf.reward.math_dapo_fn import math_dapo_reward
from mindrlhf.reward.deepscaler_fn import base_deepscaler_reward, cot_deepscaler_reward
from mindrlhf.worker.worker import GRPOData
from mindrlhf.configs.grpo_configs import VllmMode
from mindrlhf.utils.utils import MsProbe


class GRPOExperienceMaker:
    """GRPO ExperienceMaker"""

    def __init__(
        self,
        train_model,
        infer_model,
        ref_model,
        old_policy_model,
        grpo_config,
        tokenizer,
        tensor_writer,
        dataset_ignore_step,
    ):
        super(GRPOExperienceMaker, self).__init__()
        self.train = train_model
        self.infer = infer_model
        self.ref = ref_model
        self.old_policy = old_policy_model
        self.grpo_config = grpo_config
        self.tokenizer = tokenizer
        self.tensor_writer = tensor_writer
        self.infer_dp = self.infer.get_infer_dp()
        self.ref_dp = self.ref.get_ref_dp()

        self.i_step = dataset_ignore_step
        self.make_exp_step = 0
        self.total_processed_tokens = 0
        self.step_num = 0
        self.step_total_tokens = 0
        self.profiler_iteration = 0

        self.num_train_prompts = (
            self.infer_dp * self.grpo_config.rl_config.chunk_size * self.grpo_config.rl_config.num_rollouts
        )

        if self.grpo_config.rl_config.dynamic_pack_level:
            self.max_pack_level = self.grpo_config.rl_config.max_pack_level
            self.min_pack_level = self.grpo_config.rl_config.min_pack_level
            self.max_pack_num = 2 ** (self.max_pack_level)
        else:
            self.max_pack_level = 0
            self.min_pack_level = 0
            self.max_pack_num = 1

        self.step_pack_level = self.max_pack_level
        self.step_pack_num = self.max_pack_num

        self._init_grpo_experience_dataset()
        self._init_reward_fn()

    def _init_grpo_experience_dataset(self):
        """
        Build dataset for generating.
        """
        self.mind_dataset_dir = self.grpo_config.rl_config.dataset_file
        logger.info(f"GRPOExperienceMaker: _init_grpo_experience_dataset, dataset dir {self.mind_dataset_dir}")
        if self.mind_dataset_dir is not None:
            columns_to_project = ["prompt_ids", "pretrain_ids"]
            ms.dataset.config.set_seed(2023)
            bs = self.grpo_config.rl_config.chunk_size * self.infer_dp
            self.prompt_dataset = MindDataset(self.mind_dataset_dir).project(columns=columns_to_project)
            self.prompt_dataloader = self.prompt_dataset.take()
            self.prompt_dataloader = self.prompt_dataloader.batch(batch_size=bs, drop_remainder=True)
            self.prompt_iterator = self.prompt_dataloader.create_tuple_iterator()
            self.step_num = self.prompt_dataloader.get_dataset_size() // self.grpo_config.rl_config.num_rollouts
            logger.info(f"total step is: {self.step_num}.")
            if self.i_step > 0:
                for _ in range(self.i_step):
                    _ = self._get_batch(self.grpo_config.rl_config.num_rollouts)
                logger.info(f"The beginning {self.i_step} batch data will be skipped.")
        else:
            logger.info("In main task, there is not dataset for making experience")

    def _get_batch(self, num_rollouts):
        """get batch"""
        full_batch = None
        for _ in range(num_rollouts):
            try:
                batch = next(self.prompt_iterator)
            except StopIteration:
                ms.dataset.config.set_seed(2023)
                self.prompt_iterator = self.prompt_dataloader.create_tuple_iterator()
                batch = next(self.prompt_iterator)
            if full_batch is None:
                full_batch = batch
            else:
                full_batch = (mint.cat((full_batch[0], batch[0])), mint.cat((full_batch[1], batch[1])))
        return full_batch

    def _init_reward_fn(self):
        """init reward function"""
        logger.info("GRPOExperienceMaker: _init_reward_fn")
        if self.grpo_config.reward_config.verifier_function:
            verifier_function_list = self.grpo_config.reward_config.verifier_function
        else:
            verifier_function_list = ["accuracy_reward", "format_reward"]
        if self.grpo_config.reward_config.verifier_weight:
            verifier_weight = self.grpo_config.reward_config.verifier_weight
        else:
            verifier_weight = [1.0, 1.0]
        logger.info(f"verifier_function_list:{verifier_function_list}")
        logger.info(f"verifier_weight:{verifier_weight}")

        verifier_function = []
        for reward_func_str in verifier_function_list:
            if "qwen" in self.grpo_config.rl_config.model_name and reward_func_str == "accuracy_reward":
                reward_func_str = "qwen_accuracy_reward"
            if reward_func_str == "accuracy_reward":
                verifier_function.append(accuracy_reward)
            elif reward_func_str == "format_reward":
                verifier_function.append(format_reward)
            elif reward_func_str == "qwen_accuracy_reward":
                verifier_function.append(qwen_accuracy_reward)
            elif reward_func_str == "kk_reward":
                verifier_function.append(kk_reward)
            elif reward_func_str == "math_dapo_reward":
                verifier_function.append(math_dapo_reward)
            elif reward_func_str == "base_deepscaler_reward":
                verifier_function.append(base_deepscaler_reward)
            elif reward_func_str == "cot_deepscaler_reward":
                verifier_function.append(cot_deepscaler_reward)
            else:
                raise ValueError(f"Unsupported reward function {reward_func_str}")
        self.verifier_function = verifier_function

        # Reward weights
        if len(verifier_weight) != len(verifier_function):
            raise ValueError(
                f"Number of reward weights ({len(verifier_weight)}) must match number of reward "
                f"functions ({len(verifier_function)})"
            )
        self.verifier_weight = np.array(verifier_weight, dtype=np.float32)

    @staticmethod
    def _split_for_data_parallel(batch_inputs, data_parallel_size):
        """
        split batch_inputs for data parallel
        """
        split_size = batch_inputs.shape[0] // data_parallel_size
        dp_rank_id = get_dp_rank(data_parallel_size)

        start = dp_rank_id * split_size
        stop = (dp_rank_id + 1) * split_size
        batch_inputs_for_this_rank = batch_inputs[start:stop]
        return batch_inputs_for_this_rank

    @staticmethod
    def _remove_right_padding(token_ids, padding_token=0):
        """remove_right_padding"""
        begin_time = time.time()
        counts = np.sum(token_ids != padding_token, axis=1)
        trimmed_sequences = [token_ids[i, :cnt] for i, cnt in enumerate(counts)]
        end_time = time.time()
        logger.info(f"remove right padding time: {end_time - begin_time}")
        return trimmed_sequences

    @staticmethod
    def _remove_left_padding(token_ids, padding_token=0):
        """remove_left_padding"""
        begin_time = time.time()
        counts = np.sum(token_ids != padding_token, axis=1)
        trimmed_sequences = [token_ids[i, -cnt:] for i, cnt in enumerate(counts)]
        end_time = time.time()
        logger.info(f"remove left padding time: {end_time - begin_time}")
        return trimmed_sequences

    @staticmethod
    def _construct_inputs_packing(packed_samples, batch_size=None, idx=None):
        """construct inputs for packing"""
        tmp_ids = []
        tmp_actual_seq_len = []
        tmp_responses_mask = []
        if batch_size:
            for i in range(batch_size):
                tmp_ids.append(packed_samples[i + idx * batch_size]["prompt_completion_ids"])
                tmp_actual_seq_len.append(packed_samples[i + idx * batch_size]["actual_sequence_length"])
                tmp_responses_mask.append(packed_samples[i + idx * batch_size]["responses_mask"])

        tmp_ids = np.array(tmp_ids)
        tmp_actual_seq_len = np.array(tmp_actual_seq_len)
        tmp_responses_mask = np.array(tmp_responses_mask)
        return tmp_ids, tmp_actual_seq_len, tmp_responses_mask

    @staticmethod
    def group_data(data_dict_list, pack_num):
        """group data for good packing strategy"""

        def get_total_length(data):
            if isinstance(data, list):
                return sum(elem["response_end_index"] - elem["prompt_start_idx"] for elem in data)
            return data["response_end_index"] - data["prompt_start_idx"]

        current_list = deepcopy(data_dict_list)
        cur_pack = 1
        while cur_pack < pack_num:
            sorted_list = sorted(current_list, key=get_total_length, reverse=True)
            current_list = []
            while sorted_list:
                top = sorted_list.pop(0)
                bottom = sorted_list.pop(-1)
                if isinstance(top, list):
                    current_list.append(top + bottom)
                else:
                    current_list.append([top, bottom])
            cur_pack *= 2
        res = []
        for e in current_list:
            res += e
        return res

    def create_pack_group(self, data_dict_list, pack_num):
        """create pack group"""
        if pack_num > 1:
            data_dict_list = self.group_data(data_dict_list, pack_num)
        sample_num = len(data_dict_list)
        pack_group, each_group = [], []
        current_group_length = 0
        for i in range(sample_num):
            sample_length = data_dict_list[i]["response_end_index"] - data_dict_list[i]["prompt_start_idx"] + 2
            needed_length = current_group_length + sample_length + (pack_num - len(each_group) - 1)
            if len(each_group) >= pack_num or needed_length > self.grpo_config.rl_config.seq_length:
                pack_group.append(each_group)
                each_group = []
                current_group_length = 0
            each_group.append(data_dict_list[i])
            current_group_length += sample_length
        if each_group:
            pack_group.append(each_group)
        return pack_group

    @staticmethod
    def pad_sequence_to_length(sequence, target_length, pad_value):
        """Pad sequence to target length with specified pad value."""
        current_length = len(sequence)
        if current_length < target_length:
            return np.pad(sequence, (0, target_length - current_length), mode="constant", constant_values=pad_value)
        return sequence[:target_length]

    def pack_grouped_data(self, pack_list, pack_num=1):
        """pack grouped data"""
        real_sample_num = len(pack_list)
        dummy_sample_num = pack_num - real_sample_num
        pad_to_length = self.grpo_config.rl_config.seq_length - dummy_sample_num
        pad_token_id = self.tokenizer.eos_token_id

        prompt_completion_ids = []
        actual_sequence_length = []
        responses_mask = []
        sample_index = []
        sample_valid_length = []
        advantages = []
        occupied_length = 0
        for i, data_dict in enumerate(pack_list):
            sample_prompt_completion_ids = data_dict["prompt_completion_ids"]
            sample_response_mask = data_dict["response_mask"]
            sample_advantage = data_dict["advantage"]
            prompt_start_idx = data_dict["prompt_start_idx"]
            response_end_index = data_dict["response_end_index"]
            sample_length = response_end_index - prompt_start_idx + 2

            sample_prompt_completion_ids = sample_prompt_completion_ids[prompt_start_idx : response_end_index + 1]
            sample_prompt_completion_ids = np.pad(
                sample_prompt_completion_ids, (0, 1), mode="constant", constant_values=pad_token_id
            )

            sample_response_mask = sample_response_mask[prompt_start_idx : response_end_index + 1]
            sample_response_mask = np.pad(sample_response_mask, (0, 1), mode="constant", constant_values=0)

            sample_actual_sequence_length = occupied_length + sample_length
            this_sample_index = np.array([i] * sample_length)
            sample_advantage = np.array([sample_advantage] * sample_length)

            if i == real_sample_num - 1:
                pad_length = pad_to_length - occupied_length
                sample_prompt_completion_ids = self.pad_sequence_to_length(
                    sample_prompt_completion_ids, pad_length, pad_token_id
                )
                sample_response_mask = self.pad_sequence_to_length(sample_response_mask, pad_length, 0)
                sample_advantage = self.pad_sequence_to_length(sample_advantage, pad_length, 0)
                this_sample_index = self.pad_sequence_to_length(this_sample_index, pad_length, i)
                sample_actual_sequence_length = pad_to_length

            prompt_completion_ids.append(sample_prompt_completion_ids)
            responses_mask.append(sample_response_mask)
            advantages.append(sample_advantage)
            actual_sequence_length.append(sample_actual_sequence_length)
            sample_index.append(this_sample_index)
            sample_valid_length.append(np.sum(sample_response_mask))

            occupied_length += sample_length

        for i in range(dummy_sample_num):
            prompt_completion_ids.append(np.array([pad_token_id]))
            responses_mask.append(np.array([0]))
            advantages.append(np.array([0]))
            actual_sequence_length.append(actual_sequence_length[-1] + 1)
            sample_index.append(np.array([real_sample_num + i]))
            sample_valid_length.append(1)

        result = {
            "prompt_completion_ids": np.concatenate(prompt_completion_ids, axis=0),
            "responses_mask": np.concatenate(responses_mask, axis=0),
            "advantages": np.concatenate(advantages, axis=0),
            "actual_sequence_length": np.array(actual_sequence_length),
            "sample_index": np.concatenate(sample_index, axis=0),
            "sample_valid_length": np.array(sample_valid_length),
        }

        return result

    def _generate_old_logps(self, packed_samples):
        """generate old log probs"""
        all_old_per_token_logps = np.zeros(
            (len(packed_samples), self.grpo_config.rl_config.seq_length), dtype=np.float32
        )
        if self.grpo_config.rl_config.calculate_entropy:
            all_entropy = np.zeros((len(packed_samples), self.grpo_config.rl_config.seq_length), dtype=np.float32)
            all_responses_mask = np.zeros(
                (len(packed_samples), self.grpo_config.rl_config.seq_length), dtype=np.float32
            )
        self.old_policy.load()
        logger.info("old_policy load")

        batch_size = self.grpo_config.rl_config.batch_size * self.old_policy.get_old_policy_dp()
        logger.info(f"old_policy_bs batch_size: {batch_size}")
        step_num = len(packed_samples) // batch_size
        logger.info(f"old policy model total steps: {step_num}")
        with TimeConsumingCollector(f"old_policy model all steps {step_num}"):
            for idx in range(step_num):
                prompt_completion_ids, actual_sequence_length, responses_mask = self._construct_inputs_packing(
                    packed_samples, batch_size=batch_size, idx=idx
                )

                prompt_completion_ids = np.pad(
                    prompt_completion_ids,
                    ((0, 0), (0, 1)),
                    "constant",
                    constant_values=self.grpo_config.generate_config.sampling_config.pad_token_id,
                )
                responses_mask = np.pad(responses_mask, ((0, 0), (0, 1)), "constant", constant_values=0)

                samples_tensor = Tensor(prompt_completion_ids[:, 1:], dtype=ms.int32)
                input_prompt_ids = Tensor(prompt_completion_ids[:, :-1], dtype=ms.int32)
                actual_sequence_length = Tensor(actual_sequence_length, dtype=ms.int32)
                responses_mask = Tensor(responses_mask[:, 1:], dtype=ms.int32)

                # Step 2: run old policy model.
                logger.info("old policy model step {} start".format(idx))
                with TimeConsumingCollector(f"old policy model step {idx}"):
                    generation_entropy, old_per_token_logps = self.old_policy.compute_old_log_prob(
                        input_prompt_ids, samples=samples_tensor, actual_sequence_length=actual_sequence_length
                    )
                    old_per_token_logps = old_per_token_logps.asnumpy().astype(np.float32)
                    generation_entropy = generation_entropy.asnumpy().astype(np.float32)
                logger.info("old policy model step {} end".format(idx))

                start_index = idx * batch_size
                end_index = (idx + 1) * batch_size
                all_old_per_token_logps[start_index:end_index, :] = old_per_token_logps
                if self.grpo_config.rl_config.calculate_entropy:
                    all_entropy[start_index:end_index, :] = generation_entropy
                    all_responses_mask[start_index:end_index, :] = responses_mask
        if self.grpo_config.rl_config.calculate_entropy:
            generation_entropy_mean = np.sum(all_entropy * all_responses_mask) / (np.sum(all_responses_mask) + 1e-8)
        else:
            generation_entropy_mean = None
        self.old_policy.offload()
        logger.info("old_policy offload")
        return generation_entropy_mean, all_old_per_token_logps

    def pack_grpo_data(self, prompt_completion_ids, prompts_mask, responses_mask, advantages, pack_num=1):
        """pack grpo data"""
        data_dict_list = []
        bs = prompt_completion_ids.shape[0]
        advantages = advantages.reshape(-1)
        logger.info(f"#1 advantages shape: {advantages.shape}")
        for i in range(bs):
            sample_prompt_mask = prompts_mask[i]
            sample_response_mask = responses_mask[i]
            indices = np.nonzero(sample_prompt_mask)[0]
            if indices.size:
                prompt_start_idx = indices[0]
            else:
                logger.warning(f"prompts_mask is all zero for index {i}!")
                continue
            indices = np.nonzero(sample_response_mask)[0]
            if indices.size:
                response_end_index = indices[-1]
            else:
                logger.warning(f"responses_mask is all zero for index {i}!")
                continue
            data_dict = {
                "prompt_completion_ids": prompt_completion_ids[i],
                "prompt_mask": prompts_mask[i],
                "response_mask": responses_mask[i],
                "advantage": advantages[i],
                "prompt_start_idx": prompt_start_idx,
                "response_end_index": response_end_index,
            }
            data_dict_list.append(data_dict)
        pack_group = self.create_pack_group(data_dict_list, pack_num)
        result = []
        for _, pack_list in enumerate(pack_group):
            packed = self.pack_grouped_data(pack_list, pack_num)
            result.append(packed)
        return result

    def _print_data_str(self, data, name):
        decoded_str = self.tokenizer.decode(data)
        logger.info(f"{name} str is {decoded_str}")

    def _save_grpoelement(self, save_path):
        """save grpo element"""
        if save_path:
            logger.info(f"start save grpo {save_path}")
            schema = {
                "prompt_completion_ids": {"type": "int32", "shape": [-1]},
                "responses_mask": {"type": "int32", "shape": [-1]},
                "ref_per_token_logps": {"type": "float32", "shape": [-1]},
                "advantages": {"type": "float32", "shape": [-1]},
                "actual_sequence_length": {"type": "int32", "shape": [-1]},
                "sample_index": {"type": "int32", "shape": [-1]},
                "sample_valid_length": {"type": "int32", "shape": [-1]},
                "old_per_token_logps": {"type": "float32", "shape": [-1]},
            }

            writer = FileWriter(file_name=save_path, shard_num=1, overwrite=True)
            writer.add_schema(schema)
            count = 0
            for ele in self.train.store:
                count += 1
                logger.info(f"add data element {asdict(ele)}")
                writer.write_raw_data([asdict(ele)])
            writer.commit()
            logger.info(f"end save grpo {save_path}")

    def generate_sequence(self, micro_bs, micro_num, prompt_tensors_full):
        """generate_sequence"""
        input_ids_numpy = self._split_for_data_parallel(prompt_tensors_full, self.infer_dp)
        self.infer.load()
        logger.info("generation start")
        with TimeConsumingCollector("infer generate"):
            max_tokens = self.grpo_config.generate_config.sampling_config.max_tokens
            if self.infer.use_vllm == VllmMode.ORIGIN:
                results = []
                for idx in range(micro_num):
                    result = self.infer.generate(
                        input_ids_numpy[idx * micro_bs : (idx + 1) * micro_bs, :], max_tokens=max_tokens
                    )
                    num_result = len(result)
                    for res_idx in range(num_result):
                        if len(results) == num_result:
                            results[res_idx] = np.concatenate((results[res_idx], result[res_idx]))
                        else:
                            results.append(result[res_idx])
            else:
                results = self.infer.generate(input_ids_numpy, max_tokens)
        logger.info("generation end")

        self.infer.offload()

        logger.info("model_infer offload")
        logger.info(f"generate sequence results is {results} type {type(results)}")
        return self.infer.post_process_infer_outputs(results)

    def compute_rewards(self, left_padding_prompts, right_padding_responses, repeat_solution_ids, sample_size):
        """compute_rewards"""
        logger.info("calculate reward start")
        pad_token_id = self.grpo_config.generate_config.sampling_config.pad_token_id
        solution_ids = self._remove_right_padding(repeat_solution_ids, padding_token=pad_token_id)
        if self.grpo_config.rl_config.tokenizer_type == "llama":
            solution = self.tokenizer.batch_decode(solution_ids, skip_special_tokens=True)
        else:
            solution = self.tokenizer.decode(solution_ids, skip_special_tokens=True)
        reward_kwargs = {"solution": solution}
        logger.info(f"solution: {solution}")

        metrics = {}
        all_elements_completion_len = []
        logger.info(f"left_padding_prompts is {type(left_padding_prompts)}")
        no_padding_prompts = self._remove_left_padding(left_padding_prompts, padding_token=pad_token_id)
        no_padding_responses = self._remove_right_padding(right_padding_responses, padding_token=pad_token_id)

        prompts_length_list = np.array([len(item) for item in no_padding_prompts])
        responses_length_list = np.array([len(item) for item in no_padding_responses])
        mean_prompts_length = prompts_length_list.mean()
        mean_responses_length = responses_length_list.mean()
        self.step_total_tokens = (mean_prompts_length + mean_responses_length) * sample_size
        self.total_processed_tokens += self.step_total_tokens
        logger.info(
            f"token_count mean_prompt_len: {mean_prompts_length}, "
            f"max_prompt_len: {prompts_length_list.max()},"
            f" min_prompt_len: {prompts_length_list.min()}"
        )
        logger.info(
            f"token_count mean_response_len: {mean_responses_length}, "
            f"max_response_len: {responses_length_list.max()}, "
            f"min_response_len: {responses_length_list.min()}"
        )
        max_tokens = self.grpo_config.generate_config.sampling_config.max_tokens
        clip_count = np.count_nonzero(responses_length_list == max_tokens)
        response_clip_ratio = clip_count / len(responses_length_list)
        metrics[MetricData.RESPONSE_LENGTH_MEAN.value] = mean_responses_length
        metrics[MetricData.RESPONSE_LENGTH_MAX.value] = responses_length_list.max()
        metrics[MetricData.RESPONSE_LENGTH_MIN.value] = responses_length_list.min()
        metrics[MetricData.RESPONSE_LENGTH_CLIP_RATIO.value] = response_clip_ratio

        metrics[MetricData.PROMPT_LENGTH_MEAN.value] = mean_prompts_length
        metrics[MetricData.PROMPT_LENGTH_MAX.value] = prompts_length_list.max()
        metrics[MetricData.PROMPT_LENGTH_MIN.value] = prompts_length_list.min()

        if self.grpo_config.rl_config.tokenizer_type == "llama":
            prompts = self.tokenizer.batch_decode(no_padding_prompts, skip_special_tokens=True)
            completions = self.tokenizer.batch_decode(no_padding_responses, skip_special_tokens=True)
        else:
            prompts = self.tokenizer.decode(no_padding_prompts, skip_special_tokens=True)
            completions = self.tokenizer.decode(no_padding_responses, skip_special_tokens=True)

        logger.info(f"prompts: \n {prompts[:2]}")
        logger.info(f"completions: \n {completions[:2]}")

        MsProbe.save_string_list("responses", completions)
        MsProbe.save_string_list("prompts", prompts)

        all_elements_completion_len.extend([len(com) for com in completions])
        completions_length = np.array([len(com) for com in completions])
        mean_len = completions_length.mean()
        logger.info(f"mean completions.length: \n {mean_len}")

        rewards_per_func = np.zeros((sample_size, len(self.verifier_function)), dtype=np.float32)
        answer_parsed_lst = []
        for i, reward_func in enumerate(self.verifier_function):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            if reward_func in [accuracy_reward, qwen_accuracy_reward, base_deepscaler_reward, cot_deepscaler_reward]:
                output_reward_func, answer_parsed_lst = reward_func(
                    prompts=prompts, completions=completions, **reward_kwargs
                )
            else:
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = np.array(output_reward_func, dtype=np.float32)
        rewards = (rewards_per_func * self.verifier_weight[np.newaxis, :]).sum(axis=1)
        logger.info(f"precision rewards are {rewards}")
        logger.info(f"precision parse answer are {answer_parsed_lst}")

        all_mean_len = np.mean(all_elements_completion_len)
        logger.info(f"all_elements_completion_len mean: {all_mean_len}")
        if self.tensor_writer:
            self.tensor_writer.add_scalar("mean-completion-length", all_mean_len, global_step=self.make_exp_step)

        all_rewards = np.array(rewards, dtype=np.float32)
        logger.info(f"loaded_all_rewards: {all_rewards}")
        metrics[MetricData.REWARD_MEAN.value] = np.mean(all_rewards)
        metrics[MetricData.REWARD_MAX.value] = np.max(all_rewards)
        metrics[MetricData.REWARD_MIN.value] = np.min(all_rewards)
        if (
            self.grpo_config.rl_config.save_prompt_completions_data
            and self.make_exp_step % self.grpo_config.rl_config.save_prompt_completions_interval == 0
        ):
            save_kwargs = {
                MetricData.QUESTION.value: prompts,
                MetricData.ANSWER.value: completions,
                MetricData.PARSED_ANSWER.value: answer_parsed_lst,
                MetricData.SOLUTION.value: solution,
                MetricData.REWARD_PER_QUESTION.value: rewards,
                MetricData.COMPLETION_LENGTH_PER_QUESTION.value: list(responses_length_list),
            }
            save_prompt_completions_data(
                self.grpo_config.rl_config.save_prompt_completions_dir, self.make_exp_step, **save_kwargs
            )
        logger.info("calculate reward end")
        return all_rewards, metrics

    # regroup the index of all data
    # assume the input shape is [num_generations*num_rollout*num_questions, -1]
    # [1,2,3,4,1,2,3,4]--->[1,1,2,2,3,3,4,4]
    @staticmethod
    def _reconstruct_index(x, num_generations):
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=1)
        seq = x.shape[-1]
        x = x.reshape((num_generations, -1, seq)).transpose((1, 0, 2))
        return x.reshape((-1, seq))

    def compute_advantages(self, all_rewards, metrics, num_generations, eps=1e-6):
        """compute_advantages"""
        all_mean_grouped_rewards = []
        total_size = all_rewards.shape[0]
        advantages = np.zeros((total_size,))
        tmp_all_rewards = all_rewards.copy()
        samples_per_step = total_size // num_generations
        for i in range(samples_per_step):
            listnum = list(range(i, total_size, samples_per_step))
            temp_rewards = tmp_all_rewards[listnum]
            mean_grouped_rewards = temp_rewards.mean()
            if temp_rewards.shape[0] == 1:
                std_grouped_rewards = temp_rewards.std()
            else:
                std_grouped_rewards = temp_rewards.std(ddof=1)
            adv_tem = (temp_rewards - mean_grouped_rewards) / (std_grouped_rewards + eps)
            logger.info(f"mean_grouped_rewards: \n {mean_grouped_rewards}")
            all_mean_grouped_rewards.append(mean_grouped_rewards)
            advantages[i::samples_per_step] = adv_tem.reshape((-1,))

        logger.info(f"advantages: {advantages}")
        logger.info(f"all_mean_grouped_rewards: {all_mean_grouped_rewards}")
        metrics[MetricData.ADVANTAGE_MEAN.value] = np.mean(advantages)
        metrics[MetricData.ADVANTAGE_MAX.value] = np.max(advantages)
        metrics[MetricData.ADVANTAGE_MIN.value] = np.min(advantages)
        logger.info(f"Metrics of total step {self.make_exp_step}: {metrics}")
        logger.info(f"all_mean_grouped_rewards: {all_mean_grouped_rewards}")
        avg_scores = np.mean(np.array(all_mean_grouped_rewards))
        logger.info(f"Avg scores: {avg_scores}")
        if self.tensor_writer:
            self.tensor_writer.add_scalar("average-scores", avg_scores, global_step=self.make_exp_step)
        return advantages

    def construct_packed_samples(self, generate_result, advantages, num_generations):
        """construct_packed_samples"""
        right_padding_responses, responses_mask_gather, left_padding_prompts, prompts_mask_gather = generate_result
        all_prompts_mask = np.concatenate(
            (prompts_mask_gather, np.zeros_like(responses_mask_gather, dtype=np.int32)), axis=1
        )
        all_responses_mask = np.concatenate(
            (np.zeros_like(left_padding_prompts, dtype=np.int32), responses_mask_gather), axis=1
        )
        all_prompt_completion_ids = np.concatenate((left_padding_prompts, right_padding_responses), axis=1)

        all_prompt_completion_ids = self._reconstruct_index(all_prompt_completion_ids, num_generations)
        all_prompts_mask = self._reconstruct_index(all_prompts_mask, num_generations)
        all_responses_mask = self._reconstruct_index(all_responses_mask, num_generations)
        advantages = self._reconstruct_index(advantages, num_generations)

        pack_level = self.max_pack_level
        pack_num = 2**pack_level
        while pack_level >= self.min_pack_level:
            packed_samples = self.pack_grpo_data(
                all_prompt_completion_ids, all_prompts_mask, all_responses_mask, advantages, pack_num
            )
            expected_data_num = self.num_train_prompts * num_generations // pack_num
            logger.info(
                f"Packing pack_num: {pack_num}; expected {expected_data_num} packed data, got {len(packed_samples)}"
            )
            if len(packed_samples) == expected_data_num:
                logger.info(f"Step: {self.make_exp_step}, pack num: {pack_num}")
                self.step_pack_num = pack_num
                self.step_pack_level = pack_level
                break
            elif pack_level == self.min_pack_level:
                logger.warning(
                    f"Step: {self.make_exp_step}, reached min pack level, \
                    expected {expected_data_num} packed data, got {len(packed_samples)}!"
                )
                self.step_pack_num = pack_num
                self.step_pack_level = pack_level
                break
            pack_num = pack_num // 2
            pack_level -= 1

        return packed_samples

    def compute_ref_log_probs(self, packed_samples):
        """compute_ref_log_probs"""
        pad_token_id = self.grpo_config.generate_config.sampling_config.pad_token_id
        ref_model_batch_size = self.grpo_config.ref_config.ref_model_batch_size
        total_ref_batch_size = ref_model_batch_size * self.ref_dp
        logger.info(
            f"total_ref_batch_size: ref_model_batch_size * ref_dp, {ref_model_batch_size} * "
            f"{self.ref_dp} = {total_ref_batch_size}"
        )
        while len(packed_samples) < total_ref_batch_size:
            packed_samples.append(packed_samples[0])
        all_ref_per_token_logps = np.zeros(
            (len(packed_samples), self.grpo_config.rl_config.seq_length), dtype=np.float32
        )
        self.ref.load()
        logger.info("ref_model load")
        ref_step_num = len(packed_samples) // total_ref_batch_size
        logger.info(f"ref model total steps: {ref_step_num}")
        with TimeConsumingCollector(f"reference model all steps {ref_step_num}"):
            for idx in range(ref_step_num):
                # responses_mask will be updated before ref model infer.
                prompt_completion_ids, actual_sequence_length, _ = self._construct_inputs_packing(
                    packed_samples, batch_size=total_ref_batch_size, idx=idx
                )

                prompt_completion_ids = np.pad(
                    prompt_completion_ids, ((0, 0), (0, 1)), "constant", constant_values=pad_token_id
                )
                sampels_tensor = Tensor(prompt_completion_ids[:, 1:], dtype=ms.int32)
                input_prompt_ids = Tensor(prompt_completion_ids[:, :-1], dtype=ms.int32)
                actual_sequence_length = Tensor(actual_sequence_length, dtype=ms.int32)

                logger.info("reference model step {} start".format(idx))
                with TimeConsumingCollector(f"ref model step {idx}"):
                    ref_per_token_logps = self.ref.compute_ref_log_prob(
                        input_prompt_ids, None, samples=sampels_tensor, actual_sequence_length=actual_sequence_length
                    )
                    ref_per_token_logps = ref_per_token_logps.asnumpy().astype(np.float32)
                logger.info("reference model step {} end".format(idx))

                all_ref_per_token_logps[idx * total_ref_batch_size : (idx + 1) * total_ref_batch_size, :] = (
                    ref_per_token_logps
                )

        self.ref.offload()
        logger.info("ref_model offload")
        return all_ref_per_token_logps

    def add_experience(self, packed_samples, all_ref_per_token_logps, all_old_per_token_logps):
        """add_experience"""
        grpo_rl_elements = []
        pad_token_id = self.grpo_config.generate_config.sampling_config.pad_token_id
        num_data_packed = len(packed_samples)

        if all_old_per_token_logps is None:
            all_old_per_token_logps = np.zeros(
                (num_data_packed, self.grpo_config.rl_config.seq_length), dtype=np.float32
            )
        if all_ref_per_token_logps is None:
            all_ref_per_token_logps = (
                -np.ones((num_data_packed, self.grpo_config.rl_config.seq_length), dtype=np.float32) * 20
            )

        for i in range(num_data_packed):
            prompt_completion_ids_temp = np.pad(
                packed_samples[i]["prompt_completion_ids"], ((0, 1),), "constant", constant_values=pad_token_id
            ).astype(np.int32)

            responses_mask_temp = np.pad(
                packed_samples[i]["responses_mask"], ((0, 1),), "constant", constant_values=0
            ).astype(np.int32)
            responses_mask_temp = responses_mask_temp[1:]

            ref_per_token_logps = all_ref_per_token_logps[i].astype(np.float32)
            ref_per_token_logps = ref_per_token_logps * responses_mask_temp
            ref_per_token_logps[np.isnan(ref_per_token_logps)] = 0.0

            old_per_token_logps = all_old_per_token_logps[i].astype(np.float32)
            old_per_token_logps = old_per_token_logps * responses_mask_temp
            old_per_token_logps[np.isnan(old_per_token_logps)] = 0.0

            grpodata = GRPOData(
                prompt_completion_ids=prompt_completion_ids_temp.astype(np.int32),
                responses_mask=responses_mask_temp.astype(np.int32),
                ref_per_token_logps=ref_per_token_logps.astype(np.float32),
                advantages=packed_samples[i]["advantages"].astype(np.float32),
                actual_sequence_length=packed_samples[i]["actual_sequence_length"].astype(np.int32),
                sample_index=packed_samples[i]["sample_index"].astype(np.int32),
                sample_valid_length=packed_samples[i]["sample_valid_length"].astype(np.int32),
                old_per_token_logps=old_per_token_logps.astype(np.float32),
            )
            grpo_rl_elements.append(grpodata)

        logger.info(f"grpo_rl_elements.length: {len(grpo_rl_elements)}")
        self.train.step_pack_num = self.step_pack_num
        self.train.step_pack_level = self.step_pack_level
        self.train.push_to_store(grpo_rl_elements)

    def make_experience(self, num_rollouts: int = 1, num_generations: int = 16):
        """make experience"""
        logger.info("Make experience start")
        logger.info(f"Generate {num_generations} times")

        batch = self._get_batch(num_rollouts)
        prompt_tensors = Tensor(batch[0], mstype.int32).asnumpy()
        solution_ids = Tensor(batch[1], mstype.int32).asnumpy()
        prompt_tensors_full = prompt_tensors
        repeat_solution_ids = solution_ids
        for _ in range(num_generations - 1):
            prompt_tensors_full = np.concatenate((prompt_tensors_full, prompt_tensors))
            repeat_solution_ids = np.concatenate((repeat_solution_ids, solution_ids))

        n_questions = batch[0].shape[0] // num_rollouts
        logger.info(f"n_questions:{n_questions}")
        logger.info(f"num_generations:{num_generations}")
        logger.info(f"num_rollouts:{num_rollouts}")
        sample_size = n_questions * num_generations * num_rollouts
        # Step 1: generate responses and masks.
        micro_bs = n_questions // self.infer_dp
        micro_num = num_rollouts * num_generations
        generate_profiler = profiler_start(
            self.grpo_config.profiler_config, role="actor_generate", profiler_iteration=self.profiler_iteration
        )
        generate_results = self.generate_sequence(micro_bs, micro_num, prompt_tensors_full)
        profiler_step(generate_profiler)

        # Step 2: calculate reward and advantages.
        with TimeConsumingCollector("calculate reward"):
            right_padding_responses, _, left_padding_prompts, _ = generate_results
            all_rewards, metrics = self.compute_rewards(
                left_padding_prompts, right_padding_responses, repeat_solution_ids, sample_size
            )

        advantages = self.compute_advantages(all_rewards, metrics, num_generations)

        MsProbe.save_data("advantages", advantages)

        packed_samples = self.construct_packed_samples(generate_results, advantages, num_generations)

        # Step 3: compute ref log probs.
        all_ref_per_token_logps = None
        if self.grpo_config.rl_config.enable_ref:
            ref_log_prob_profiler = profiler_start(
                self.grpo_config.profiler_config, role="reference_log_prob", profiler_iteration=self.profiler_iteration
            )
            all_ref_per_token_logps = self.compute_ref_log_probs(packed_samples)
            profiler_step(ref_log_prob_profiler)

        MsProbe.save_data("ref_log_prob", all_ref_per_token_logps)

        # Step 4: generate old log probs
        all_old_per_token_logps = None
        if self.grpo_config.rl_config.enable_oldpolicy:
            old_log_prob_profiler = profiler_start(
                self.grpo_config.profiler_config, role="actor_old_log_prob", profiler_iteration=self.profiler_iteration
            )
            generation_entropy, all_old_per_token_logps = self._generate_old_logps(packed_samples)
            if self.grpo_config.rl_config.calculate_entropy:
                logger.info(f"generation_entropy is {generation_entropy}")
                if self.tensor_writer:
                    self.tensor_writer.add_scalar(
                        "Training_grpo/generation-entropy", generation_entropy, global_step=self.make_exp_step
                    )

            profiler_step(old_log_prob_profiler)

        MsProbe.save_data("old_log_prob", all_old_per_token_logps)

        self.add_experience(packed_samples, all_ref_per_token_logps, all_old_per_token_logps)

        logger.info(f"total step {self.make_exp_step} metrics: {metrics}")
        if self.tensor_writer:
            add_metrics_to_tensorboard(self.tensor_writer, metrics, self.make_exp_step)
            logger.info(f"Add metrics of step {self.make_exp_step} to tensorboard")

        save_data_file = self.grpo_config.rl_config.save_data_file
        if save_data_file:
            if get_rank() % 8 == 0:
                self._save_grpoelement(save_data_file)
        self.make_exp_step += 1
        self.profiler_iteration += 1
        logger.info("Make experience end")
