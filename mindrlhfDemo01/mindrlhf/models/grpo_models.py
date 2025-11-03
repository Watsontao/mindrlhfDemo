# Copyright 2020-2025 Huawei Technologies Co., Ltd
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
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor, ops, mint, Parameter
from mindspore.ops import operations as P
from mindformers.models.utils import lazy_inline
from mindformers import logger
from mindrlhf.utils.generator import GeneratorMixin
from mindrlhf.configs import GRPOConfig
from .base_model import BaseModel

ENABLE_MSPROBE = True
try:
    from msprobe.mindspore import save
except ImportError:  # pylint: disable=W0703
    print("Failed to import save from msprobe.mindspore")
    ENABLE_MSPROBE = False


class CausalLMHybrid(BaseModel):
    """
    CausalLMHybrid
    """

    def __init__(self, model_config, grpo_config: GRPOConfig, is_training=True):
        super(CausalLMHybrid, self).__init__()
        if not is_training:
            model_config.dropout_rate = 0.0
        self.model_config = model_config
        self.grpo_config = grpo_config
        self.select_actor_model(model_config)
        self.lm_head.pipeline_stage = model_config.parallel_config.pipeline_stage - 1
        dp = model_config.parallel_config.data_parallel
        mp = model_config.parallel_config.model_parallel
        cp = model_config.parallel_config.context_parallel

        self.vocab_size = model_config.vocab_size
        self.chunk_size = grpo_config.rl_config.chunk_size
        self.seq_length = grpo_config.rl_config.seq_length
        self.cast = P.Cast()
        self.all_ones_attention_mask = Tensor(np.ones((1, 1, self.seq_length)), mstype.float32)

        self.squeeze = P.Squeeze(axis=-1).shard(((dp, 1, 1),))
        self.squeeze_no_shard_1 = P.Squeeze(axis=-1).shard(((1, 1),))
        self.squeeze_no_shard_2 = P.Squeeze(axis=-1).shard(((1, 1, 1),))
        self.unsqueeze = P.ExpandDims().shard(((dp * mp * cp,),))
        self.unsqueeze_2 = P.ExpandDims().shard(((dp, 1),))
        self.reshape = P.Reshape()
        self.gather = P.Gather().shard(((dp, mp), (1,)))
        self.gatherd = P.GatherD().shard(((dp * mp * cp, 1), (dp * mp * cp, 1)))
        self.gatherd_2 = P.GatherD().shard(((dp, 1, 1), (dp, 1, 1)))
        self.logsoftmax = P.LogSoftmax().shard(((dp * mp * cp, 1),))
        self.logsoftmax_1 = P.LogSoftmax().shard(((dp * mp * cp, 1),))
        self.logsoftmax_2 = P.LogSoftmax().shard(((dp, 1, 1),))
        self.expaned = P.ExpandDims().shard(((dp, mp),))
        self.lsf_dtype = mstype.float32

        self.pow = P.Pow().shard(((dp, 1), ()))
        self.argmax_no_shard = P.Argmax(-1).shard(((1, 1),))
        self.argmax = P.Argmax(-1).shard(((dp, mp),))
        self.add_shard = P.Add().shard(((1, 1, 1), ()))

        self.pad_logits = ops.Pad(((0, 0), (1, 0), (0, 0))).shard(((dp * mp, 1, 1),))
        self.pad_samples = ops.Pad(((0, 0), (1, 0))).shard(((dp * mp, 1),))

        self.expaned = P.ExpandDims().shard(((dp, mp),))
        self.dp = dp
        self.old_log_ratio = Parameter(Tensor(0, mstype.float32), name="old_log_ratio", requires_grad=False)
        self.kl_loss = Parameter(Tensor(0, mstype.float32), name="kl_loss", requires_grad=False)
        self.actor_loss = Parameter(Tensor(0, mstype.float32), name="actor_loss", requires_grad=False)
        self.clipfrac = Parameter(Tensor(0, mstype.float32), name="clipfrac", requires_grad=False)

        self.softmax = P.Softmax(axis=-1).shard(((dp * mp * cp, 1),))
        self.reducemax = P.ReduceMax(keep_dims=True).shard(((dp * mp * cp, 1),))
        self.reducesum = P.ReduceSum(keep_dims=False).shard(((dp * mp * cp, 1),))
        self.exp = P.Exp().shard(((dp * mp * cp, 1),))
        self.log = P.Log().shard(((dp * mp * cp,),))
        self.mul = P.Mul().shard(((dp * mp * cp,), (dp * mp * cp,)))

    def offset_actual_sequence_length(self, data, offset):
        """add offset to data"""
        bs = data.shape[0] // self.dp
        n = data.shape[1]
        data_type = data.dtype
        data = data.reshape((self.dp, bs, n))
        offsets = self.cast(ops.range(0, bs * offset, offset).reshape((1, bs, 1)), data_type)
        data = data + offsets
        actual_seq_lenth = self.cast(ops.reshape(data, (-1,)), data_type)
        return actual_seq_lenth

    def process_logits2(self, logits, current_index=None, is_first_iteration=False, use_past=False):
        r"""
        process_logits2
        """
        logits = logits.reshape(-1, logits.shape[-1])
        if use_past and not is_first_iteration:
            logits = logits
        elif current_index is not None:
            index = current_index.view(-1)
            logits = self.gather(logits, index, 0)
        top_token_id = self.argmax_no_shard(logits)
        top_token_id = top_token_id.view(-1, 1)
        return top_token_id

    def logprobs_of_labels(self, logits, samples):
        """
        Calculate the log value of the label
        """
        logits = self.cast(logits, self.lsf_dtype)
        bs, seq_len = samples.shape
        logprobs = self.logsoftmax(logits)  # [bs*seq_len, vocab_size]
        samples = self.expaned(samples, -1)
        samples = self.reshape(samples, (bs * seq_len, -1))
        logprobs = self.squeeze_no_shard_1(self.gatherd(logprobs, -1, samples))  # [bs, seq_len]
        logprobs = self.reshape(logprobs, (bs, seq_len))
        return logprobs  # [bs, seq_len-1]

    def entropy_from_logits(self, logits):
        """Calculate entropy from logits"""
        pb = self.softmax(logits)
        mul_res = self.mul(pb, logits)
        sum_res = self.reducesum(mul_res, axis=-1)
        entropy = self.logsumexp(logits, dim=-1) - sum_res
        return entropy

    def logsumexp(self, x, dim, keep_dims=False):
        """Calculate log_sum_exp"""
        x_max = self.reducemax(x, dim)
        x_exp = self.exp(x - x_max)
        x_sumexp = self.reducesum(x_exp, axis=-1)
        x_logsumexp = self.log(x_sumexp)
        if not keep_dims:
            x_max = self.squeeze_no_shard_1(x_max)
        return x_logsumexp + x_max

    def construct(
        self,
        # inputs for the llm
        input_ids,
        input_position=None,
        batch_valid_length=None,
        slot_mapping=None,
        # inputs for `process_logits`
        is_first_iteration=False,
        use_past=False,
        # inputs for choosing the output branch
        samples=None,
        actual_sequence_length=None,
        return_full_logit=False,
        is_ref=False,
        calculate_entropy=False,
    ):
        """
        construct function for CausalLMHybrid
        """
        batch_size, seq_length = input_ids.shape

        if actual_sequence_length is not None:
            if len(actual_sequence_length.shape) > 1:
                bsz, _ = actual_sequence_length.shape
                if bsz > 1:
                    actual_sequence_length = self.offset_actual_sequence_length(
                        actual_sequence_length, input_ids.shape[1]
                    )
        if self.model_type == "llama":
            if self.model.phase == "train":
                tokens = input_ids
            else:
                tokens = input_ids
                if batch_valid_length is None or slot_mapping is None:
                    bsz, seqlen = tokens.shape
                    batch_valid_length = ops.ones((bsz,), mstype.int32).reshape(-1)
                    slot_mapping = Tensor(np.ones(shape=tuple([bsz * seqlen])), mstype.int32)
            output_states = self.backbone(
                tokens,
                None,
                batch_valid_length,
                None,
                None,
                None,
                slot_mapping,
                None,
                None,
                None,
                None,
                None,
                actual_sequence_length,
            )
            logits_2d = self.lm_head(output_states)
        elif self.model_type == "deepseek_infer":
            tokens = input_ids
            if batch_valid_length is None or slot_mapping is None:
                bsz, seqlen = tokens.shape
                batch_valid_length = ops.ones((bsz,), mstype.int32).reshape(-1)
                slot_mapping = Tensor(np.ones(shape=tuple([bsz * seqlen])), mstype.int32)
            output_states = self.backbone(tokens, batch_valid_length=batch_valid_length, slot_mapping=slot_mapping)
            logits_2d = self.lm_head(output_states)
        elif self.model_type == "deepseek_training":
            tokens = input_ids
            output_states, _ = self.backbone(tokens, extra_loss=0.0)
            logits_2d = self.lm_head(output_states)
        else:
            raise NotImplementedError("only support {}".format(" ".join(self.model_type)))

        logits = self.reshape(logits_2d, (batch_size * seq_length, -1))
        # used in inference of make_experience and grpo
        if samples is not None:
            if is_ref:
                return logits
            logprobs_labels = self.logprobs_of_labels(logits, samples)
            if calculate_entropy:
                entropy = self.entropy_from_logits(logits)
                entropy = entropy.reshape(batch_size, seq_length)
                return entropy, logprobs_labels
            return logprobs_labels
        # used in generate
        if not return_full_logit:
            outputs = self.process_logits2(logits, input_position, is_first_iteration, use_past)
            return outputs
        # used in pretrain loss
        logits = self.add_shard(logits, 0)
        return logits


class GRPOModel(nn.Cell, GeneratorMixin):
    """GRPOModel"""

    @lazy_inline
    def __init__(self, grpo_config: GRPOConfig, policy_model):
        super(GRPOModel, self).__init__()
        self.grpo_config = grpo_config
        self.beta = grpo_config.rl_config.beta
        self.pad_token_id = Tensor(grpo_config.generate_config.sampling_config.pad_token_id, mstype.int32)
        self.policy_model = policy_model
        self.enable_oldpolicy = self.grpo_config.rl_config.enable_oldpolicy
        self.enable_full_monitor = self.grpo_config.rl_config.enable_full_monitor
        self.epsilon_high = self.grpo_config.rl_config.epsilon_high
        self.epsilon_low = self.grpo_config.rl_config.epsilon_low
        logger.info(
            f"enable_oldpolicy: {self.enable_oldpolicy}, "
            f"epsilon_low: {self.epsilon_low}, epsilon_high: {self.epsilon_high}"
        )

        self.exp = P.Exp()
        self.gamma = 1
        self.lam = 0.95
        self.depend = P.Depend()
        self.dp = self.policy_model.dp
        self.cast = P.Cast()
        self.dump = P.TensorDump()

        if grpo_config.rl_config.loss_agg_mode in ["seq-mean-token-mean", "token-mean"]:
            self.loss_agg_mode = grpo_config.rl_config.loss_agg_mode
        else:
            raise ValueError("loss_agg_mode only supports 'seq-mean-token-mean' or 'token-mean'")

    def masked_mean(self, data, mask, sample_index, pack_sample_num, sample_valid_length, dim=None):
        """Masked mean"""
        if mask is None:
            return data.mean(axis=dim)
        deno = self.batch_unsorted_segment_sum((data * mask), sample_index, pack_sample_num)  # [bs, packed_sample_num]
        nume = sample_valid_length  # [bs, packed_sample_num]
        return deno / nume

    def compute_approx_kl(self, log_probs, log_probs_base):
        """Compute approx kl"""
        log_ratio = self.cast(log_probs, mstype.float32) - self.cast(log_probs_base, mstype.float32)
        log_ratio = -log_ratio
        log_ratio = self.exp(log_ratio) - 1 - log_ratio
        return log_ratio

    def batch_unsorted_segment_sum(self, input_ids, segments_ids, num_segments):
        """
        GRPOModel
        batch unsorted_segment_sum

        Args:
            input_ids: shape (batch_size, seq_len)
            segments_ids: shape (batch_size, seq_len)
            num_segments: int
        Returns:
            shape (batch_size, num_segments)
        """
        bs, _ = input_ids.shape
        offsets = ops.arange(0, bs * num_segments, num_segments)
        # ensure segment_id uniqueness after reshape
        seg_off = segments_ids + offsets.view(bs, 1)
        flat_sum = ops.unsorted_segment_sum(input_ids.view(-1), seg_off.view(-1), bs * num_segments)
        return flat_sum.view(bs, num_segments)

    def agg_loss(
        self,
        loss_mat,
        loss_mask,
        sample_index,
        pack_sample_num,
        sample_valid_len,
        real_sample_num,
        step_mean_response_len,
    ):
        """aggregate loss"""
        masked_loss = loss_mat * loss_mask  # [bs, seq_len]
        if self.loss_agg_mode == "seq-mean-token-mean":
            nume = self.batch_unsorted_segment_sum(
                masked_loss, sample_index, pack_sample_num
            )  # [bs, packed_sample_num]
            deno = sample_valid_len  # [bs, packed_sample_num]
            loss = (nume / deno).sum() / real_sample_num
        elif self.loss_agg_mode == "token-mean":
            loss_sum = masked_loss.sum()
            deno = step_mean_response_len * pack_sample_num + 1e-8
            loss = loss_sum / deno
        return loss

    def offset_actual_sequence_length(self, data, offset):
        """add offset to data"""
        bs = data.shape[0] // self.dp
        n = data.shape[1]
        data_type = data.dtype
        data = data.reshape((self.dp, bs, n))
        offsets = self.cast(ops.range(0, bs * offset, offset).reshape((1, bs, 1)), data_type)
        data = data + offsets
        actual_seq_lenth = self.cast(ops.reshape(data, (-1,)), data_type)
        return actual_seq_lenth

    def get_ratio(self):
        """get ratio"""
        return self.old_log_ratio.value()

    def get_clipfrac(self):
        """get clip fraction"""
        return self.clipfrac.value()

    def get_klloss(self):
        """get KL loss"""
        return self.kl_loss.value()

    def get_actor_loss(self):
        """get actor loss"""
        return self.actor_loss.value()

    def set_klloss(self):
        """set KL loss"""
        ops.assign(self.kl_loss, Parameter(Tensor(0, mstype.float32), name="kl_loss", requires_grad=False))

    def set_actor_loss(self):
        """set actor loss"""
        ops.assign(self.actor_loss, Parameter(Tensor(0, mstype.float32), name="actor_loss", requires_grad=False))

    def set_ratio(self):
        """set ratio"""
        ops.assign(self.old_log_ratio, Parameter(Tensor(0, mstype.float32), name="old_log_ratio", requires_grad=False))

    def set_clipfrac(self):
        """set clip fraction"""
        ops.assign(self.clipfrac, Parameter(Tensor(0, mstype.float32), name="clipfrac", requires_grad=False))

    def construct(
        self,
        prompt_completion_ids,  # [bs, seq_len]
        responses_mask,  # [bs, seq_len]
        ref_per_token_logps,  # [bs, seq_len]
        advantages,  # [bs, seq_len]
        actual_sequence_length,  # [bs, packed_sample_num]
        sample_index,  # [bs, seq_len]
        sample_valid_length,  # [bs, packed_sample_num]
        old_per_token_logps,  # [bs, seq_len]
        step_mean_response_len,
    ):
        """construct function for GRPOModel"""
        pack_sample_num = sample_valid_length.shape[1]
        real_sample_num = ops.sum(sample_valid_length != 1, dtype=mstype.int32)

        input_ids = prompt_completion_ids[:, :-1]  # [bs, seq_len]
        samples = prompt_completion_ids[:, 1:]  # [bs, seq_len]
        actual_sequence_length = self.offset_actual_sequence_length(actual_sequence_length, input_ids.shape[1])
        per_token_logps = self.policy_model(
            input_ids, None, None, None, False, False, samples, actual_sequence_length, False, False
        )  # [bs, seq_len]
        per_token_logps = per_token_logps * responses_mask

        per_token_kl = self.compute_approx_kl(per_token_logps, ref_per_token_logps)

        kl_loss = self.agg_loss(
            per_token_kl,
            responses_mask,
            sample_index,
            pack_sample_num,
            sample_valid_length,
            real_sample_num,
            step_mean_response_len,
        )

        if not self.enable_oldpolicy:
            old_per_token_logps = ops.stop_gradient(per_token_logps)
        ratio = self.exp(per_token_logps - old_per_token_logps)
        if self.enable_full_monitor:
            old_log_ratio = self.masked_mean(
                ratio, responses_mask, sample_index, pack_sample_num, sample_valid_length, dim=-1
            ).mean()
            ops.assign_add(self.old_log_ratio, old_log_ratio)
        surr1 = ratio * advantages
        if self.enable_oldpolicy:
            surr2 = mint.clamp(ratio, min=(1.0 - self.epsilon_low), max=(1.0 + self.epsilon_high)) * advantages
            per_token_pg = -mint.min(surr1, surr2)
            if self.enable_full_monitor:
                clipfrac = mint.gt(surr1, surr2)
                clipfrac = self.masked_mean(
                    clipfrac, responses_mask, sample_index, pack_sample_num, sample_valid_length, dim=-1
                ).mean()
                ops.assign_add(self.clipfrac, clipfrac)
        else:
            per_token_pg = -surr1
        pg_loss = self.agg_loss(
            per_token_pg,
            responses_mask,
            sample_index,
            pack_sample_num,
            sample_valid_length,
            real_sample_num,
            step_mean_response_len,
        )

        if self.enable_full_monitor:
            ops.assign_add(self.kl_loss, kl_loss)
            ops.assign_add(self.actor_loss, pg_loss)
        loss = pg_loss + kl_loss * self.beta

        if ENABLE_MSPROBE and self.grpo_config.msprobe_config.msprobe and self.grpo_config.msprobe_config.key_data_dump:
            save(self.grpo_config.msprobe_config.dump_path, "log_prob", per_token_logps)
            save(self.grpo_config.msprobe_config.dump_path, "kl_loss", kl_loss)
            save(self.grpo_config.msprobe_config.dump_path, "loss", loss)
        return loss


class GRPOModelInfer(nn.Cell):
    def __init__(self, grpo_config: GRPOConfig, policy_model):
        super(GRPOModelInfer, self).__init__()
        self.grpo_model = GRPOModel(grpo_config, policy_model)

    def construct(self, *args, **kwargs):
        return self.grpo_model(*args, **kwargs)


class GRPOModelTrain(nn.Cell):
    def __init__(self, grpo_config: GRPOConfig, policy_model):
        super(GRPOModelTrain, self).__init__()
        self.grpo_model_train = GRPOModel(grpo_config, policy_model)

    def construct(self, *args, **kwargs):
        return self.grpo_model_train(*args, **kwargs)
