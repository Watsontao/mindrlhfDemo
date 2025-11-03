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
"""Model utils collection."""
import os

from mindformers import MindFormerConfig
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.models.llama import LlamaTokenizerFast

from mindrlhf.configs.grpo_configs import GRPOConfig
from .qwen2_5 import Qwen2_5Tokenizer


def is_model_supported(model_name: str):
    """
    Whether model is supported, only support qwen, llama, deepseek.

    Args:
        model_name (str): model name.

    Returns:
        bool, whether supported.
    """
    return model_name in ["qwen", "llama", "deepseek"]


class TokenizerFactory:
    """
    Tokenizer factory to create tokenizer.
    """

    @classmethod
    def init_tokenizer(cls, grpo_config: GRPOConfig):
        """
        Create tokenizer instance.

        Args:
            grpo_config (GRPOConfig): grpo config instance.

        Returns:
            Tokenizer, tokenizer.
        """
        tokenizer_dir = grpo_config.rl_config.tokenizer_dir
        tokenizer_type = grpo_config.rl_config.tokenizer_type
        if not os.path.exists(tokenizer_dir):
            raise FileNotFoundError(f"{tokenizer_dir} is not existed.")
        if not os.path.isdir(tokenizer_dir):
            raise NotADirectoryError(f"{tokenizer_dir} is not a directory.")
        if tokenizer_type == "qwen":
            # For Qwen Model
            vocab_path = os.path.join(tokenizer_dir, "vocab.json")
            merges_file_path = os.path.join(tokenizer_dir, "merges.txt")
            return Qwen2_5Tokenizer(vocab_path, merges_file_path, add_bos_token=False, add_eos_token=False)
        if tokenizer_type == "deepseek":
            # For Deepseek Distill Model
            # e.g. When using Distill-Qwen Model,
            #      user should set model_name as 'qwen' and tokenizer_type as 'deepseek'
            sft_config_infer = MindFormerConfig(grpo_config.generate_config.model_config)
            tokenizer = LlamaTokenizerFast(
                os.path.join(tokenizer_dir, "tokenizer.json"),
                os.path.join(tokenizer_dir, "tokenizer.json"),
                unk_token=sft_config_infer.processor.tokenizer.unk_token,
                bos_token=sft_config_infer.processor.tokenizer.bos_token,
                eos_token=sft_config_infer.processor.tokenizer.eos_token,
                fast_tokenizer=True,
                trust_remote_code=True,
            )
            return tokenizer
        if tokenizer_type == "llama":
            # For Llama Model
            sft_config_infer = MindFormerConfig(grpo_config.generate_config.model_config)
            sft_config_infer.processor.tokenizer.tokenizer_file = tokenizer_dir
            sft_config_infer.processor.tokenizer.vocab_file = tokenizer_dir
            return build_tokenizer(
                sft_config_infer.processor.tokenizer, use_legacy=False, pretrained_model_dir=tokenizer_dir
            )
        raise ValueError(f"tokenizer_type should in [qwen, deepseek, llama], but get {tokenizer_type}")
