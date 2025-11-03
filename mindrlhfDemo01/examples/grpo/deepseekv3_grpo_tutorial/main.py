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
""" main """

import argparse
from mindspore import Tensor

no_patch_tensor_shape = Tensor.shape
#pylint: disable=C0413
from mindrlhf.trainer.spmd.grpo_trainer import GRPOTrainer


def main(input_args):
    """main process"""
    trainer = GRPOTrainer(no_patch_tensor_shape=no_patch_tensor_shape, args=input_args)
    trainer.run_grpo_train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="deepseek grpo train")
    parser.add_argument("--config", type=str, default=None, help="configs path", required=True)
    parser.add_argument("--model_name", type=str, default="deepseek", help="custom model name")
    parser.add_argument("--dataset_file", type=str, default=None, help="dataset file for training")
    parser.add_argument("--resume_training", action="store_true", default=False, help="resume training")
    parser.add_argument("--tokenizer_dir", type=str, default=None, help="the directory contain hf tokenizer files")
    parser.add_argument("--actor_checkpoint_path", type=str, default=None, help="the actor model file path for loading")
    parser.add_argument(
        "--ref_checkpoint_path", type=str, default=None, help="the reference model file path for loading"
    )
    parser.add_argument(
        "--generate_checkpoint_path", type=str, default=None, help="the generate model file path for loading"
    )
    parser.add_argument("--tokenizer_type", type=str, default="deepseek", help="tokenizer type")
    parser.add_argument("--verifier_function", type=str, default=None, help="verifier funcs")
    parser.add_argument("--verifier_weight", type=str, default=None, help="verifier weights")
    parser.add_argument("--tensorboard", type=str, default=None, help="enable tensorboard")
    parser.add_argument("--save_checkpoint_dir", type=str, default=None, help="save model path")
    args = parser.parse_args()
    main(args)
