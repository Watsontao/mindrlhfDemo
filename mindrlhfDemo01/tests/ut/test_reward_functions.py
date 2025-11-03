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
"test_reward_functions.py"

import os
import sys
import json
import pytest

# Configure mindformer path to access the modules
WORKDIR = os.path.dirname(os.path.abspath(__file__))
print(f"WORKDIR is {WORKDIR}")
MINDRLHF_PATH = os.path.join(WORKDIR, "../../")
MINDFORMERS_PATH = os.path.join(WORKDIR, "../st/mindformers")
sys.path = [MINDRLHF_PATH, MINDFORMERS_PATH] + sys.path
print(f"sys.path is {sys.path}")

# Import the reward functions to be tested
# pylint: disable=C0413
from mindrlhf.reward.reward_fn import accuracy_reward, qwen_accuracy_reward, accuracy_reward_2, format_reward

# pylint: disable=C0413
from mindrlhf.reward.kk_reward_fn import kk_reward


# ================================================
# Test: accuracy_reward (basic numeric matching)
# ================================================
@pytest.mark.parametrize(
    "completions, solutions, expected_rewards",
    [
        (["42"], ["42"], [1.0]),  # exact match
        (["43"], ["42"], [0.0]),  # mismatch
        (["4.2"], ["4.2"], [1.0]),  # decimal
        (["1,234"], ["1234"], [1.0]),  # comma-separated number
        (["-42"], ["-42"], [1.0]),  # negative number
        (["Answer is 42 and 43"], ["43"], [1.0]),  # last number match
        ([""], ["42"], [0.0]),  # empty prediction
        (["No numbers here"], ["42"], [0.0]),  # no number at all
    ],
)
def test_accuracy_reward(completions, solutions, expected_rewards):
    """
    Feature: accuracy_reward function (basic numeric matching)
    Description: This test checks basic number parsing by comparing a list of
                 completions to a list of solutions
    Expectation: The rewards should be 1.0 for matching numbers and 0.0 otherwise
    """
    rewards, _ = accuracy_reward(completions, solutions)
    assert rewards == expected_rewards


# ======================================================
# Test: qwen_accuracy_reward (\boxed{} number parsing)
# ======================================================
@pytest.mark.parametrize(
    "completions, solutions, expected_rewards",
    [
        (["\\boxed{42}"], ["42"], [1.0]),  # correct boxed
        (["Text \\boxed{43} extra"], ["42"], [0.0]),  # mismatch
        (["\\boxed{-42}"], ["-42"], [1.0]),  # negative number
        (["\\boxed{4.2}"], ["4.2"], [1.0]),  # decimal
        (["\\boxed{43}\\boxed{42}"], ["42"], [1.0]),  # takes last boxed
        (["No boxed number"], ["42"], [0.0]),  # no box
        (["\\boxed{}"], ["42"], [0.0]),  # empty box
    ],
)
def test_qwen_accuracy_reward(completions, solutions, expected_rewards):
    """
    Feature: qwen_accuracy_reward function (LaTeX \boxed{} parsing)
    Description: This test checks the extraction and comparison of numbers
                 within \boxed{} LaTeX tags
    Expectation: The rewards should be 1.0 when the boxed number matches
                 the solution, 0.0 otherwise
    """
    rewards, _ = qwen_accuracy_reward(completions, solutions)
    assert rewards == expected_rewards


with open(os.path.join(WORKDIR, "qwen_reward_dataset.json"), "r", encoding="utf-8") as _f:
    qwen_data = json.load(_f)

# Build param list: (completion, solution, expected_reward)
qwen_cases = [(entry["answer"], entry["solution"], entry["reward"]) for entry in qwen_data]


@pytest.mark.parametrize("completion, solution, expected_reward", qwen_cases)
def test_qwen_accuracy_reward_with_real_dataset(completion, solution, expected_reward):
    """
    Feature: qwen_accuracy_reward function with real dataset
    Description: Test the function against real-world examples loaded from
                 a JSON dataset file
    Expectation: The computed rewards should match the expected rewards
                 from the dataset
    """
    rewards, _ = qwen_accuracy_reward([completion], [solution])
    assert rewards == [expected_reward]


# ===========================================
# Test: accuracy_reward_2 (symbolic math)
# ===========================================


@pytest.mark.parametrize(
    "completion, solution, expected_reward",
    [
        (r"\boxed{42}", "$42$", 1.0),  # exact numeric match with different LaTeX wrappers
        (r"\boxed{x}", "$x$", 1.0),  # symbolic variable matching
        (r"\boxed{y = mx + b}", "$y = mx + b$", 1.0),  # equation literal match
        (r"\boxed{ 2 + 2 }", "$2+2$", 1.0),  # whitespace insensitivity
        (r"\boxed{43}", "$42$", 0.0),  # number mismatch
        (r"\boxed{a}", "$b$", 0.0),  # symbol mismatch
        (r"\frac{1{2}", r"\frac{1}{2}", 0.0),  # malformed LaTeX input
        (r"", r"42", 0.0),  # empty completion
        (r"42", r"\invalid", 0.0),  # unparsable solution
    ],
)
def test_accuracy_reward_2_math_validation(completion, solution, expected_reward):
    """
    Feature: accuracy_reward_2 function (symbolic math validation)
    Description: Test symbolic math equivalence checking through LaTeX parsing
                 and mathematical verification
    Expectation: The rewards should be 1.0 when expressions are mathematically
                 equivalent, 0.0 otherwise
    """
    # Attempt to import required symbolic math modules; skip if unavailable
    pytest.importorskip("latex2sympy2_extended")
    pytest.importorskip("math_verify")

    rewards, _ = accuracy_reward_2([completion], [solution])
    assert rewards == [expected_reward]


# ========================================
# Test: format_reward (<think> + <answer>)
# ========================================
@pytest.mark.parametrize(
    "completions, expected_rewards",
    [
        (["<think>abc</think> <answer>42</answer>"], [1.0]),  # valid format
        (["<think>reason</think>\n<answer>42</answer>"], [1.0]),  # multiline
        (["<think>abc</think><answer>42</answer>"], [1.0]),  # no space
        (["<think>abc</think> answer 42"], [0.0]),  # missing answer tag
        (["<answer>42</answer>"], [0.0]),  # missing think tag
        (["random text"], [0.0]),  # no tags
        (["<think><answer>42</answer></think>"], [0.0]),  # tag nesting error
        ([""], [0.0]),  # empty input
    ],
)
def test_format_reward(completions, expected_rewards):
    """
    Feature: format_reward function
    Description: Test for proper formatting with both think (<think>...</think>)
                 and answer (<answer>...</answer>) tags
    Expectation: The rewards should be 1.0 for properly formatted inputs,
                 0.0 otherwise
    """
    rewards = format_reward(completions)
    assert rewards == expected_rewards


# ======================================================================================================
# Test: kk_reward (K&K dataset: https://huggingface.co/datasets/K-and-K/knights-and-knaves)
#                 (Paper: https://arxiv.org/abs/2502.14768)
# ======================================================================================================

WORKDIR = os.path.dirname(__file__)
with open(os.path.join(WORKDIR, "kk_reward_dataset.json"), "r") as f:
    kk_cases = json.load(f)


@pytest.mark.parametrize(
    "prompts,completions,solutions,expected_rewards",
    [(case["prompts"], case["completions"], case["solutions"], case["expected_rewards"]) for case in kk_cases],
)
def test_kk_reward_with_realistic_examples(prompts, completions, solutions, expected_rewards):
    """
    Feature: kk_reward function (Knights & Knaves dataset)
    Description: Test the specialized reward function against realistic examples
                 from the Knights & Knaves reasoning dataset
    Expectation: The computed rewards should match the expected rewards provided
                 in the test dataset
    """
    rewards = kk_reward(prompts, completions, solutions)
    assert rewards == expected_rewards
