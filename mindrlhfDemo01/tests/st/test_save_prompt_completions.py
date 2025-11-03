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
"""save prompt completions test case"""

import os
import pytest

from mindrlhf.utils import MetricData, save_prompt_completions_data
from mindrlhf.configs.grpo_configs import GRPOConfig

prompts = [
    "User: How many of the following are equal to $x^x+x^x$ for all $x>0$?\n$\\textbf{I:}$ $2x^x$ $\\qquad"
    "\\textbf{II:}$ $x^{2x}$ $\\qquad\\textbf{III:}$ $(2x)^x$ $\\qquad\\textbf{IV:}$ $(2x)^{2x}$\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant:<think>\n",
    "User: 9. (16 points) If the function\n$$\nf(x)=256 x^{9}-576 x^{7}+432 x^{5}-120 x^{3}+9 x \\text {, }\n"
    "$$\n\nfind the range of the function $f(x)$ for $x \\in[-1,1]$\nPlease reason step by step, and put your final "
    "answer within \\boxed{{}}.\n\nAssistant:<think>\n",
]
completions = [
    "The function attains its maximum value of 1 at \\( x = 1 \\) and its minimum value of -1 at \\( x = -1 \\) "
    "and \\( x = \\pm \\frac{1}{2} \\).\n\nThus, the range of the function \\( f(x) \\) over the interval \\( [-1, 1] "
    "\\) is:\n\n\\[\n\\boxed{[-1, 1]}\n\\]",
    "Sum of numerator and denominator: \\(277 + 280 = 557\\).\n\n\\boxed{557}",
]
answer_parsed_lst = [
    "$[-1,1]$",
    "557",
]
solution = [
    "$[-1,1]$",
    "$97$",
]
rewards = [1.0, 0.0]
responses_length_list = [200, 100]


@pytest.mark.save_prompt_completions
def test_save_interval_1():
    """
    Feature: transform param with zero redundancy
    Description: dptppp transform
    Expectation: Run success
    """
    config_path = 'qwen2_5/grpo_config_st.yaml'
    grpo_config = GRPOConfig(config_path)
    grpo_config.rl_config.save_prompt_completions_data = True
    grpo_config.rl_config.save_prompt_completions_interval = 1
    grpo_config.rl_config.save_prompt_completions_dir = './test_save_interval_1'

    for make_exp_step in range(0, 10):
        if (grpo_config.rl_config.save_prompt_completions_data and
                make_exp_step % grpo_config.rl_config.save_prompt_completions_interval == 0):
            save_kwargs = {
                MetricData.QUESTION.value: prompts,
                MetricData.ANSWER.value: completions,
                MetricData.PARSED_ANSWER.value: answer_parsed_lst,
                MetricData.SOLUTION.value: solution,
                MetricData.REWARD_PER_QUESTION.value: rewards,
                MetricData.COMPLETION_LENGTH_PER_QUESTION.value: responses_length_list
            }
            save_prompt_completions_data(grpo_config.rl_config.save_prompt_completions_dir, make_exp_step,
                                         **save_kwargs)
    for i in range(10):
        filename = f"prompt_completions_step_{i}.json"
        assert os.path.isfile(os.path.join(grpo_config.rl_config.save_prompt_completions_dir, filename))


@pytest.mark.save_prompt_completions
def test_save_interval_5():
    """
    Feature: transform param with zero redundancy
    Description: dptppp transform
    Expectation: Run success
    """
    config_path = 'qwen2_5/grpo_config_st.yaml'
    grpo_config = GRPOConfig(config_path)
    grpo_config.rl_config.save_prompt_completions_data = True
    grpo_config.rl_config.save_prompt_completions_interval = 5
    grpo_config.rl_config.save_prompt_completions_dir = './test_save_interval_5'

    for make_exp_step in range(0, 10):
        if (grpo_config.rl_config.save_prompt_completions_data and
                make_exp_step % grpo_config.rl_config.save_prompt_completions_interval == 0):
            save_kwargs = {
                MetricData.QUESTION.value: prompts,
                MetricData.ANSWER.value: completions,
                MetricData.PARSED_ANSWER.value: answer_parsed_lst,
                MetricData.SOLUTION.value: solution,
                MetricData.REWARD_PER_QUESTION.value: rewards,
                MetricData.COMPLETION_LENGTH_PER_QUESTION.value: responses_length_list
            }
            save_prompt_completions_data(grpo_config.rl_config.save_prompt_completions_dir, make_exp_step,
                                         **save_kwargs)
    for i in range(10):
        filename = f"prompt_completions_step_{i}.json"
        if i in (0, 5):
            assert os.path.isfile(os.path.join(grpo_config.rl_config.save_prompt_completions_dir, filename))
        else:
            assert not os.path.isfile(os.path.join(grpo_config.rl_config.save_prompt_completions_dir, filename))
