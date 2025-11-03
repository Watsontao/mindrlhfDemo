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
"""deepscaler reward function"""

def base_deepscaler_reward(prompts, completions, solution, **kwargs):
    """
    Base model deepscaler reward fn.

    Args:
        prompts (list): a list of prompts.
        completions (list): a list of completions.
        solution (list): a list of solutions.
        **kwargs: other parameters.

    Returns:
        Tuple[list, list], the first is a list of rewards, the second is a list of parsed answers.
    """
    from verifier.rule_based_rm_cot import compute_score
    rewards = []
    answer_parsed_lst = []
    for prompt, content, sol in zip(prompts, completions, solution):
        # base model force_think='<think>'
        reward_dict = compute_score('deepscaler', content, sol, prompt, force_think='<think>')
        reward = reward_dict['score']
        prediction = reward_dict['parsed_answer']
        rewards.append(reward)
        answer_parsed_lst.append(prediction)
    return rewards, answer_parsed_lst

def cot_deepscaler_reward(prompts, completions, solution, **kwargs):
    """
    Cot model deepscaler reward fn.

    Args:
        prompts (list): a list of prompts.
        completions (list): a list of completions.
        solution (list): a list of solutions.
        **kwargs: other parameters.

    Returns:
        Tuple[list, list], the first is a list of rewards, the second is a list of parsed answers.
    """
    from verifier.rule_based_rm_cot import compute_score
    rewards = []
    answer_parsed_lst = []
    for prompt, content, sol in zip(prompts, completions, solution):
        # cot model force_think=None
        reward_dict = compute_score('deepscaler', content, sol, prompt, force_think=None)
        reward = reward_dict['score']
        prediction = reward_dict['parsed_answer']
        rewards.append(reward)
        answer_parsed_lst.append(prediction)
    return rewards, answer_parsed_lst
