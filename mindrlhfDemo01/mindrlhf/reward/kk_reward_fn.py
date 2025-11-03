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
"""Reward functions for GRPO training on K&K dataset."""
import re
from typing import Dict, Tuple, Optional

from mindformers import logger


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.

    Args:
        solution_str: Raw response string from the language model

    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        logger.info("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))

    if not matches:
        logger.info("[Error] No valid answer tags found")
        return None, processed_str

    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str


def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """Parses ground truth solution text into status dictionary.

    Args:
        solution_text: Formatted solution text from dataset

    Returns:
        Dictionary mapping character names to their roles (knight/knave)
    """
    status_dict = {}
    logger.info("\n[Ground Truth Parsing]")

    for line in solution_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        match = re.search(r"\b([A-Za-z]+)\b.*?\b(knight|knave)\b", line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
            logger.info(f"  Found: {name} → {role}")
        else:
            logger.info(f"  [Warning] Unparsable line: '{line}'")

    return status_dict


def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """Parses model's answer text into status dictionary.

    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_names: List of character names requiring identification

    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    status_dict = {}
    logger.info("\n[Model Answer Parsing]")
    logger.info(f"  Expected characters: {expected_names}")

    knight_count = answer_text.lower().count("knight")
    knave_count = answer_text.lower().count("knave")

    logger.info(f"  Number of predicted roles: {knight_count + knave_count}")
    if knight_count + knave_count != len(expected_names):
        logger.info(f"  [Error] Number of characters mismatch: {knight_count + knave_count} != {len(expected_names)}")
        return None

    for name in expected_names:
        pattern = re.compile(rf"\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b", re.IGNORECASE)
        match = pattern.search(answer_text)

        if match:
            role = match.group(1).lower()
            status_dict[name] = role
            logger.info(f"  Found: {name} → {role}")
        else:
            logger.info(f"  [Error] Missing identification for {name}")
            return None

    return status_dict


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    logger.info("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        "think_start": ("<think>", 1),
        "think_end": ("</think>", 1),
        "answer_start": ("<answer>", 1),
        "answer_end": ("</answer>", 1),
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)

        logger.info(f"  {tag_str}: count={count}, position={pos}")

        if count != expected_count:
            logger.info(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    # pylint: disable=C0330
    if (
        positions.get("think_start", 0) > positions.get("think_end", 0)
        or positions.get("think_end", 0) > positions.get("answer_start", 0)
        or positions.get("answer_start", 0) > positions.get("answer_end", 0)
    ):
        logger.info("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        logger.info("  Tag sequence validation passed")

    return validation_passed


def compute_score(solution_str: str, ground_truth: str, format_reward: int = 1):
    """Computes comprehensive score for model response.

    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_reward: Points awarded/deducted for format correctness

    Returns:
        Total score (sum of format and answer rewards)
    """
    logger.info("\n" + "=" * 80)
    logger.info(" Processing New Sample ".center(80, "="))

    # Parse ground truth data
    solution_text = ground_truth
    gt_status = parse_solution_text_format(solution_text)
    expected_names = list(gt_status.keys())
    logger.info(f"[Ground Truth] Final identities: {gt_status}")

    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    logger.info(f"\n[Model Response]\n{processed_str}")

    # Validate response structure
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)
    logger.info(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    logger.info(f"  Format score: {format_score}")

    # Validate answer content
    answer_score = 0
    if format_correct and answer_text:
        pred_status = parse_model_answer(answer_text, expected_names)
        if pred_status:
            logger.info(f"\n[Content Validation]")
            logger.info(f"  Expected: {gt_status}")
            logger.info(f"  Predicted: {pred_status}")

            if pred_status == gt_status:
                answer_score = 2
                logger.info("  Content validation: FULL MATCH")
            else:
                answer_score = -1.5
                logger.info("  Content validation: MISMATCH")
        else:
            answer_score = -2
            logger.info("Fail to parse answer")
    else:
        answer_score = -2
        logger.info("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score
    logger.info("\n" + "-" * 80)
    logger.info(f" Final Score ".center(80, "-"))
    logger.info(f"  Format: {format_score}")
    logger.info(f"  Answer: {answer_score}")
    logger.info(f"  Total: {total_score}")
    logger.info("=" * 80 + "\n")

    return total_score


# pylint: disable=W0613
def kk_reward(prompts, completions, solution, **kwargs):
    """kk_reward"""
    rewards = []
    for i, _ in enumerate(completions):
        solution_str = prompts[i] + completions[i]
        ground_truth = solution[i]
        reward = compute_score(solution_str=solution_str, ground_truth=ground_truth)
        rewards.append(reward)
    return rewards
