"""
Reward computation for code localization multi-agent RL training.

Implements:
- DICE coefficient for comparing predicted vs ground truth function sets
- Navigator reward: DICE(S_pred, S_gt) + ToolSuccessRate
- Inspector reward: DICE(L_pred, F_relevant) + ToolSuccessRate
  where F_relevant is the subset of ground truth within the Inspector's explored files
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_location(loc: str) -> str:
    """Normalize a function location string for consistent comparison.

    Handles formats like:
    - "file_path:function_name"
    - "file_path:ClassName.method_name"
    """
    return loc.strip()


def _extract_file(loc: str) -> str:
    """Extract the file path from a function location string."""
    if ":" in loc:
        return loc.split(":")[0].strip()
    return loc.strip()


def compute_dice(pred_set: set[str], gt_set: set[str]) -> float:
    """Compute DICE coefficient between two sets.

    DICE = 2 * |pred & gt| / (|pred| + |gt|)
    """
    if not pred_set and not gt_set:
        return 0.0
    if not pred_set or not gt_set:
        return 0.0
    intersection = pred_set & gt_set
    return 2.0 * len(intersection) / (len(pred_set) + len(gt_set))


def compute_truncated_dice(
    pred_list: list[str],
    gt_set: set[str],
) -> float:
    """Compute DICE with truncation.

    k = min(|pred|, |gt|)
    pred* = pred[:k]
    DICE(pred*, gt) = 2|pred* & gt| / (|pred*| + |gt|)

    Args:
        pred_list: Ordered list of predicted locations (most suspicious first).
        gt_set: Set of ground truth locations.

    Returns:
        DICE score in [0, 1].
    """
    if not pred_list and not gt_set:
        return 0.0
    if not pred_list or not gt_set:
        return 0.0

    k = min(len(pred_list), len(gt_set))
    truncated = set(_normalize_location(loc) for loc in pred_list[:k])
    gt_normalized = set(_normalize_location(loc) for loc in gt_set)
    return compute_dice(truncated, gt_normalized)


def compute_tool_success_rate(
    total_tool_calls: int,
    successful_tool_calls: int,
) -> float:
    """Compute tool call success rate.

    ToolSuccessRate = successful / total
    """
    if total_tool_calls == 0:
        return 0.0  # No tool calls = no failures, but I need tool calls, no tool calls means low score
    return successful_tool_calls / total_tool_calls


def compute_navigator_reward(
    predicted_locations: list[str],
    ground_truth_locations: list[str],
    total_tool_calls: int = 0,
    successful_tool_calls: int = 0,
) -> dict[str, float]:
    """Compute Navigator (Main Agent) reward.

    R_nav = DICE(S_pred, S_gt) + ToolSuccessRate

    Args:
        predicted_locations: List of predicted suspicious function locations.
        ground_truth_locations: List of ground truth function locations.
        total_tool_calls: Total number of tool calls made by Navigator.
        successful_tool_calls: Number of successfully executed tool calls.

    Returns:
        Dict with 'score', 'dice', and 'tool_success_rate'.
    """
    pred_set = set(_normalize_location(loc) for loc in predicted_locations)
    gt_set = set(_normalize_location(loc) for loc in ground_truth_locations)
    dice = compute_dice(pred_set, gt_set)
    tool_rate = compute_tool_success_rate(total_tool_calls, successful_tool_calls)
    score = dice + tool_rate

    return {
        "score": score,
        "dice": dice,
        "tool_success_rate": tool_rate,
    }


def compute_inspector_reward(
    predicted_locations: list[str],
    ground_truth_locations: list[str],
    explored_files: set[str],
    total_tool_calls: int = 0,
    successful_tool_calls: int = 0,
) -> dict[str, Any]:
    """Compute Inspector (Sub Agent) reward.

    F_relevant = {f in S_gt | file(f) in T_ins}
    R_ins = DICE(L_pred, F_relevant) + ToolSuccessRate

    If F_relevant is empty, returns None to indicate the trajectory should be discarded.

    Args:
        predicted_locations: List of predicted suspicious function locations.
        ground_truth_locations: List of ground truth function locations.
        explored_files: Set of files explored by the Inspector.
        total_tool_calls: Total number of tool calls made.
        successful_tool_calls: Number of successful tool calls.

    Returns:
        Dict with 'score', 'dice', 'tool_success_rate', 'f_relevant_size', 'discard'.
        If F_relevant is empty, 'discard' is True and 'score' is None.
    """
    # Compute F_relevant: ground truth functions whose file is in explored_files
    gt_normalized = [_normalize_location(loc) for loc in ground_truth_locations]
    f_relevant = set()
    for loc in gt_normalized:
        file_part = _extract_file(loc)
        if file_part in explored_files:
            f_relevant.add(loc)

    if not f_relevant:
        return {
            "score": None,
            "dice": 0.0,
            "tool_success_rate": compute_tool_success_rate(total_tool_calls, successful_tool_calls),
            "f_relevant_size": 0,
            "discard": True,
        }

    pred_set = set(_normalize_location(loc) for loc in predicted_locations)
    dice = compute_dice(pred_set, f_relevant)
    tool_rate = compute_tool_success_rate(total_tool_calls, successful_tool_calls)
    score = dice + tool_rate

    return {
        "score": score,
        "dice": dice,
        "tool_success_rate": tool_rate,
        "f_relevant_size": len(f_relevant),
        "discard": False,
    }


def parse_ground_truth(ground_truth: Any) -> list[str]:
    """Parse ground truth from string, JSON, or sequence format.

    Supports formats:
    - JSON list: '["file:func1", "file:Class.method"]'
    - Newline-separated: "file:func1\\nfile:Class.method"
    - Comma-separated: "file:func1, file:Class.method"
    - Python list/tuple/set: ["file:func1", "file:Class.method"]
    """
    if not ground_truth:
        return []

    if isinstance(ground_truth, list | tuple | set):
        return [loc_str for loc_str in (str(loc).strip() for loc in ground_truth) if loc_str]

    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)

    # Try JSON
    try:
        parsed = json.loads(ground_truth)
        if isinstance(parsed, list):
            return [str(loc) for loc in parsed]
        if isinstance(parsed, str):
            ground_truth = parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Try newline-separated
    if "\n" in ground_truth:
        return [loc.strip() for loc in ground_truth.strip().split("\n") if loc.strip()]

    # Try comma-separated
    if "," in ground_truth:
        return [loc.strip() for loc in ground_truth.split(",") if loc.strip()]

    # Single location
    return [ground_truth.strip()] if ground_truth.strip() else []


def parse_predicted_locations(response_text: str) -> list[str]:
    """Parse predicted locations from model response text.

    Looks for <result> block with JSON containing 'suspicious' list.
    Falls back to extracting locations from the text.
    """
    # Try to find <result> block
    result_match = re.search(r"<result>(.*?)</result>", response_text, re.DOTALL | re.IGNORECASE)
    if result_match:
        try:
            result_json = json.loads(result_match.group(1).strip())
            suspicious = result_json.get("suspicious", [])
            return [item["location"] for item in suspicious if "location" in item]
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    return []


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Compute reward score for code localization task.

    This function serves as the entry point for verl's reward computation system.
    It dispatches to navigator or inspector reward based on the role in extra_info.

    Args:
        data_source: Dataset identifier (e.g., "code_localization").
        solution_str: The model's full response text.
        ground_truth: Ground truth function locations. Can be a JSON string or a Python sequence.
        extra_info: Dict containing 'role', 'explored_files', 'tool_stats', etc.

    Returns:
        Dict with 'score' and additional reward breakdown info.
    """
    extra_info = extra_info or {}
    role = extra_info.get("role", "navigator")
    gt_locations = parse_ground_truth(ground_truth)

    # Extract tool stats
    tool_stats = extra_info.get("tool_stats", {})
    total_tool_calls = tool_stats.get("total", 0)
    successful_tool_calls = tool_stats.get("successful", 0)

    if role == "inspector":
        # Inspector reward
        predicted = parse_predicted_locations(solution_str)
        explored_files = set(extra_info.get("explored_files", []))

        result = compute_inspector_reward(
            predicted_locations=predicted,
            ground_truth_locations=gt_locations,
            explored_files=explored_files,
            total_tool_calls=total_tool_calls,
            successful_tool_calls=successful_tool_calls,
        )
    else:
        # Navigator reward — uses confirmed_suspicious from extra_info
        confirmed = extra_info.get("confirmed_suspicious", [])
        if isinstance(confirmed, list) and confirmed and isinstance(confirmed[0], dict):
            predicted = [item["location"] for item in confirmed if "location" in item]
        elif isinstance(confirmed, list):
            predicted = [str(loc) for loc in confirmed]
        else:
            predicted = parse_predicted_locations(solution_str)

        result = compute_navigator_reward(
            predicted_locations=predicted,
            ground_truth_locations=gt_locations,
            total_tool_calls=total_tool_calls,
            successful_tool_calls=successful_tool_calls,
        )

    return result
