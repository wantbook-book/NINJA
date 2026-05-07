from __future__ import annotations
from collections import Counter
from copy import deepcopy
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import argparse
import glob
import json
import os

from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
from evaluation.utils import (
    _extract_trace_locs,
    average_precision,
    construct_pred_func,
    extract_locs,
    extract_predicted_methods,
    locagent_acc_at_k,
    ndcg_at_k,
    parse_gt_methods,
    recall_at_k,
    reciprocal_rank,
    top_k_accuracy,
)

# from LocAgent.evaluation.eval_metric import acc_at_k, ndcg_at_k, recall_at_k
# from LocAgent.util.process_output import get_loc_results_from_raw_outputs

USAGE_KEYS = (
    "total_tokens",
    "tool_tokens",
    "assistant_tokens",
    "total_turns",
    "tool_turns",
    "assistant_turns",
)


def _is_tool_feedback_message(message: Any) -> bool:
    return isinstance(message, dict) and str(message.get("_message_kind", "") or "") == "tool_result"

def _augment_token_counts(
    messages: list[dict[str, Any]],
) -> tuple[int, int, int, int, int]:
    counts = []
    for msg in messages:
        token_count = 0
        if isinstance(msg, dict):
            try:
                token_count = int(msg.get("token_count", 0) or 0)
            except (TypeError, ValueError):
                token_count = 0
        counts.append(token_count)
    total_tokens = sum(counts)
    total_tool_tokens = sum(
        count
        for msg, count in zip(messages, counts)
        if isinstance(msg, dict) and (msg.get("role") == "tool" or _is_tool_feedback_message(msg))
    )
    total_assistant_tokens = sum(
        count for msg, count in zip(messages, counts) if isinstance(msg, dict) and msg.get("role") == "assistant"
    )
    total_turns = sum(1 for msg in messages if isinstance(msg, dict))
    total_tool_turns = sum(
        1
        for msg in messages
        if isinstance(msg, dict) and (msg.get("role") == "tool" or _is_tool_feedback_message(msg))
    )
    total_assistant_turns = sum(1 for msg in messages if isinstance(msg, dict) and msg.get("role") == "assistant")
    return (
        total_tokens,
        total_tool_tokens,
        total_assistant_tokens,
        total_turns,
        total_tool_turns,
        total_assistant_turns,
    )


def _empty_usage_record() -> dict[str, int]:
    return {
        "total_tokens": 0,
        "tool_tokens": 0,
        "assistant_tokens": 0,
        "total_turns": 0,
        "tool_turns": 0,
        "assistant_turns": 0,
        "tool_execute_count": 0,
        "tool_error_count": 0,
    }


def _usage_record_from_tuple(
    usage_tuple: tuple[int, int, int, int, int, int],
) -> dict[str, int]:
    (
        total_tokens,
        tool_tokens,
        assistant_tokens,
        total_turns,
        tool_turns,
        assistant_turns,
    ) = usage_tuple
    payload = _empty_usage_record()
    payload.update(
        {
            "total_tokens": int(total_tokens),
            "tool_tokens": int(tool_tokens),
            "assistant_tokens": int(assistant_tokens),
            "total_turns": int(total_turns),
            "tool_turns": int(tool_turns),
            "assistant_turns": int(assistant_turns),
            "tool_execute_count": int(tool_turns),
        }
    )
    return payload


def _sum_usage_records(records: list[dict[str, int]]) -> dict[str, int]:
    total = _empty_usage_record()
    for record in records:
        if not isinstance(record, dict):
            continue
        for key in total:
            try:
                total[key] += int(record.get(key, 0) or 0)
            except (TypeError, ValueError):
                continue
    return total


def _resolve_summary_dir(entry: dict[str, Any], input_path: str) -> str:
    component_summary_path = str(entry.get("component_summary_path", "") or "")
    if component_summary_path:
        return osp.dirname(component_summary_path)

    input_dir = osp.dirname(osp.abspath(input_path))
    candidate_dirs = []
    if osp.basename(input_dir) == "traj":
        candidate_dirs.append(osp.join(osp.dirname(input_dir), "summary"))
    candidate_dirs.append(osp.join(input_dir, "summary"))
    candidate_dirs.append(osp.join(osp.dirname(input_dir), "summary"))
    for candidate_dir in candidate_dirs:
        if candidate_dir and osp.isdir(candidate_dir):
            return candidate_dir
    return candidate_dirs[0] if candidate_dirs else ""


def _component_detail_paths(entry: dict[str, Any], input_path: str) -> list[str]:
    instance_id = str(entry.get("instance_id", "") or "")
    if not instance_id:
        return []
    summary_dir = _resolve_summary_dir(entry, input_path)
    if not summary_dir:
        return []
    component_detail_dir = osp.join(summary_dir, "components", instance_id)
    if not osp.isdir(component_detail_dir):
        return []
    return sorted(glob.glob(osp.join(component_detail_dir, "*.json")))


def _load_component_mode_usage(entry: dict[str, Any], input_path: str) -> dict[str, int] | None:
    detail_paths = _component_detail_paths(entry, input_path)
    if not detail_paths:
        return None
    usage_records: list[dict[str, int]] = []
    for detail_path in detail_paths:
        with open(detail_path, "r", encoding="utf-8") as f:
            detail_payload = json.load(f)
        usage_payload = detail_payload.get("usage")
        if isinstance(usage_payload, dict) and any(key in usage_payload for key in USAGE_KEYS):
            usage_records.append(
                {
                    **_empty_usage_record(),
                    **{
                        key: int(usage_payload.get(key, 0) or 0)
                        for key in _empty_usage_record()
                    },
                }
            )
            continue

        detail_messages = detail_payload.get("messages")
        if not isinstance(detail_messages, list):
            detail_messages = []
        summary_trace = detail_payload.get("component_summary_trace", {})
        summary_messages: list[dict[str, Any]] = []
        merge_messages: list[dict[str, Any]] = []
        if isinstance(summary_trace, dict):
            if isinstance(summary_trace.get("summary_messages"), list):
                summary_messages = summary_trace.get("summary_messages", [])
            if isinstance(summary_trace.get("merge_messages"), list):
                merge_messages = summary_trace.get("merge_messages", [])
        usage_record = _usage_record_from_tuple(
            _augment_token_counts([*detail_messages, *summary_messages, *merge_messages])
        )
        try:
            usage_record["tool_execute_count"] = int(
                detail_payload.get("tool_execute_count", usage_record["tool_turns"]) or 0
            )
        except (TypeError, ValueError):
            pass
        try:
            usage_record["tool_error_count"] = int(detail_payload.get("tool_error_count", 0) or 0)
        except (TypeError, ValueError):
            pass
        usage_records.append(usage_record)

    return _sum_usage_records(usage_records)


def _detail_path_for_entry(
    entry: dict[str, Any],
    input_path: str,
    *,
    field_name: str,
    summary_subdir: str,
) -> str:
    explicit_path = str(entry.get(field_name, "") or "")
    if explicit_path and osp.isfile(explicit_path):
        return explicit_path
    instance_id = str(entry.get("instance_id", "") or "")
    if not instance_id:
        return ""
    summary_dir = _resolve_summary_dir(entry, input_path)
    if not summary_dir:
        return ""
    candidate_path = osp.join(summary_dir, summary_subdir, f"{instance_id}.json")
    return candidate_path if osp.isfile(candidate_path) else ""


def _usage_from_detail_file(path: str) -> dict[str, int] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        detail_payload = json.load(f)
    usage_payload = detail_payload.get("usage")
    if isinstance(usage_payload, dict) and any(key in usage_payload for key in USAGE_KEYS):
        return {
            **_empty_usage_record(),
            **{
                key: int(usage_payload.get(key, 0) or 0)
                for key in _empty_usage_record()
            },
        }
    detail_messages = detail_payload.get("messages")
    if isinstance(detail_messages, list):
        return _usage_record_from_tuple(_augment_token_counts(detail_messages))
    return None


def _load_name_query_usage(entry: dict[str, Any], input_path: str) -> dict[str, int] | None:
    return _usage_from_detail_file(
        _detail_path_for_entry(
            entry,
            input_path,
            field_name="name_query_detail_path",
            summary_subdir="name_query",
        )
    )


def _load_default_fallback_usage(entry: dict[str, Any], input_path: str) -> dict[str, int] | None:
    fallback_usage = _usage_from_detail_file(
        _detail_path_for_entry(
            entry,
            input_path,
            field_name="default_fallback_detail_path",
            summary_subdir="default_fallback",
        )
    )
    if fallback_usage is not None:
        return fallback_usage
    default_fallback_messages = entry.get("default_fallback_messages")
    if isinstance(default_fallback_messages, list):
        return _usage_record_from_tuple(_augment_token_counts(default_fallback_messages))
    return None


def _default_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    if ext:
        return f"{base}_eval{ext}"
    return f"{input_path}_eval.jsonl"


def _resolve_output_base_dir(input_path: str) -> str:
    """Derive the output base directory from the input JSONL path.

    The input file is typically ``<output_dir>/traj/trajs.jsonl``.
    ``sub_agent_detail_path`` values are stored relative to ``<output_dir>``.
    """
    input_dir = osp.dirname(osp.abspath(input_path))
    if osp.basename(input_dir) == "traj":
        return osp.dirname(input_dir)
    return input_dir


def _load_sub_agent_detail(sa_summary: dict[str, Any], base_dir: str) -> dict[str, Any]:
    """Load full sub-agent data from its detail JSON file.

    Falls back to the inline summary when the file is unavailable.
    """
    detail_path = str(sa_summary.get("sub_agent_detail_path", "") or "")
    if not detail_path:
        return sa_summary
    abs_path = osp.join(base_dir, detail_path)
    if not osp.isfile(abs_path):
        return sa_summary
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return sa_summary


def _build_entry_usage(
    usage_record: dict[str, int],
    *,
    usage_source: str,
) -> dict[str, int]:
    payload = {
        "total_tokens": int(usage_record.get("total_tokens", 0) or 0),
        "tool_tokens": int(usage_record.get("tool_tokens", 0) or 0),
        "assistant_tokens": int(usage_record.get("assistant_tokens", 0) or 0),
        "total_turns": int(usage_record.get("total_turns", 0) or 0),
        "tool_turns": int(usage_record.get("tool_turns", 0) or 0),
        "assistant_turns": int(usage_record.get("assistant_turns", 0) or 0),
        "tool_execute_count": int(usage_record.get("tool_execute_count", 0) or 0),
        "tool_error_count": int(usage_record.get("tool_error_count", 0) or 0),
        "usage_source": usage_source,
    }
    return payload


def _resolve_entry_usage(
    entry: dict[str, Any],
    messages: list[dict[str, Any]],
    input_path: str,
) -> tuple[dict[str, int], str]:
    search_mode = str(entry.get("search_mode", "") or "")
    if search_mode in {"component_graph", "bfs_component_graph"}:
        component_mode_usage = _load_component_mode_usage(entry, input_path)
        if component_mode_usage is not None:
            return component_mode_usage, "component_details"
    if search_mode == "name_guided_component_graph":
        component_mode_usage = _load_component_mode_usage(entry, input_path)
        main_usage = _usage_record_from_tuple(_augment_token_counts(messages))
        usage_records = []
        usage_sources = []
        name_query_usage = _load_name_query_usage(entry, input_path)
        if name_query_usage is not None:
            usage_records.append(name_query_usage)
            usage_sources.append("name_query_detail")
        if component_mode_usage is not None:
            usage_records.append(component_mode_usage)
            usage_sources.append("component_details")
        usage_records.append(main_usage)
        usage_sources.append("main_messages")
        default_fallback_usage = _load_default_fallback_usage(entry, input_path)
        if default_fallback_usage is not None:
            usage_records.append(default_fallback_usage)
            usage_sources.append("default_fallback_detail")
        if len(usage_records) > 1:
            return _sum_usage_records(usage_records), "_plus_".join(usage_sources)
    return _usage_record_from_tuple(_augment_token_counts(messages)), "main_messages"


def _usage_fail_weight(usage_record: dict[str, int]) -> int:
    try:
        tool_execute_count = int(usage_record.get("tool_execute_count", 0) or 0)
    except (TypeError, ValueError):
        tool_execute_count = 0
    if tool_execute_count > 0:
        return tool_execute_count
    try:
        return int(usage_record.get("tool_turns", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _usage_fail_numerator(usage_record: dict[str, int], entry_fail_ratio: float) -> float:
    fail_weight = _usage_fail_weight(usage_record)
    if fail_weight <= 0:
        return 0.0
    try:
        tool_error_count = int(usage_record.get("tool_error_count", 0) or 0)
    except (TypeError, ValueError):
        tool_error_count = 0
    if tool_error_count > 0:
        return float(tool_error_count)
    return float(entry_fail_ratio) * float(fail_weight)


def _build_rank_metrics(gt: set[str], preds: list[str], ndcg_k: int = 5) -> dict[str, Any]:
    return {
        "acc": {
            "top1": float(top_k_accuracy(gt, preds, 1)),
            "top3": float(top_k_accuracy(gt, preds, 3)),
            "top5": float(top_k_accuracy(gt, preds, 5)),
        },
        "locagent_acc": {
            "top1": locagent_acc_at_k(gt, preds, 1),
            "top3": locagent_acc_at_k(gt, preds, 3),
            "top5": locagent_acc_at_k(gt, preds, 5),
        },
        "recall": {
            "top1": recall_at_k(gt, preds, 1),
            "top3": recall_at_k(gt, preds, 3),
            "top5": recall_at_k(gt, preds, 5),
        },
        "f1": {
            "top1": _set_f1_score(gt, preds, 1),
            "top3": _set_f1_score(gt, preds, 3),
            "top5": _set_f1_score(gt, preds, 5),
        },
        "ap": average_precision(gt, preds),
        "rr": reciprocal_rank(gt, preds),
        "ndcg": {
            f"top{ndcg_k}": ndcg_at_k(gt, preds, ndcg_k),
        },
    }


def _set_f1_score(gt: set[str], preds: list[str], k: int | None = None) -> float:
    if k is not None:
        preds = preds[:k]
    pred_set = set(preds)
    tp = len(gt & pred_set)
    if tp == 0:
        return 0.0
    return (2.0 * tp) / (len(gt) + len(pred_set))


def _build_entry_metrics(
    gt_files: set[str],
    pred_files: list[str],
    gt_methods: set[str],
    pred_methods: list[str],
    usage: dict[str, int],
    fail_ratio: float,
    sub_agent_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "file_level": {
            **_build_rank_metrics(gt_files, pred_files),
        },
        "function_level": {
            **_build_rank_metrics(gt_methods, pred_methods),
        },
        "usage": usage,
        "fail_ratio": fail_ratio,
    }
    if sub_agent_stats:
        result["sub_agent_stats"] = sub_agent_stats
    return result


def _safe_average(total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return total / count


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _resolve_search_execution_mode(entry: dict[str, Any]) -> str:
    search_execution_mode = str(entry.get("search_execution_mode", "") or "").strip()
    if search_execution_mode:
        return search_execution_mode
    search_mode = str(entry.get("search_mode", "") or "").strip()
    if search_mode:
        return f"legacy_{search_mode}"
    return "unknown"


def evaluate_file(
    input_path: str,
    output_path: str,
    k_values: list[int] | None,
    main_agent_token_budget: int = 0,
    sub_agent_token_budget: int = 0,
) -> None:
    sample_entry = None

    # File-level TOP-N statistics
    top1_file_correct = 0
    top3_file_correct = 0
    top5_file_correct = 0
    locagent_acc1_file = 0.0
    locagent_acc3_file = 0.0
    locagent_acc5_file = 0.0

    recall1_file = 0
    recall3_file = 0
    recall5_file = 0

    # Function-level TOP-N statistics
    top1_func_correct = 0
    top3_func_correct = 0
    top5_func_correct = 0
    locagent_acc1_func = 0.0
    locagent_acc3_func = 0.0
    locagent_acc5_func = 0.0

    recall1_func = 0
    recall3_func = 0
    recall5_func = 0

    # Initialize MAP and MRR accumulators
    file_AP_sum = 0.0
    file_RR_sum = 0.0
    func_AP_sum = 0.0
    func_RR_sum = 0.0
    file_ndcg_sum = 0.0
    func_ndcg_sum = 0.0
    file_f1_1_sum = 0.0
    file_f1_3_sum = 0.0
    file_f1_5_sum = 0.0
    func_f1_1_sum = 0.0
    func_f1_3_sum = 0.0
    func_f1_5_sum = 0.0
    empty_count = 0
    total_instances = 0
    total_tokens = 0
    total_tool_tokens = 0
    total_assistant_tokens = 0
    total_assistant_turns = 0
    total_tool_turns = 0
    total_turns = 0
    total_tool_execute_count = 0
    total_tool_error_count = 0
    total_tool_calls = 0
    total_distinct_tool_calls = 0
    total_repeated_tool_calls = 0
    total_tool_repeat_rate = 0.0
    total_tool_failed_rate = 0.0
    total_fail_numerator = 0.0
    total_fail_weight = 0
    total_inference_time = 0.0
    entries_with_inference_time = 0
    total_model_call_time = 0.0
    entries_with_model_call_time = 0
    # Token-budget exhaustion counters
    main_budget_exhausted_count = 0
    sub_budget_exhausted_count = 0
    sub_agent_total_for_budget = 0
    search_execution_mode_counts: Counter[str] = Counter()
    # Sub-agent statistics accumulators
    total_sub_agent_count = 0
    total_sub_agent_tokens = 0
    total_sub_agent_tool_calls = 0
    total_sub_agent_failed_tool_calls = 0
    total_sub_agent_repeated_tool_calls = 0
    entries_with_sub_agents = 0
    output_base_dir = _resolve_output_base_dir(input_path)
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc="Evaluating", unit="lines"):
            line = line.strip()
            if not line:
                continue
            
            entry = json.loads(line)
            search_execution_mode = _resolve_search_execution_mode(entry)
            entry["search_execution_mode"] = search_execution_mode
            search_execution_mode_counts[search_execution_mode] += 1
            gt_files, gt_methods = parse_gt_methods(entry.get("ground_truth"))
            messages = entry.get("messages")
            if not isinstance(messages, list):
                messages = []
                entry["messages"] = messages
            usage_record, usage_source = _resolve_entry_usage(entry, messages, input_path)
            entry_total_tokens = int(usage_record.get("total_tokens", 0) or 0)
            if main_agent_token_budget > 0 and entry_total_tokens >= main_agent_token_budget:
                main_budget_exhausted_count += 1
            entry_tool_tokens = int(usage_record.get("tool_tokens", 0) or 0)
            entry_assistant_tokens = int(usage_record.get("assistant_tokens", 0) or 0)
            entry_total_turns = int(usage_record.get("total_turns", 0) or 0)
            entry_tool_turns = int(usage_record.get("tool_turns", 0) or 0)
            entry_assistant_turns = int(usage_record.get("assistant_turns", 0) or 0)
            total_tokens += entry_total_tokens
            total_tool_tokens += entry_tool_tokens
            total_assistant_tokens += entry_assistant_tokens
            total_turns += entry_total_turns
            total_tool_turns += entry_tool_turns
            total_assistant_turns += entry_assistant_turns
            total_tool_execute_count += int(usage_record.get("tool_execute_count", 0) or 0)
            total_tool_error_count += int(usage_record.get("tool_error_count", 0) or 0)
            tool_call_stats = entry.get("tool_call_stats")
            if isinstance(tool_call_stats, dict):
                entry_total_tool_calls = _safe_int(tool_call_stats.get("total_tool_calls"))
                entry_repeated_tool_calls = _safe_int(tool_call_stats.get("repeated_tool_calls"))
                entry_repeat_rate = _safe_float(tool_call_stats.get("repeat_rate"))
                if "repeat_rate" not in tool_call_stats and entry_total_tool_calls > 0:
                    entry_repeat_rate = float(entry_repeated_tool_calls) / float(entry_total_tool_calls)
                total_tool_calls += entry_total_tool_calls
                total_distinct_tool_calls += _safe_int(tool_call_stats.get("distinct_tool_calls"))
                total_repeated_tool_calls += entry_repeated_tool_calls
                total_tool_repeat_rate += entry_repeat_rate
                if entry_total_tool_calls > 0:
                    total_tool_failed_rate += (
                        float(_safe_int(tool_call_stats.get("failed_tool_calls")))
                        / float(entry_total_tool_calls)
                    )

            entry_fail_ratio = float(entry.get("fail_ratio", 0.0) or 0.0)
            entry_fail_weight = _usage_fail_weight(usage_record)
            total_fail_numerator += _usage_fail_numerator(usage_record, entry_fail_ratio)
            total_fail_weight += entry_fail_weight

            # ---- Inference time ----
            entry_inference_time = _safe_float(entry.get("inference_time_seconds"))
            if entry_inference_time > 0:
                total_inference_time += entry_inference_time
                entries_with_inference_time += 1

            # ---- Model call time ----
            entry_model_call_time = _safe_float(entry.get("model_call_time_seconds"))
            if entry_model_call_time > 0:
                total_model_call_time += entry_model_call_time
                entries_with_model_call_time += 1

            # ---- Sub-agent statistics ----
            # Prefer extracting from sub_agent_trajectories (detailed), fall
            # back to the pre-aggregated sub_agent_tool_call_stats field.
            sub_agent_trajs = entry.get("sub_agent_trajectories")
            entry_sub_agent_stats: dict[str, Any] = {}
            if isinstance(sub_agent_trajs, list) and len(sub_agent_trajs) > 0:
                entry_sub_agent_count = len(sub_agent_trajs)
                entries_with_sub_agents += 1
                total_sub_agent_count += entry_sub_agent_count
                sa_total_tokens = 0
                sa_total_tool_calls = 0
                sa_total_failed = 0
                sa_total_repeated = 0
                sa_total_distinct = 0
                sa_total_turns = 0
                sa_total_tool_turns = 0
                sa_total_assistant_turns = 0
                per_sub_agent: list[dict[str, Any]] = []
                for sa in sub_agent_trajs:
                    # Load full detail (with messages) from the separate
                    # JSON file when the inline summary lacks messages.
                    sa_full = sa
                    if not isinstance(sa.get("messages"), list) or not sa.get("messages"):
                        sa_full = _load_sub_agent_detail(sa, output_base_dir)
                    sa_msgs = sa_full.get("messages") if isinstance(sa_full.get("messages"), list) else []
                    sa_tokens = sum(
                        _safe_int(m.get("token_count")) for m in sa_msgs if isinstance(m, dict)
                    )
                    if sub_agent_token_budget > 0:
                        sub_agent_total_for_budget += 1
                        if sa_tokens >= sub_agent_token_budget:
                            sub_budget_exhausted_count += 1
                    sa_turns = len(sa_msgs)
                    sa_assistant_turns = sum(1 for m in sa_msgs if isinstance(m, dict) and m.get("role") == "assistant")
                    sa_tool_t = sum(
                        1 for m in sa_msgs
                        if isinstance(m, dict) and (
                            m.get("role") == "tool"
                            or str(m.get("_message_kind", "") or "") == "tool_result"
                        )
                    )
                    sa_stats = sa_full.get("sub_agent_stats") if isinstance(sa_full.get("sub_agent_stats"), dict) else {}
                    sa_tc = _safe_int(sa_stats.get("total_tool_calls"))
                    sa_fc = _safe_int(sa_stats.get("failed_tool_calls"))
                    sa_rc = _safe_int(sa_stats.get("repeated_tool_calls"))
                    sa_dc = _safe_int(sa_stats.get("distinct_tool_calls"))
                    sa_total_tokens += sa_tokens
                    sa_total_tool_calls += sa_tc
                    sa_total_failed += sa_fc
                    sa_total_repeated += sa_rc
                    sa_total_distinct += sa_dc
                    sa_total_turns += sa_turns
                    sa_total_tool_turns += sa_tool_t
                    sa_total_assistant_turns += sa_assistant_turns
                    per_sub_agent.append({
                        "entry_file": str(sa_full.get("entry_file", "") or ""),
                        "tokens": sa_tokens,
                        "messages": len(sa_msgs),
                        "turns": sa_turns,
                        "assistant_turns": sa_assistant_turns,
                        "tool_turns": sa_tool_t,
                        "tool_calls": sa_tc,
                        "failed_tool_calls": sa_fc,
                        "repeated_tool_calls": sa_rc,
                        "distinct_tool_calls": sa_dc,
                        "suspicious_count": len(sa_full.get("suspicious", []) if isinstance(sa_full.get("suspicious"), list) else []),
                        "error": str(sa_full.get("error", "") or ""),
                    })
                total_sub_agent_tokens += sa_total_tokens
                total_sub_agent_tool_calls += sa_total_tool_calls
                total_sub_agent_failed_tool_calls += sa_total_failed
                total_sub_agent_repeated_tool_calls += sa_total_repeated
                entry_sub_agent_stats = {
                    "sub_agent_count": entry_sub_agent_count,
                    "total_sub_agent_tokens": sa_total_tokens,
                    "avg_sub_agent_tokens": sa_total_tokens / entry_sub_agent_count if entry_sub_agent_count > 0 else 0,
                    "total_sub_agent_tool_calls": sa_total_tool_calls,
                    "avg_sub_agent_tool_calls": sa_total_tool_calls / entry_sub_agent_count if entry_sub_agent_count > 0 else 0,
                    "total_sub_agent_failed_tool_calls": sa_total_failed,
                    "total_sub_agent_repeated_tool_calls": sa_total_repeated,
                    "total_sub_agent_distinct_tool_calls": sa_total_distinct,
                    "sub_agent_fail_rate": sa_total_failed / sa_total_tool_calls if sa_total_tool_calls > 0 else 0.0,
                    "sub_agent_repeat_rate": sa_total_repeated / sa_total_tool_calls if sa_total_tool_calls > 0 else 0.0,
                    "total_sub_agent_turns": sa_total_turns,
                    "avg_sub_agent_turns": sa_total_turns / entry_sub_agent_count if entry_sub_agent_count > 0 else 0,
                    "total_sub_agent_tool_turns": sa_total_tool_turns,
                    "total_sub_agent_assistant_turns": sa_total_assistant_turns,
                    "per_sub_agent": per_sub_agent,
                }
            else:
                # Fallback: read from pre-aggregated sub_agent_tool_call_stats
                sa_pre = entry.get("sub_agent_tool_call_stats")
                if isinstance(sa_pre, dict) and _safe_int(sa_pre.get("sub_agent_count")) > 0:
                    sa_count = _safe_int(sa_pre.get("sub_agent_count"))
                    entries_with_sub_agents += 1
                    total_sub_agent_count += sa_count
                    total_sub_agent_tokens += _safe_int(sa_pre.get("total_tokens"))
                    total_sub_agent_tool_calls += _safe_int(sa_pre.get("total_tool_calls"))
                    total_sub_agent_failed_tool_calls += _safe_int(sa_pre.get("failed_tool_calls"))
                    total_sub_agent_repeated_tool_calls += _safe_int(sa_pre.get("repeated_tool_calls"))
                    entry_sub_agent_stats = {
                        "sub_agent_count": sa_count,
                        "total_sub_agent_tokens": _safe_int(sa_pre.get("total_tokens")),
                        "avg_sub_agent_tokens": _safe_float(sa_pre.get("avg_tokens_per_sub_agent")),
                        "total_sub_agent_tool_calls": _safe_int(sa_pre.get("total_tool_calls")),
                        "avg_sub_agent_tool_calls": _safe_float(sa_pre.get("avg_tool_calls_per_sub_agent")),
                        "total_sub_agent_failed_tool_calls": _safe_int(sa_pre.get("failed_tool_calls")),
                        "total_sub_agent_repeated_tool_calls": _safe_int(sa_pre.get("repeated_tool_calls")),
                        "total_sub_agent_distinct_tool_calls": _safe_int(sa_pre.get("distinct_tool_calls")),
                        "sub_agent_fail_rate": _safe_float(sa_pre.get("fail_rate")),
                        "sub_agent_repeat_rate": _safe_float(sa_pre.get("repeat_rate")),
                        "total_sub_agent_turns": _safe_int(sa_pre.get("total_turns")),
                        "avg_sub_agent_turns": _safe_float(sa_pre.get("avg_turns_per_sub_agent")),
                        "total_sub_agent_tool_turns": _safe_int(sa_pre.get("total_tool_turns")),
                        "total_sub_agent_assistant_turns": _safe_int(sa_pre.get("total_assistant_turns")),
                    }
            entry["sub_agent_stats_summary"] = entry_sub_agent_stats

            entry_usage = _build_entry_usage(
                usage_record,
                usage_source=usage_source,
            )

            entry["trace_locs"] = ""
            entry["pred_files"] = []
            entry["pred_methods"] = []
            entry["metrics"] = _build_entry_metrics(
                gt_files,
                [],
                gt_methods,
                [],
                entry_usage,
                entry_fail_ratio,
                sub_agent_stats=entry_sub_agent_stats or None,
            )

            for msg in reversed(messages):
                if msg['role'] == 'assistant':
                    total_instances += 1
                    content = msg['content']
                    trace_locs = _extract_trace_locs(content)
                    entry["trace_locs"] = trace_locs

                    found_related_locs = extract_locs(trace_locs)
                    pred_funcs = construct_pred_func(found_related_locs)
                    pred_methods = extract_predicted_methods(pred_funcs)
                    pred_files = list(found_related_locs.keys())
                    if not pred_files:
                        empty_count += 1

                    entry['pred_files'] = pred_files
                    entry['pred_methods'] = pred_methods

                    # Compute TOP-N accuracy (file level)
                    acc1_file = top_k_accuracy(gt_files, pred_files, 1)
                    acc3_file = top_k_accuracy(gt_files, pred_files, 3)
                    acc5_file = top_k_accuracy(gt_files, pred_files, 5)
                    if acc1_file:
                        top1_file_correct += 1
                    if acc3_file:
                        top3_file_correct += 1
                    if acc5_file:
                        top5_file_correct += 1

                    locagent_acc1_file += locagent_acc_at_k(gt_files, pred_files, 1)
                    locagent_acc3_file += locagent_acc_at_k(gt_files, pred_files, 3)
                    locagent_acc5_file += locagent_acc_at_k(gt_files, pred_files, 5)

                    entry_recall1_file = recall_at_k(gt_files, pred_files, 1)
                    entry_recall3_file = recall_at_k(gt_files, pred_files, 3)
                    entry_recall5_file = recall_at_k(gt_files, pred_files, 5)
                    recall1_file += entry_recall1_file
                    recall3_file += entry_recall3_file
                    recall5_file += entry_recall5_file

                    # Compute TOP-N accuracy (function level, only file functions and class methods)
                    acc1_func = top_k_accuracy(gt_methods, pred_methods, 1)
                    acc3_func = top_k_accuracy(gt_methods, pred_methods, 3)
                    acc5_func = top_k_accuracy(gt_methods, pred_methods, 5)
                    if acc1_func:
                        top1_func_correct += 1
                    if acc3_func:
                        top3_func_correct += 1
                    if acc5_func:
                        top5_func_correct += 1

                    locagent_acc1_func += locagent_acc_at_k(gt_methods, pred_methods, 1)
                    locagent_acc3_func += locagent_acc_at_k(gt_methods, pred_methods, 3)
                    locagent_acc5_func += locagent_acc_at_k(gt_methods, pred_methods, 5)

                    entry_recall1_func = recall_at_k(gt_methods, pred_methods, 1)
                    entry_recall3_func = recall_at_k(gt_methods, pred_methods, 3)
                    entry_recall5_func = recall_at_k(gt_methods, pred_methods, 5)
                    recall1_func += entry_recall1_func
                    recall3_func += entry_recall3_func
                    recall5_func += entry_recall5_func

                    # Compute MAP and MRR
                    ap_file = average_precision(gt_files, pred_files)
                    rr_file = reciprocal_rank(gt_files, pred_files)
                    ap_func = average_precision(gt_methods, pred_methods)
                    rr_func = reciprocal_rank(gt_methods, pred_methods)
                    ndcg_file = ndcg_at_k(gt_files, pred_files, 5)
                    ndcg_func = ndcg_at_k(gt_methods, pred_methods, 5)
                    file_f1_1 = _set_f1_score(gt_files, pred_files, 1)
                    file_f1_3 = _set_f1_score(gt_files, pred_files, 3)
                    file_f1_5 = _set_f1_score(gt_files, pred_files, 5)
                    func_f1_1 = _set_f1_score(gt_methods, pred_methods, 1)
                    func_f1_3 = _set_f1_score(gt_methods, pred_methods, 3)
                    func_f1_5 = _set_f1_score(gt_methods, pred_methods, 5)

                    file_AP_sum += ap_file
                    file_RR_sum += rr_file
                    func_AP_sum += ap_func
                    func_RR_sum += rr_func
                    file_ndcg_sum += ndcg_file
                    func_ndcg_sum += ndcg_func
                    file_f1_1_sum += file_f1_1
                    file_f1_3_sum += file_f1_3
                    file_f1_5_sum += file_f1_5
                    func_f1_1_sum += func_f1_1
                    func_f1_3_sum += func_f1_3
                    func_f1_5_sum += func_f1_5

                    entry["metrics"] = _build_entry_metrics(
                        gt_files,
                        pred_files,
                        gt_methods,
                        pred_methods,
                        entry_usage,
                        entry_fail_ratio,
                        sub_agent_stats=entry_sub_agent_stats or None,
                    )
                    
                    break
            
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if sample_entry is None:
                sample_entry = entry
    
    # Compute TOP-N accuracy percentages

    top1_file_accuracy = _safe_average(top1_file_correct, total_instances) * 100
    top3_file_accuracy = _safe_average(top3_file_correct, total_instances) * 100
    top5_file_accuracy = _safe_average(top5_file_correct, total_instances) * 100
    locagent_acc1_file_avg = _safe_average(locagent_acc1_file, total_instances) * 100
    locagent_acc3_file_avg = _safe_average(locagent_acc3_file, total_instances) * 100
    locagent_acc5_file_avg = _safe_average(locagent_acc5_file, total_instances) * 100

    top1_func_accuracy = _safe_average(top1_func_correct, total_instances) * 100
    top3_func_accuracy = _safe_average(top3_func_correct, total_instances) * 100
    top5_func_accuracy = _safe_average(top5_func_correct, total_instances) * 100
    locagent_acc1_func_avg = _safe_average(locagent_acc1_func, total_instances) * 100
    locagent_acc3_func_avg = _safe_average(locagent_acc3_func, total_instances) * 100
    locagent_acc5_func_avg = _safe_average(locagent_acc5_func, total_instances) * 100

    recall1_file_avg = _safe_average(recall1_file, total_instances) * 100
    recall3_file_avg = _safe_average(recall3_file, total_instances) * 100
    recall5_file_avg = _safe_average(recall5_file, total_instances) * 100

    recall1_func_avg = _safe_average(recall1_func, total_instances) * 100
    recall3_func_avg = _safe_average(recall3_func, total_instances) * 100
    recall5_func_avg = _safe_average(recall5_func, total_instances) * 100

    # Compute MAP and MRR (multiply by 100 to convert to percentages)
    map_file = _safe_average(file_AP_sum, total_instances) * 100
    mrr_file = _safe_average(file_RR_sum, total_instances) * 100
    map_func = _safe_average(func_AP_sum, total_instances) * 100
    mrr_func = _safe_average(func_RR_sum, total_instances) * 100
    ndcg_file_5 = _safe_average(file_ndcg_sum, total_instances) * 100
    ndcg_func_5 = _safe_average(func_ndcg_sum, total_instances) * 100
    f1_file_1 = _safe_average(file_f1_1_sum, total_instances) * 100
    f1_file_3 = _safe_average(file_f1_3_sum, total_instances) * 100
    f1_file_5 = _safe_average(file_f1_5_sum, total_instances) * 100
    f1_func_1 = _safe_average(func_f1_1_sum, total_instances) * 100
    f1_func_3 = _safe_average(func_f1_3_sum, total_instances) * 100
    f1_func_5 = _safe_average(func_f1_5_sum, total_instances) * 100

    empty_percent = _safe_average(empty_count, total_instances) * 100
    if sample_entry is None:
        sample_entry = {}

    usage_stats = {
        "avg_total_tokens": _safe_average(total_tokens, total_instances),
        "avg_tool_tokens": _safe_average(total_tool_tokens, total_instances),
        "avg_assistant_tokens": _safe_average(total_assistant_tokens, total_instances),
        "avg_total_turns": _safe_average(total_turns, total_instances),
        "avg_tool_turns": _safe_average(total_tool_turns, total_instances),
        "avg_assistant_turns": _safe_average(total_assistant_turns, total_instances),
        "avg_tool_execute_count": _safe_average(total_tool_execute_count, total_instances),
        "avg_tool_error_count": _safe_average(total_tool_error_count, total_instances),
        "avg_total_tool_calls": _safe_average(total_tool_calls, total_instances),
        "avg_distinct_tool_calls": _safe_average(total_distinct_tool_calls, total_instances),
        "avg_repeated_tool_calls": _safe_average(total_repeated_tool_calls, total_instances),
        "avg_tool_repeat_rate": _safe_average(total_tool_repeat_rate, total_instances),
        "avg_failed_tool_call_rate": _safe_average(total_tool_failed_rate, total_instances),
    }

    sub_agent_usage_stats: dict[str, Any] = {}
    if entries_with_sub_agents > 0:
        sub_agent_usage_stats = {
            "entries_with_sub_agents": entries_with_sub_agents,
            "avg_sub_agent_count": _safe_average(total_sub_agent_count, entries_with_sub_agents),
            "avg_sub_agent_tokens_per_entry": _safe_average(total_sub_agent_tokens, entries_with_sub_agents),
            "avg_sub_agent_tokens_per_call": _safe_average(total_sub_agent_tokens, total_sub_agent_count),
            "avg_sub_agent_tool_calls_per_entry": _safe_average(total_sub_agent_tool_calls, entries_with_sub_agents),
            "avg_sub_agent_tool_calls_per_call": _safe_average(total_sub_agent_tool_calls, total_sub_agent_count),
            "total_sub_agent_count": total_sub_agent_count,
            "total_sub_agent_tokens": total_sub_agent_tokens,
            "total_sub_agent_tool_calls": total_sub_agent_tool_calls,
            "total_sub_agent_failed_tool_calls": total_sub_agent_failed_tool_calls,
            "total_sub_agent_repeated_tool_calls": total_sub_agent_repeated_tool_calls,
            "sub_agent_fail_rate": _safe_average(total_sub_agent_failed_tool_calls, total_sub_agent_tool_calls),
            "sub_agent_repeat_rate": _safe_average(total_sub_agent_repeated_tool_calls, total_sub_agent_tool_calls),
        }

    metrics = {
        "file_level": {
            "TOP 1": top1_file_accuracy,
            "TOP 3": top3_file_accuracy,
            "TOP 5": top5_file_accuracy,
            "Acc@1": locagent_acc1_file_avg,
            "Acc@3": locagent_acc3_file_avg,
            "Acc@5": locagent_acc5_file_avg,
            "recall@1": recall1_file_avg,
            "recall@3": recall3_file_avg,
            "recall@5": recall5_file_avg,
            "MAP": map_file,
            "MRR": mrr_file,
            "nDCG@5": ndcg_file_5,
            "F1@1": f1_file_1,
            "F1@3": f1_file_3,
            "F1@5": f1_file_5,
            "empty": empty_percent
        },
        "function_level": {
            "TOP 1": top1_func_accuracy,
            "TOP 3": top3_func_accuracy,
            "TOP 5": top5_func_accuracy,
            "Acc@1": locagent_acc1_func_avg,
            "Acc@3": locagent_acc3_func_avg,
            "Acc@5": locagent_acc5_func_avg,
            "recall@1": recall1_func_avg,
            "recall@3": recall3_func_avg,
            "recall@5": recall5_func_avg,
            "MAP": map_func,
            "MRR": mrr_func,
            "nDCG@5": ndcg_func_5,
            "F1@1": f1_func_1,
            "F1@3": f1_func_3,
            "F1@5": f1_func_5,
        },
        "usage": usage_stats,
        "sub_agent_usage": sub_agent_usage_stats,
        "search_execution_mode_counts": dict(sorted(search_execution_mode_counts.items())),
        "fail_ratio": total_fail_numerator / total_fail_weight if total_fail_weight > 0 else 0.0,
        "inference_time": {
            "entries_with_inference_time": entries_with_inference_time,
            "total_inference_time_seconds": total_inference_time,
            "avg_inference_time_seconds": _safe_average(total_inference_time, entries_with_inference_time),
            "entries_with_model_call_time": entries_with_model_call_time,
            "total_model_call_time_seconds": total_model_call_time,
            "avg_model_call_time_seconds": _safe_average(total_model_call_time, entries_with_model_call_time),
        },
        "token_budget": {
            "main_agent_token_budget": main_agent_token_budget,
            "main_agent_total_entries": total_instances,
            "main_budget_exhausted_count": main_budget_exhausted_count,
            "main_budget_exhausted_ratio": _safe_average(main_budget_exhausted_count, total_instances),
            "sub_agent_token_budget": sub_agent_token_budget,
            "sub_agent_total_checked": sub_agent_total_for_budget,
            "sub_budget_exhausted_count": sub_budget_exhausted_count,
            "sub_budget_exhausted_ratio": _safe_average(sub_budget_exhausted_count, sub_agent_total_for_budget),
        },
    }

    output_dir = os.path.dirname(output_path) or "."
    metrics_path = os.path.join(output_dir, "metrics.json")
    sample_path = os.path.join(output_dir, "eval_sample.json")
    with open(metrics_path, "w", encoding="utf-8") as f_metrics:
        json.dump(metrics, f_metrics, ensure_ascii=False, indent=2)
    with open(sample_path, "w", encoding="utf-8") as f_sample:
        json.dump(sample_entry, f_sample, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JSONL results with trace_locs metrics.")
    parser.add_argument("--input", help="Path to input JSONL file")
    parser.add_argument("--output", default=None, help="Path to output JSONL file")
    parser.add_argument(
        "--k",
        default=None,
        help="Comma-separated k values (e.g. 1,3,5). Defaults to GT size per sample.",
    )
    parser.add_argument(
        "--main_agent_token_budget",
        type=int,
        default=0,
        help=(
            "Main agent token budget threshold (0 = disabled). "
            "Entries whose total token count >= this value are counted as budget-exhausted."
        ),
    )
    parser.add_argument(
        "--sub_agent_token_budget",
        type=int,
        default=0,
        help=(
            "Sub-agent token budget threshold (0 = disabled). "
            "Sub-agent trajectories whose total token count >= this value are counted as budget-exhausted."
        ),
    )
    args = parser.parse_args()

    k_values = None
    if args.k:
        k_values = [int(value) for value in args.k.split(",") if value.strip()]

    output_path = args.output or _default_output_path(args.input)
    evaluate_file(
        args.input,
        output_path,
        k_values,
        main_agent_token_budget=args.main_agent_token_budget,
        sub_agent_token_budget=args.sub_agent_token_budget,
    )
    print(f"Saved evaluation output to: {output_path}")


if __name__ == "__main__":
    main()
