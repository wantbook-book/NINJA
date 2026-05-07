from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import os.path as osp
import re
import tempfile
from collections import Counter, defaultdict
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
TOOL_CALL_MARKER_RE = re.compile(r"<tool_call>", re.IGNORECASE)
TOOL_RESULT_PREFIX_RE = re.compile(r"by\s+'([^']+)'\s+with arguments:\s*", re.IGNORECASE)
RETRIEVAL_TOOL_PREFIX_RE = re.compile(r"Retrieval tool:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.IGNORECASE)
RECEIVED_TOOL_NAME_RE = re.compile(r"Received tool name:\s*(.+)")
RECEIVED_ARGUMENTS_RE = re.compile(r"Received arguments:\s*(.+)")


def _json_dumps_sorted_safe(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(payload)


def _json_safe_value(payload: Any) -> Any:
    try:
        json.dumps(payload, ensure_ascii=False)
        return payload
    except Exception:
        return str(payload)


def _parse_serialized_value(payload: Any) -> Any:
    if not isinstance(payload, str):
        return payload
    text = payload.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return payload


def _extract_balanced_json(text: str, start_index: int) -> str | None:
    index = int(start_index)
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] not in "{[":
        return None

    stack: list[str] = []
    in_string = False
    escape_next = False
    quote_char = ""
    for position in range(index, len(text)):
        char = text[position]
        if in_string:
            if escape_next:
                escape_next = False
            elif char == "\\":
                escape_next = True
            elif char == quote_char:
                in_string = False
            continue

        if char in {'"', "'"}:
            in_string = True
            quote_char = char
            continue
        if char in "{[":
            stack.append("}" if char == "{" else "]")
            continue
        if char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return text[index: position + 1]
    return None


def _tool_call_record(
    *,
    tool_name: str,
    arguments: Any,
    source: str,
    failed: bool = False,
    invalid: bool = False,
) -> dict[str, Any]:
    normalized_tool_name = str(tool_name or "")
    arguments_text = _json_dumps_sorted_safe(arguments)
    return {
        "tool_name": normalized_tool_name,
        "arguments": _json_safe_value(arguments),
        "arguments_text": arguments_text,
        "tool_call_key": f"{normalized_tool_name}:{arguments_text}",
        "source": str(source or "unknown"),
        "failed": bool(failed),
        "invalid": bool(invalid),
    }


def _tool_call_stats(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    normalized_records = [record for record in records if isinstance(record, dict)]
    key_counter = Counter(str(record.get("tool_call_key", "") or "") for record in normalized_records)
    name_counter = Counter(str(record.get("tool_name", "") or "") for record in normalized_records)
    unique_by_tool_name: dict[str, set[str]] = defaultdict(set)
    exemplar_by_key: dict[str, dict[str, Any]] = {}
    for record in normalized_records:
        tool_name = str(record.get("tool_name", "") or "")
        key = str(record.get("tool_call_key", "") or "")
        unique_by_tool_name[tool_name].add(key)
        exemplar_by_key.setdefault(key, record)

    total_calls = len(normalized_records)
    distinct_calls = len(key_counter)
    repeated_calls = total_calls - distinct_calls
    repeated_details = []
    for key, count in sorted(key_counter.items(), key=lambda item: (-item[1], item[0])):
        if count <= 1:
            continue
        exemplar = exemplar_by_key.get(key, {})
        repeated_details.append(
            {
                "tool_name": exemplar.get("tool_name", ""),
                "arguments": exemplar.get("arguments"),
                "arguments_text": exemplar.get("arguments_text", ""),
                "count": count,
            }
        )

    return {
        "total_tool_calls": total_calls,
        "distinct_tool_calls": distinct_calls,
        "repeated_tool_calls": repeated_calls,
        "repeat_rate": (float(repeated_calls) / float(total_calls)) if total_calls else 0.0,
        "invalid_tool_calls": sum(1 for record in normalized_records if record.get("invalid")),
        "failed_tool_calls": sum(1 for record in normalized_records if record.get("failed")),
        "tool_name_counts": dict(sorted(name_counter.items())),
        "distinct_tool_calls_by_tool_name": {
            tool_name: len(keys) for tool_name, keys in sorted(unique_by_tool_name.items())
        },
        "repeated_tool_call_details": repeated_details,
    }


def _record_from_name_and_arguments(
    tool_name: Any,
    arguments: Any,
    *,
    source: str,
    failed: bool = False,
    invalid: bool = False,
) -> dict[str, Any] | None:
    normalized_tool_name = str(tool_name or "").strip()
    if not normalized_tool_name:
        return None
    return _tool_call_record(
        tool_name=normalized_tool_name,
        arguments=_parse_serialized_value(arguments),
        source=source,
        failed=failed,
        invalid=invalid,
    )


def _record_from_tool_call_payload(payload: Any, *, source: str) -> dict[str, Any] | None:
    payload = _parse_serialized_value(payload)
    if not isinstance(payload, dict):
        return None

    function_payload = payload.get("function")
    if isinstance(function_payload, dict):
        return _record_from_name_and_arguments(
            function_payload.get("name"),
            function_payload.get("arguments", {}),
            source=source,
        )

    return _record_from_name_and_arguments(
        payload.get("name") or payload.get("tool_name"),
        payload.get("arguments", {}),
        source=source,
    )


def _dedupe_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = (
            str(record.get("source", "") or ""),
            str(record.get("tool_name", "") or ""),
            str(record.get("arguments_text", "") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _extract_assistant_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    structured_tool_calls = message.get("tool_calls")
    if isinstance(structured_tool_calls, list):
        for tool_call in structured_tool_calls:
            record = _record_from_tool_call_payload(tool_call, source="assistant_tool_call")
            if record is not None:
                records.append(record)

    function_call = message.get("function_call")
    if isinstance(function_call, dict):
        record = _record_from_tool_call_payload({"function": function_call}, source="assistant_tool_call")
        if record is not None:
            records.append(record)

    content = str(message.get("content", "") or "")
    for block in TOOL_CALL_BLOCK_RE.findall(content):
        record = _record_from_tool_call_payload(block, source="assistant_tool_call")
        if record is not None:
            records.append(record)
    for match in TOOL_CALL_MARKER_RE.finditer(content):
        json_text = _extract_balanced_json(content, match.end())
        if json_text is None:
            continue
        record = _record_from_tool_call_payload(json_text, source="assistant_tool_call")
        if record is not None:
            records.append(record)
    return _dedupe_records(records)


def _extract_tool_result_records(message: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    content = str(message.get("content", "") or "")

    for match in TOOL_RESULT_PREFIX_RE.finditer(content):
        json_text = _extract_balanced_json(content, match.end())
        if json_text is None:
            continue
        record = _record_from_name_and_arguments(
            match.group(1),
            json_text,
            source="tool_result",
            failed='"error"' in json_text or "'error'" in json_text,
        )
        if record is not None:
            records.append(record)

    for match in RETRIEVAL_TOOL_PREFIX_RE.finditer(content):
        json_text = _extract_balanced_json(content, match.end())
        if json_text is None:
            continue
        record = _record_from_name_and_arguments(match.group(1), json_text, source="proactive_node_code_fetch")
        if record is not None:
            records.append(record)

    if "Wrong tool call format" in content:
        tool_name_match = RECEIVED_TOOL_NAME_RE.search(content)
        arguments_match = RECEIVED_ARGUMENTS_RE.search(content)
        if tool_name_match:
            tool_name = _parse_serialized_value(tool_name_match.group(1).strip())
            arguments = arguments_match.group(1).strip() if arguments_match else {}
            record = _record_from_name_and_arguments(
                tool_name,
                arguments,
                source="tool_result",
                failed=True,
                invalid=True,
            )
            if record is not None:
                records.append(record)

    return _dedupe_records(records)


def _is_tool_feedback_message(message: Any) -> bool:
    return isinstance(message, dict) and str(message.get("_message_kind", "") or "") == "tool_result"


def _remove_matching_pending(pending_records: list[dict[str, Any]], result_records: list[dict[str, Any]]) -> None:
    for result_record in result_records:
        result_key = str(result_record.get("tool_call_key", "") or "")
        for index, pending_record in enumerate(pending_records):
            if str(pending_record.get("tool_call_key", "") or "") == result_key:
                pending_records.pop(index)
                break


def collect_tool_call_records_from_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []

    records: list[dict[str, Any]] = []
    pending_assistant_records: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "") or "")
        if role == "assistant":
            records.extend(pending_assistant_records)
            pending_assistant_records = _extract_assistant_tool_calls(message)
            continue

        is_tool_result = role == "tool" or _is_tool_feedback_message(message)
        if not is_tool_result:
            continue
        result_records = _extract_tool_result_records(message)
        if not result_records:
            continue
        records.extend(result_records)
        _remove_matching_pending(pending_assistant_records, result_records)

    records.extend(pending_assistant_records)
    return records


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


def _collect_entry_records(
    entry: dict[str, Any],
    input_path: str,
    *,
    include_component_details: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    main_records = collect_tool_call_records_from_messages(entry.get("messages"))
    detail_records: list[dict[str, Any]] = []
    detail_paths = _component_detail_paths(entry, input_path) if include_component_details else []
    for detail_path in detail_paths:
        try:
            with open(detail_path, "r", encoding="utf-8") as f:
                detail_payload = json.load(f)
        except Exception:
            continue
        detail_records.extend(collect_tool_call_records_from_messages(detail_payload.get("messages")))

    return (
        [*main_records, *detail_records],
        {
            "main_tool_call_records": len(main_records),
            "component_detail_tool_call_records": len(detail_records),
            "component_detail_files": len(detail_paths),
        },
    )


def _default_output_path(input_path: str) -> str:
    base, ext = osp.splitext(input_path)
    if ext:
        return f"{base}_tool_call_stats{ext}"
    return f"{input_path}_tool_call_stats.jsonl"


def _default_summary_path(output_path: str) -> str:
    return osp.join(osp.dirname(osp.abspath(output_path)) or ".", "tool_call_stats_summary.json")


def _write_jsonl_atomic(output_path: str, rows: list[dict[str, Any]], *, input_path: str) -> None:
    output_dir = osp.dirname(osp.abspath(output_path)) or "."
    os.makedirs(output_dir, exist_ok=True)
    if osp.abspath(output_path) == osp.abspath(input_path):
        fd, tmp_path = tempfile.mkstemp(prefix=".tool_call_stats.", suffix=".jsonl", dir=output_dir)
        os.close(fd)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(tmp_path, output_path)
        finally:
            if osp.exists(tmp_path):
                os.unlink(tmp_path)
        return

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize_stats(stats_by_entry: list[dict[str, Any]]) -> dict[str, Any]:
    total_instances = len(stats_by_entry)
    total_tool_calls = sum(int(stats.get("total_tool_calls", 0) or 0) for stats in stats_by_entry)
    distinct_tool_calls = sum(int(stats.get("distinct_tool_calls", 0) or 0) for stats in stats_by_entry)
    repeated_tool_calls = sum(int(stats.get("repeated_tool_calls", 0) or 0) for stats in stats_by_entry)
    invalid_tool_calls = sum(int(stats.get("invalid_tool_calls", 0) or 0) for stats in stats_by_entry)
    failed_tool_calls = sum(int(stats.get("failed_tool_calls", 0) or 0) for stats in stats_by_entry)
    tool_name_counts: Counter[str] = Counter()
    for stats in stats_by_entry:
        tool_name_counts.update(stats.get("tool_name_counts", {}) or {})
    return {
        "total_instances": total_instances,
        "total_tool_calls": total_tool_calls,
        "total_distinct_tool_calls_per_instance": distinct_tool_calls,
        "total_repeated_tool_calls": repeated_tool_calls,
        "overall_repeat_rate": (float(repeated_tool_calls) / float(total_tool_calls)) if total_tool_calls else 0.0,
        "avg_tool_calls_per_instance": (float(total_tool_calls) / float(total_instances)) if total_instances else 0.0,
        "avg_distinct_tool_calls_per_instance": (
            float(distinct_tool_calls) / float(total_instances)
        ) if total_instances else 0.0,
        "invalid_tool_calls": invalid_tool_calls,
        "failed_tool_calls": failed_tool_calls,
        "tool_name_counts": dict(sorted(tool_name_counts.items())),
    }


def backfill_file(
    input_path: str,
    output_path: str,
    *,
    summary_path: str | None,
    include_component_details: bool,
    preserve_existing: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    stats_by_entry: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Backfilling tool call stats", unit="lines"):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if preserve_existing and isinstance(entry.get("tool_call_stats"), dict):
                stats = entry["tool_call_stats"]
                source = entry.get("tool_call_stats_source", {})
                if not isinstance(source, dict):
                    source = {}
                source.setdefault("preserved_existing", 1)
                entry["tool_call_stats_source"] = source
            else:
                records, source = _collect_entry_records(
                    entry,
                    input_path,
                    include_component_details=include_component_details,
                )
                stats = _tool_call_stats(records)
                entry["tool_call_stats"] = stats
                entry["tool_call_stats_source"] = source
            rows.append(entry)
            stats_by_entry.append(stats)

    summary = _summarize_stats(stats_by_entry)
    _write_jsonl_atomic(output_path, rows, input_path=input_path)
    if summary_path:
        summary_dir = osp.dirname(osp.abspath(summary_path)) or "."
        os.makedirs(summary_dir, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill per-trajectory tool call stats into inference JSONL.")
    parser.add_argument("--input", required=True, help="Path to old inference trajs.jsonl")
    parser.add_argument("--output", default=None, help="Output JSONL path. Defaults to *_tool_call_stats.jsonl")
    parser.add_argument(
        "--summary",
        default=None,
        help="Summary JSON path. Defaults to tool_call_stats_summary.json next to output.",
    )
    parser.add_argument(
        "--no_component_details",
        action="store_true",
        help="Only scan top-level trajectory messages, ignoring summary/components/<instance_id>/*.json.",
    )
    parser.add_argument(
        "--preserve_existing",
        action="store_true",
        help="Do not recompute entries that already contain tool_call_stats.",
    )
    args = parser.parse_args()

    output_path = args.output or _default_output_path(args.input)
    summary_path = args.summary if args.summary is not None else _default_summary_path(output_path)
    summary = backfill_file(
        args.input,
        output_path,
        summary_path=summary_path,
        include_component_details=not args.no_component_details,
        preserve_existing=bool(args.preserve_existing),
    )
    print(f"Saved tool-call stats JSONL to: {output_path}")
    if summary_path:
        print(f"Saved summary to: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
