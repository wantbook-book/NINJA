"""Check and clean Inspector SFT assistant message format.

This script scans SFT records produced by ``evaluation.build_inspector_sft_data``
and writes every assistant content that does not follow:

``<think>...</think>[<result>...</result>|<trace_locs>...</trace_locs>]``
``<tool_call>...</tool_call>[<tool_call>...</tool_call>...]``,
``<result>...</result>``, or ``<trace_locs>...</trace_locs>``

It also applies conservative format-only repairs and exports records whose
assistant contents are valid after cleaning.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Any

import pandas as pd

from evaluation.sft_data_utils import coerce_messages, load_records, parse_jsonish


ASSISTANT_BLOCK_RE = re.compile(r"<(?P<tag>think|result|trace_locs|tool_call)>(?P<body>.*?)</(?P=tag)>", re.DOTALL)
THINKING_OPEN_RE = re.compile(r"<thinking>", re.IGNORECASE)
THINKING_CLOSE_RE = re.compile(r"</thinking>", re.IGNORECASE)
THINK_WITH_THINKING_CLOSE_RE = re.compile(r"<think>.*?</thinking>", re.DOTALL | re.IGNORECASE)
EXTRA_CLOSING_TOOL_CALL_TAIL_RE = re.compile(r"^(?:</tool_call>\s*)+$", re.IGNORECASE)

TYPE_THINK_CLOSED_BY_THINKING = "think_closed_by_thinking"
TYPE_UNWRAPPED_TEXT_BETWEEN_BLOCKS = "unwrapped_text_between_blocks"
TYPE_MISSING_THINK_BLOCK = "missing_think_block"
TYPE_EXTRA_CLOSING_TOOL_CALL = "extra_closing_tool_call"
TYPE_THINKING_BLOCK = "thinking_block"
TYPE_MULTIPLE_THINK_BLOCKS = "multiple_think_blocks"
TYPE_TRACE_LOCS_WITH_THINK_MISSING_TOOL_CALL = "trace_locs_with_think_missing_tool_call"
TYPE_UNCLASSIFIED = "unclassified"
DEFAULT_EXIT_TOOL_CALL_BODY = '\n{"name": "exit", "arguments": {}}\n'


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _assistant_content_blocks(content: str) -> list[str] | None:
    """Return block tags if content is made only of supported blocks and whitespace."""
    tags: list[str] = []
    cursor = 0
    for match in ASSISTANT_BLOCK_RE.finditer(content):
        if content[cursor : match.start()].strip():
            return None
        tags.append(match.group("tag"))
        cursor = match.end()

    if content[cursor:].strip():
        return None
    return tags


def is_valid_assistant_content(content: str) -> bool:
    """Check for the supported Inspector assistant block layouts."""
    tags = _assistant_content_blocks(content)
    if tags is None:
        return False
    if tags == ["result"] or tags == ["trace_locs"]:
        return True
    if len(tags) < 2:
        return False
    if tags[0] != "think":
        return False

    first_tool_call_index = 2 if len(tags) > 1 and tags[1] in {"result", "trace_locs"} else 1
    if first_tool_call_index >= len(tags):
        return False
    return all(tag == "tool_call" for tag in tags[first_tool_call_index:])


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _render_block(tag: str, body: str) -> str:
    return f"<{tag}>{body}</{tag}>"


def clean_assistant_content_with_types(content: str) -> tuple[str, list[str]]:
    """Apply conservative repairs for common Inspector assistant format errors."""
    if is_valid_assistant_content(content):
        return content, []

    correction_types: list[str] = []
    if THINK_WITH_THINKING_CLOSE_RE.search(content):
        correction_types.append(TYPE_THINK_CLOSED_BY_THINKING)
    if THINKING_OPEN_RE.search(content):
        correction_types.append(TYPE_THINKING_BLOCK)

    normalized = THINKING_OPEN_RE.sub("<think>", content)
    normalized = THINKING_CLOSE_RE.sub("</think>", normalized)

    matches = list(ASSISTANT_BLOCK_RE.finditer(normalized))
    if not matches:
        return normalized.strip(), _dedupe_preserve_order(correction_types) or [TYPE_UNCLASSIFIED]

    leading_text = normalized[: matches[0].start()].strip()
    blocks = [
        {
            "tag": match.group("tag"),
            "body": match.group("body").strip(),
        }
        for match in matches
    ]
    first_tag = matches[0].group("tag")

    if first_tag != "think":
        correction_types.append(TYPE_MISSING_THINK_BLOCK)
        blocks.insert(0, {"tag": "think", "body": leading_text})
    elif leading_text:
        correction_types.append(TYPE_UNWRAPPED_TEXT_BETWEEN_BLOCKS)

    for current_match, next_match in zip(matches, matches[1:], strict=False):
        gap = normalized[current_match.end() : next_match.start()].strip()
        if gap:
            correction_types.append(TYPE_UNWRAPPED_TEXT_BETWEEN_BLOCKS)

    trailing_text = normalized[matches[-1].end() :].strip()
    if trailing_text:
        if EXTRA_CLOSING_TOOL_CALL_TAIL_RE.fullmatch(trailing_text):
            correction_types.append(TYPE_EXTRA_CLOSING_TOOL_CALL)
        else:
            correction_types.append(TYPE_UNWRAPPED_TEXT_BETWEEN_BLOCKS)

    think_blocks = [block for block in blocks if block["tag"] == "think"]
    if len(think_blocks) > 1:
        correction_types.append(TYPE_MULTIPLE_THINK_BLOCKS)
        merged_think_body = "\n".join(block["body"] for block in think_blocks if block["body"])
        merged_blocks: list[dict[str, str]] = []
        first_think_seen = False
        for block in blocks:
            if block["tag"] != "think":
                merged_blocks.append(block)
                continue
            if first_think_seen:
                continue
            merged_blocks.append({"tag": "think", "body": merged_think_body})
            first_think_seen = True
        blocks = merged_blocks

    has_tool_call = any(block["tag"] == "tool_call" for block in blocks)
    if (
        not has_tool_call
        and len(blocks) == 2
        and blocks[0]["tag"] == "think"
        and blocks[1]["tag"] == "trace_locs"
    ):
        correction_types.append(TYPE_TRACE_LOCS_WITH_THINK_MISSING_TOOL_CALL)
        blocks.append({"tag": "tool_call", "body": DEFAULT_EXIT_TOOL_CALL_BODY})

    rendered_blocks = [_render_block(block["tag"], block["body"]) for block in blocks]
    return "\n".join(rendered_blocks), _dedupe_preserve_order(correction_types) or [TYPE_UNCLASSIFIED]


def clean_assistant_content(content: str) -> str:
    corrected_content, _ = clean_assistant_content_with_types(content)
    return corrected_content


def _instance_id(record: dict[str, Any]) -> str:
    extra_info = parse_jsonish(record.get("extra_info", {}))
    if isinstance(extra_info, dict) and extra_info.get("instance_id"):
        return str(extra_info["instance_id"])
    return str(record.get("instance_id", "") or "")


def collect_wrong_assistant_contents(
    records: list[dict[str, Any]],
    *,
    messages_field: str = "messages",
) -> list[dict[str, Any]]:
    wrong_contents: list[dict[str, Any]] = []

    for record in records:
        instance_id = _instance_id(record)
        messages = coerce_messages(record.get(messages_field))
        for message_index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            content = _stringify_content(message.get("content", ""))
            if not is_valid_assistant_content(content):
                corrected_content, correction_types = clean_assistant_content_with_types(content)
                wrong_contents.append(
                    {
                        "instance_id": instance_id,
                        "message_index": message_index,
                        "type": correction_types,
                        "content": content,
                        "corrected_content": corrected_content,
                        "corrected_valid": is_valid_assistant_content(corrected_content),
                    }
                )

    return wrong_contents


def clean_records(
    records: list[dict[str, Any]],
    *,
    messages_field: str = "messages",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cleaned_records: list[dict[str, Any]] = []
    wrong_contents: list[dict[str, Any]] = []
    correction_type_counts: Counter[str] = Counter()
    corrected_contents = 0
    uncorrected_contents = 0

    for record in records:
        instance_id = _instance_id(record)
        messages = coerce_messages(record.get(messages_field))
        cleaned_messages: list[dict[str, Any]] = []
        record_valid_after_cleaning = True

        for message_index, message in enumerate(messages):
            cleaned_message = dict(message)
            if message.get("role") != "assistant":
                cleaned_messages.append(cleaned_message)
                continue

            content = _stringify_content(message.get("content", ""))
            if is_valid_assistant_content(content):
                cleaned_message["content"] = content
                cleaned_messages.append(cleaned_message)
                continue

            corrected_content, correction_types = clean_assistant_content_with_types(content)
            corrected_valid = is_valid_assistant_content(corrected_content)
            correction_type_counts.update(correction_types)
            wrong_contents.append(
                {
                    "instance_id": instance_id,
                    "message_index": message_index,
                    "type": correction_types,
                    "content": content,
                    "corrected_content": corrected_content,
                    "corrected_valid": corrected_valid,
                }
            )
            if corrected_valid:
                corrected_contents += 1
                cleaned_message["content"] = corrected_content
            else:
                uncorrected_contents += 1
                cleaned_message["content"] = corrected_content
                record_valid_after_cleaning = False
            cleaned_messages.append(cleaned_message)

        if record_valid_after_cleaning:
            cleaned_record = dict(record)
            cleaned_record[messages_field] = cleaned_messages
            cleaned_records.append(cleaned_record)

    stats = {
        "total_records": len(records),
        "cleaned_records": len(cleaned_records),
        "wrong_assistant_contents": len(wrong_contents),
        "corrected_assistant_contents": corrected_contents,
        "uncorrected_assistant_contents": uncorrected_contents,
        "correction_type_counts": dict(sorted(correction_type_counts.items())),
    }
    return cleaned_records, wrong_contents, stats


def write_wrong_assistant_contents(rows: list[dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_cleaned_records_parquet(
    records: list[dict[str, Any]],
    output_path: str,
    *,
    columns: list[str] | None = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if records:
        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame(columns=columns or [])
    df.to_parquet(output_path, index=False)


def write_summary(summary: dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)


def _default_clean_output_path(output_path: str) -> str:
    base, ext = os.path.splitext(output_path)
    if ext:
        return f"{base}.cleaned.parquet"
    return f"{output_path}.cleaned.parquet"


def _default_summary_output_path(output_path: str) -> str:
    base, ext = os.path.splitext(output_path)
    if ext:
        return f"{base}.summary.json"
    return f"{output_path}.summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and clean Inspector SFT assistant contents that do not match expected block format."
    )
    parser.add_argument("--input", required=True, help="Inspector SFT JSONL/parquet file.")
    parser.add_argument("--output", required=True, help="JSONL path for wrong assistant contents.")
    parser.add_argument(
        "--clean-output",
        default=None,
        help="Parquet path for records whose assistant contents are valid after cleaning.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="JSON path for aggregate cleaning statistics.",
    )
    parser.add_argument(
        "--messages-field",
        default="messages",
        help="Input field containing the trajectory messages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    clean_output = args.clean_output or _default_clean_output_path(args.output)
    summary_output = args.summary_output or _default_summary_output_path(args.output)
    cleaned_records, wrong_contents, stats = clean_records(records, messages_field=args.messages_field)
    summary = {
        "input": os.path.abspath(args.input),
        "wrong_output": os.path.abspath(args.output),
        "clean_output": os.path.abspath(clean_output),
        "summary_output": os.path.abspath(summary_output),
        "messages_field": args.messages_field,
        "error_type_counts": stats["correction_type_counts"],
        "total_errors": stats["wrong_assistant_contents"],
        "corrected_errors": stats["corrected_assistant_contents"],
        "uncorrected_errors": stats["uncorrected_assistant_contents"],
        "cleaned_parquet_records": stats["cleaned_records"],
        "total_records": stats["total_records"],
    }

    write_wrong_assistant_contents(wrong_contents, args.output)
    write_cleaned_records_parquet(
        cleaned_records,
        clean_output,
        columns=list(records[0].keys()) if records else None,
    )
    write_summary(summary, summary_output)

    print(f"Loaded {stats['total_records']} Inspector SFT records")
    print(f"Found {stats['wrong_assistant_contents']} wrong-format assistant contents")
    print(f"Corrected {stats['corrected_assistant_contents']} assistant contents")
    print(f"Still invalid after cleaning: {stats['uncorrected_assistant_contents']}")
    print(f"Correction type counts: {json.dumps(stats['correction_type_counts'], ensure_ascii=False, sort_keys=True)}")
    print(f"Saved wrong-format contents to: {args.output}")
    print(f"Saved cleaned records to: {clean_output}")
    print(f"Saved summary to: {summary_output}")


if __name__ == "__main__":
    main()
