"""Check Inspector SFT assistant message format.

Reads parquet/JSONL produced by ``evaluation.build_inspector_sft_data`` and
writes every assistant content that does not follow:

  <think>...</think>[<result>...</result>]<tool_call>...</tool_call>[<tool_call>...</tool_call>...]

Rules enforced:
  - Content is made *only* of <think>, <result>, <tool_call> blocks (no bare text).
  - <think> appears exactly once and must be the first block.
  - <result> is optional and may appear at most once, after <think>.
  - One or more <tool_call> blocks follow (required).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

from evaluation.sft_data_utils import coerce_messages, load_records, parse_jsonish


_BLOCK_RE = re.compile(
    r"<(?P<tag>think|result|tool_call)>(?P<body>.*?)</(?P=tag)>",
    re.DOTALL,
)


def _content_block_tags(content: str) -> list[str] | None:
    """Return ordered block tag list, or None if bare text exists outside blocks."""
    tags: list[str] = []
    cursor = 0
    for m in _BLOCK_RE.finditer(content):
        if content[cursor : m.start()].strip():
            return None
        tags.append(m.group("tag"))
        cursor = m.end()
    if content[cursor:].strip():
        return None
    return tags


def is_valid_assistant_content(content: str) -> bool:
    """Return True iff content follows the expected block structure."""
    tags = _content_block_tags(content)
    if not tags or tags[0] != "think":
        return False
    rest = tags[1:]
    if rest and rest[0] == "result":
        rest = rest[1:]
    return bool(rest) and all(t == "tool_call" for t in rest)


def _instance_id(record: dict[str, Any]) -> str:
    extra = parse_jsonish(record.get("extra_info", {}))
    if isinstance(extra, dict) and extra.get("instance_id"):
        return str(extra["instance_id"])
    return str(record.get("instance_id", "") or "")


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def collect_wrong_format_contents(
    records: list[dict[str, Any]],
    *,
    messages_field: str = "messages",
) -> list[dict[str, Any]]:
    """Return one entry per wrong-format assistant message."""
    errors: list[dict[str, Any]] = []
    for record in records:
        iid = _instance_id(record)
        for msg in coerce_messages(record.get(messages_field)):
            if msg.get("role") != "assistant":
                continue
            content = _stringify_content(msg.get("content", ""))
            if not is_valid_assistant_content(content):
                errors.append({"instance_id": iid, "content": content})
    return errors


def write_errors(errors: list[dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in errors:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Inspector SFT data for malformed assistant messages."
    )
    parser.add_argument("--input", required=True, help="JSONL or parquet SFT file to check.")
    parser.add_argument("--output", required=True, help="JSONL path for wrong-format assistant contents.")
    parser.add_argument(
        "--messages-field",
        default="messages",
        help="Record field containing the trajectory messages (default: messages).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    errors = collect_wrong_format_contents(records, messages_field=args.messages_field)
    write_errors(errors, args.output)

    total_msgs = sum(
        sum(1 for m in coerce_messages(r.get(args.messages_field)) if m.get("role") == "assistant")
        for r in records
    )
    print(f"Checked {len(records)} records ({total_msgs} assistant messages)")
    print(f"Found {len(errors)} wrong-format assistant messages")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
