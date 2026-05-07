"""Convert multi_agent_inference.py trajs.jsonl to location hint file.

Reads the trajs.jsonl produced by MultiAgentLocalizeRunner and writes a JSONL
file in the gt_location_with_support format expected by run_eval_with_patch_hints.sh.

Predicted locations from ``found_related_locs`` become ``edited_entities``.
``support_entities`` is left empty (pass --use-confirmed-as-support to populate
it from ``confirmed_suspicious`` instead).

Usage:
    python -m evaluation.convert_traj_to_location_file \\
        --input  outputs/my_run/traj/trajs.jsonl \\
        --output evaluation/gt_location/my_run/test/gt_location_with_support.jsonl

    # Also write support entities from confirmed_suspicious:
    python -m evaluation.convert_traj_to_location_file \\
        --input  outputs/my_run/traj/trajs.jsonl \\
        --output evaluation/gt_location/my_run/test/gt_location_with_support.jsonl \\
        --use-confirmed-as-support
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _parse_found_related_locs(found_related_locs: dict[str, list[str]]) -> list[str]:
    """Convert found_related_locs → flat list of 'file:func' strings.

    found_related_locs format (from _parse_trace_locs / _build_found_related_locs):
        {
            "django/db/models/base.py": ["function: Model.save\\nfunction: Model.full_clean"],
            ...
        }
    Each file maps to a single-element list whose element is a newline-joined
    string of ``function: X`` lines.
    """
    entities: list[str] = []
    seen: set[str] = set()
    for file_path, func_strs in (found_related_locs or {}).items():
        for func_block in func_strs or []:
            for line in str(func_block).split("\n"):
                line = line.strip()
                if not line.startswith("function:"):
                    continue
                func_name = line[len("function:"):].strip()
                if not func_name:
                    continue
                entity = f"{file_path}:{func_name}"
                if entity not in seen:
                    seen.add(entity)
                    entities.append(entity)
    return entities


def _parse_confirmed_suspicious(confirmed_suspicious: list[dict]) -> list[str]:
    """Convert confirmed_suspicious → flat list of 'file:func' strings.

    confirmed_suspicious format:
        [{"location": "file.py:ClassName.method", "reason": "..."}, ...]
    """
    entities: list[str] = []
    seen: set[str] = set()
    for item in confirmed_suspicious or []:
        loc = str(item.get("location") or "").strip()
        if ":" not in loc:
            continue
        if loc not in seen:
            seen.add(loc)
            entities.append(loc)
    return entities


def _build_file_changes(edit_entities: list[str]) -> list[dict]:
    """Group edit entities by file into the file_changes structure.

    Output format matches gen_oracle_locations.py:
        [
            {
                "file": "path/to/file.py",
                "changes": {
                    "edited_entities": ["path/to/file.py:FuncName", ...]
                }
            },
            ...
        ]
    """
    # Preserve insertion order per file
    file_order: list[str] = []
    file_entities: dict[str, list[str]] = {}
    for entity in edit_entities:
        file_path = entity.split(":", 1)[0]
        if file_path not in file_entities:
            file_order.append(file_path)
            file_entities[file_path] = []
        file_entities[file_path].append(entity)

    return [
        {
            "file": fp,
            "changes": {"edited_entities": file_entities[fp]},
        }
        for fp in file_order
    ]


def convert_record(
    record: dict[str, Any],
    *,
    use_confirmed_as_support: bool = False,
) -> dict[str, Any] | None:
    """Convert a single trajs.jsonl record to the location hint format.

    Returns None when the record has no usable location data.
    """
    instance_id = str(record.get("instance_id") or "").strip()
    if not instance_id:
        return None

    found_related_locs = record.get("found_related_locs") or {}
    confirmed_suspicious = record.get("confirmed_suspicious") or []

    # Primary source: found_related_locs (ranked final output)
    edit_entities = _parse_found_related_locs(found_related_locs)

    # Fallback: confirmed_suspicious when finalize produced no output
    if not edit_entities:
        edit_entities = _parse_confirmed_suspicious(confirmed_suspicious)

    support_entities: list[str] = []
    if use_confirmed_as_support:
        # Use confirmed_suspicious as support context (exclude items already in edit)
        edit_set = set(edit_entities)
        for entity in _parse_confirmed_suspicious(confirmed_suspicious):
            if entity not in edit_set:
                support_entities.append(entity)

    return {
        "instance_id": instance_id,
        "file_changes": _build_file_changes(edit_entities),
        "support_entities": support_entities,
        "repo": str(record.get("repo") or ""),
        "base_commit": record.get("base_commit") or "",
        "problem_statement": str(record.get("problem_statement") or ""),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert multi_agent_inference trajs.jsonl to location hint file."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to trajs.jsonl produced by MultiAgentLocalizeRunner.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for the gt_location_with_support.jsonl file.",
    )
    parser.add_argument(
        "--use-confirmed-as-support",
        action="store_true",
        default=False,
        help=(
            "Populate support_entities from confirmed_suspicious locations "
            "that are not already in edit_entities. Default: leave support_entities empty."
        ),
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    n_read = n_written = n_empty = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for lineno, raw in enumerate(fin, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[warn] line {lineno}: JSON decode error: {exc}", file=sys.stderr)
                continue

            n_read += 1
            out = convert_record(
                record,
                use_confirmed_as_support=args.use_confirmed_as_support,
            )
            if out is None:
                print(
                    f"[warn] line {lineno}: skipped (missing instance_id)",
                    file=sys.stderr,
                )
                continue

            if not out["file_changes"]:
                n_empty += 1

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_written += 1

    print(
        f"Done. read={n_read}  written={n_written}  "
        f"no_edit_entities={n_empty}  output={args.output}"
    )


if __name__ == "__main__":
    main()
