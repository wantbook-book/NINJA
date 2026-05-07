#!/usr/bin/env python3
"""Summarize metrics from multiple timestamped evaluate.py result directories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


TIME_DIR_RE = re.compile(r"^\d{8}_\d{6}$")

METRIC_FIELDS = (
    "file_level_acc_at_5",
    "file_level_f1_at_5",
    "function_level_acc_at_5",
    "function_level_f1_at_5",
    "avg_inference_time_seconds",
    "avg_total_plus_sub_agent_tokens_per_entry",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(path))
        current = current[key]
    return current


def _find_metrics_files(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(root.rglob("metrics.json"))

    candidates: list[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            for metrics_path in (child / "metrics.json", child / "traj" / "metrics.json"):
                if metrics_path.is_file():
                    candidates.append(metrics_path)
    root_metrics = root / "metrics.json"
    if root_metrics.is_file():
        candidates.append(root_metrics)
    root_traj_metrics = root / "traj" / "metrics.json"
    if root_traj_metrics.is_file():
        candidates.append(root_traj_metrics)
    return candidates


def _time_dir_for_metrics_path(metrics_path: Path) -> Path:
    parent = metrics_path.parent
    if parent.name == "traj":
        return parent.parent
    return parent


def _run_name(metrics_path: Path, root: Path) -> str:
    parent = _time_dir_for_metrics_path(metrics_path)
    try:
        rel_parent = parent.relative_to(root)
        return str(rel_parent) if str(rel_parent) != "." else parent.name
    except ValueError:
        return str(parent)


def _extract_run_row(metrics_path: Path, root: Path) -> tuple[dict[str, Any], list[str]]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    warnings: list[str] = []

    def metric(path: tuple[str, ...], default: float = 0.0) -> float:
        try:
            return _safe_float(_get_nested(metrics, path), default)
        except KeyError as exc:
            warnings.append(f"{_run_name(metrics_path, root)} missing {exc}")
            return default

    avg_total_tokens = metric(("usage", "avg_total_tokens"))
    avg_sub_agent_tokens = metric(("sub_agent_usage", "avg_sub_agent_tokens_per_entry"))

    row = {
        "run": _run_name(metrics_path, root),
        "metrics_path": str(metrics_path),
        "file_level_acc_at_5": metric(("file_level", "Acc@5")),
        "file_level_f1_at_5": metric(("file_level", "F1@5")),
        "function_level_acc_at_5": metric(("function_level", "Acc@5")),
        "function_level_f1_at_5": metric(("function_level", "F1@5")),
        "avg_inference_time_seconds": metric(("inference_time", "avg_inference_time_seconds")),
        "avg_total_tokens": avg_total_tokens,
        "avg_sub_agent_tokens_per_entry": avg_sub_agent_tokens,
        "avg_total_plus_sub_agent_tokens_per_entry": avg_total_tokens + avg_sub_agent_tokens,
    }
    return row, warnings


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float], ddof: int) -> float:
    if not values or len(values) <= ddof:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - ddof)
    return math.sqrt(variance)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_table(rows: list[dict[str, Any]], summary: dict[str, dict[str, float]]) -> None:
    columns = ["run", *METRIC_FIELDS]
    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            value = row[column]
            text = f"{value:.6f}" if isinstance(value, float) else str(value)
            widths[column] = max(widths[column], len(text))

    def render_row(values: dict[str, Any]) -> str:
        rendered = []
        for column in columns:
            value = values[column]
            text = f"{value:.6f}" if isinstance(value, float) else str(value)
            rendered.append(text.ljust(widths[column]))
        return "  ".join(rendered)

    print(render_row({column: column for column in columns}))
    print(render_row({column: "-" * widths[column] for column in columns}))
    for row in rows:
        print(render_row(row))

    print()
    print("summary")
    for field in METRIC_FIELDS:
        stats = summary[field]
        print(f"{field}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, n={int(stats['n'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Directory containing timestamped evaluate.py result directories.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Find metrics.json recursively instead of only under immediate child directories.",
    )
    parser.add_argument(
        "--time-dirs-only",
        action="store_true",
        help="Only include metrics.json files whose parent directory name matches YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write summary JSON. Defaults to <root_dir>/eval_runs_summary.json.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to write per-run CSV. Defaults to <root_dir>/eval_runs_summary.csv.",
    )
    parser.add_argument(
        "--std-ddof",
        type=int,
        default=1,
        help="Delta degrees of freedom for std. Use 1 for sample std, 0 for population std.",
    )
    args = parser.parse_args()

    root = args.root_dir
    if not root.is_dir():
        raise NotADirectoryError(root)

    metrics_files = _find_metrics_files(root, recursive=args.recursive)
    if args.time_dirs_only:
        metrics_files = [path for path in metrics_files if TIME_DIR_RE.match(_time_dir_for_metrics_path(path).name)]
    if not metrics_files:
        raise FileNotFoundError(f"No metrics.json files found under {root}")

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for metrics_path in metrics_files:
        row, row_warnings = _extract_run_row(metrics_path, root)
        rows.append(row)
        warnings.extend(row_warnings)

    summary = {}
    for field in METRIC_FIELDS:
        values = [_safe_float(row[field]) for row in rows]
        summary[field] = {
            "mean": _mean(values),
            "std": _std(values, args.std_ddof),
            "n": float(len(values)),
        }

    output_json = args.output_json or (root / "eval_runs_summary.json")
    output_csv = args.output_csv or (root / "eval_runs_summary.csv")
    payload = {
        "root_dir": str(root),
        "num_runs": len(rows),
        "std_ddof": args.std_ddof,
        "runs": rows,
        "summary": summary,
        "warnings": warnings,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_csv, rows)
    _print_table(rows, summary)
    print(f"\nWrote JSON summary: {output_json}")
    print(f"Wrote per-run CSV: {output_csv}")
    if warnings:
        print(f"Warnings: {len(warnings)}; see JSON summary for details.")


if __name__ == "__main__":
    main()
