"""
Extract Inspector (Sub Agent) trajectories from multi-agent inference outputs.

Reads the output directory produced by ``multi_agent_inference.py`` and extracts
per-sub-agent trajectory records into a parquet file.  Each row represents one
Inspector run for a specific (instance_id, entry_file) pair.

Output directory structure expected (from multi_agent_inference.py):
    <inference_output_dir>/
        traj/
            trajs.jsonl                          # one JSON line per instance
            sub_agents/
                <instance_id>/
                    <entry_file>.json            # full sub-agent trajectory

Output parquet columns:
    - instance_id: str
    - repo: str
    - data_source: str
    - base_commit: str
    - problem_statement: str
    - structure: str
    - patch: str
    - ground_truth: str (JSON list of ground-truth function locations)
    - entry_file: str (the file the Inspector explored)
    - confirmed_suspicious: str (JSON list of suspicious locations found)
    - messages: str (JSON-serialized conversation history)
    - explored: str (JSON list of explored function locations)
    - sub_agent_stats: str (JSON dict of tool-call statistics)
    - error: str (error message if the sub-agent failed, else empty)

Usage:
    python -m examples.code_localization.extract_inspector_trajectories \\
        --inference_output_dir outputs/multi_agent/20240101_120000 \\
        --input_file data/code_loc/raw_data.parquet \\
        --output_file data/code_loc/inspector_trajectories.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _load_input_data_index(input_file: str) -> Dict[str, Dict[str, Any]]:
    """Load original data file and build a lookup dict keyed by instance_id.

    Returns {instance_id: {problem_statement, structure, patch, ...}}.
    """
    from evaluation.utils import load_data

    data = load_data(input_file)
    index: Dict[str, Dict[str, Any]] = {}
    for item in data:
        extra_info = item.get("extra_info", {})
        if not isinstance(extra_info, dict):
            extra_info = {}
        instance_id = str(
            item.get("instance_id")
            or extra_info.get("instance_id")
            or "unknown"
        )
        index[instance_id] = {
            "problem_statement": str(
                item.get("problem_statement")
                or extra_info.get("problem_statement")
                or ""
            ),
            "structure": str(
                item.get("structure")
                or extra_info.get("structure")
                or ""
            ),
            "patch": str(item.get("patch") or extra_info.get("patch") or ""),
        }
    return index


def _load_trajs_jsonl(trajs_path: str) -> List[Dict[str, Any]]:
    """Read traj/trajs.jsonl and return a list of parsed records."""
    records = []
    with open(trajs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSONL line: {e}")
    return records


def _load_sub_agent_detail(
    output_dir: str,
    detail_path: str,
) -> Optional[Dict[str, Any]]:
    """Load a single sub-agent detail JSON file."""
    full_path = osp.join(output_dir, detail_path)
    if not osp.exists(full_path):
        logger.warning(f"Sub-agent detail file not found: {full_path}")
        return None
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read {full_path}: {e}")
        return None


def extract_from_output_dir(
    inference_output_dir: str,
    input_index: Dict[str, Dict[str, Any]],
    instance_id_filter: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Extract sub-agent trajectory records from inference output directory."""
    trajs_path = osp.join(inference_output_dir, "traj", "trajs.jsonl")
    if not osp.exists(trajs_path):
        logger.error(f"trajs.jsonl not found at {trajs_path}")
        return []

    jsonl_records = _load_trajs_jsonl(trajs_path)
    logger.info(f"Loaded {len(jsonl_records)} instance records from {trajs_path}")

    all_records: List[Dict[str, Any]] = []

    for record in tqdm(jsonl_records, desc="Extracting sub-agent trajectories"):
        instance_id = str(record.get("instance_id", ""))
        if not instance_id:
            continue
        if instance_id_filter and instance_id not in instance_id_filter:
            continue

        # Instance-level fields from JSONL
        repo = str(record.get("repo", ""))
        data_source = str(record.get("data_source", ""))
        base_commit = str(record.get("base_commit", ""))

        # Ground truth
        ground_truth = record.get("ground_truth", "")
        if isinstance(ground_truth, list):
            ground_truth = json.dumps(ground_truth)
        ground_truth = str(ground_truth)

        # Fields from original input data
        input_meta = input_index.get(instance_id, {})
        problem_statement = input_meta.get("problem_statement", "")
        structure = input_meta.get("structure", "")
        patch = input_meta.get("patch", "")

        # Iterate sub-agent trajectories
        sub_agent_summaries = record.get("sub_agent_trajectories", [])
        if not sub_agent_summaries:
            continue

        for sa_summary in sub_agent_summaries:
            entry_file = str(sa_summary.get("entry_file", ""))
            detail_path = sa_summary.get("sub_agent_detail_path", "")

            # Load full detail (with messages) from the detail JSON file
            detail = None
            if detail_path:
                detail = _load_sub_agent_detail(inference_output_dir, detail_path)

            if detail is not None:
                # Use full detail: has messages
                suspicious = detail.get("suspicious", [])
                explored = detail.get("explored", [])
                messages = detail.get("messages", [])
                sub_agent_stats = detail.get("sub_agent_stats", {})
                error = detail.get("error") or ""
            else:
                # Fall back to summary (no messages)
                suspicious = sa_summary.get("suspicious", [])
                explored = sa_summary.get("explored", [])
                messages = []
                sub_agent_stats = sa_summary.get("sub_agent_stats", {})
                error = sa_summary.get("error") or ""

            all_records.append({
                "instance_id": instance_id,
                "repo": repo,
                "data_source": data_source,
                "base_commit": base_commit,
                "problem_statement": problem_statement,
                "structure": structure,
                "patch": patch,
                "ground_truth": ground_truth,
                "entry_file": entry_file,
                "confirmed_suspicious": json.dumps(suspicious, ensure_ascii=False),
                "messages": json.dumps(messages, ensure_ascii=False),
                "explored": json.dumps(explored, ensure_ascii=False),
                "sub_agent_stats": json.dumps(sub_agent_stats, ensure_ascii=False),
                "error": error,
            })

        logger.info(
            f"  {instance_id}: extracted {len(sub_agent_summaries)} sub-agent trajectories"
        )

    return all_records


def main():
    parser = argparse.ArgumentParser(
        description="Extract Inspector sub-agent trajectories from multi-agent inference outputs"
    )
    parser.add_argument(
        "--inference_output_dir", required=True,
        help="Output directory from multi_agent_inference.py (contains traj/trajs.jsonl)"
    )
    parser.add_argument(
        "--input_file", required=True,
        help="Original input data file (parquet or jsonl) for problem_statement/structure/patch"
    )
    parser.add_argument(
        "--output_file", required=True,
        help="Output parquet file for extracted trajectories"
    )
    parser.add_argument(
        "--instance_ids", type=str, default="",
        help="Comma-separated instance IDs to filter (empty = all)"
    )
    parser.add_argument(
        "--instance_ids_file", type=str, default="",
        help="File containing instance IDs to filter (one per line)"
    )
    args = parser.parse_args()

    # Build instance ID filter
    instance_id_filter: Optional[Set[str]] = None
    selected_ids: List[str] = []
    if args.instance_ids:
        selected_ids.extend(i.strip() for i in args.instance_ids.split(",") if i.strip())
    if args.instance_ids_file and osp.exists(args.instance_ids_file):
        with open(args.instance_ids_file, "r") as f:
            for line in f:
                v = line.strip()
                if v and not v.startswith("#"):
                    selected_ids.append(v)
    if selected_ids:
        instance_id_filter = set(selected_ids)
        logger.info(f"Filtering to {len(instance_id_filter)} instance IDs")

    # Load original input data for metadata lookup
    logger.info(f"Loading input data from {args.input_file}")
    input_index = _load_input_data_index(args.input_file)
    logger.info(f"Built metadata index for {len(input_index)} instances")

    # Extract trajectories from output directory
    all_records = extract_from_output_dir(
        args.inference_output_dir, input_index, instance_id_filter
    )

    if not all_records:
        logger.warning("No trajectories extracted!")
        return

    # Save to parquet
    df = pd.DataFrame(all_records)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    df.to_parquet(args.output_file, index=False)
    logger.info(f"Saved {len(df)} trajectory records to {args.output_file}")

    # Print summary
    instance_ids = df["instance_id"].unique()
    logger.info(f"  Unique instances: {len(instance_ids)}")
    logger.info(f"  Avg trajectories per instance: {len(df) / len(instance_ids):.1f}")

    # Count how many have suspicious findings
    n_with_suspicious = sum(
        1 for _, row in df.iterrows()
        if json.loads(row["confirmed_suspicious"])
    )
    logger.info(f"  Trajectories with suspicious findings: {n_with_suspicious}/{len(df)}")


if __name__ == "__main__":
    main()
