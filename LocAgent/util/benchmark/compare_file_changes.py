"""Compare file_changes in gt_location.jsonl vs gt_location_with_support.jsonl.

For each instance present in both files, checks that every field inside
``file_changes`` (edited_entities, added_entities, edited_modules,
added_modules) is identical.  Differences are printed verbosely so they are
easy to diagnose.

Usage
-----
python compare_file_changes.py \\
    --old  evaluation/gt_location/SWE-bench_Lite/test/gt_location.jsonl \\
    --new  evaluation/gt_location/swe_bench_lite/test/gt_location_with_support.jsonl

Optional flags
    --keys  comma-separated change keys to compare (default: edited_entities,added_entities)
    --show_sample N  print the first N differing instances in full detail (default: 5)
"""

import argparse
import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> Dict[str, Any]:
    """Return {instance_id -> row} from a jsonl file."""
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{lineno} JSON decode error: {e}", file=sys.stderr)
                continue
            iid = row.get("instance_id")
            if iid is None:
                print(f"[WARN] {path}:{lineno} missing instance_id", file=sys.stderr)
                continue
            records[iid] = row
    return records


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm_list(v) -> List[str]:
    """Sorted deduplicated list of strings (handles None / missing)."""
    if not v:
        return []
    return sorted(set(str(x) for x in v))


def _extract_changes(row: Dict, keys: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns {rel_file -> {key -> sorted list}} from row['file_changes'].
    Aggregates across all hunks/files.
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    for fc in row.get("file_changes") or []:
        fname = fc.get("file", "")
        changes = fc.get("changes") or {}
        file_data: Dict[str, List[str]] = {}
        for k in keys:
            file_data[k] = _norm_list(changes.get(k))
        out[fname] = file_data
    return out


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------

def diff_instance(iid: str, old_row: Dict, new_row: Dict,
                  keys: List[str]) -> Optional[Dict]:
    """
    Compare file_changes for one instance.
    Returns a diff dict if different, None if identical.
    """
    old_fc = _extract_changes(old_row, keys)
    new_fc = _extract_changes(new_row, keys)

    all_files = sorted(set(old_fc) | set(new_fc))
    diffs = {}

    for fname in all_files:
        old_file = old_fc.get(fname, {})
        new_file = new_fc.get(fname, {})
        file_diff = {}

        for k in keys:
            old_vals = old_file.get(k) or []
            new_vals = new_file.get(k) or []
            if old_vals != new_vals:
                missing = [v for v in old_vals if v not in new_vals]
                extra   = [v for v in new_vals if v not in old_vals]
                file_diff[k] = {"missing": missing, "extra": extra}

        if file_diff:
            diffs[fname] = file_diff

    return diffs if diffs else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare file_changes between old and new gt location files.")
    parser.add_argument("--old", required=True, help="Path to old gt_location.jsonl")
    parser.add_argument("--new", required=True, help="Path to new gt_location_with_support.jsonl")
    parser.add_argument("--keys", default="edited_entities,added_entities",
                        help="Comma-separated change keys to compare")
    parser.add_argument("--show_sample", type=int, default=5,
                        help="Print full diff for the first N differing instances")
    args = parser.parse_args()

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]

    print(f"Loading old: {args.old}")
    old_data = load_jsonl(args.old)
    print(f"  → {len(old_data)} instances")

    print(f"Loading new: {args.new}")
    new_data = load_jsonl(args.new)
    print(f"  → {len(new_data)} instances")

    only_old = sorted(set(old_data) - set(new_data))
    only_new = sorted(set(new_data) - set(old_data))
    common   = sorted(set(old_data) & set(new_data))

    print(f"\nCoverage")
    print(f"  only in old : {len(only_old)}")
    print(f"  only in new : {len(only_new)}")
    print(f"  in both     : {len(common)}")

    if only_old:
        print(f"\n[MISSING in new] first 10: {only_old[:10]}")
    if only_new:
        print(f"\n[EXTRA in new] first 10:   {only_new[:10]}")

    # Compare common instances
    n_identical = 0
    n_different = 0
    by_key_diff: Dict[str, int] = defaultdict(int)
    sample_printed = 0

    for iid in common:
        diff = diff_instance(iid, old_data[iid], new_data[iid], keys)
        if diff is None:
            n_identical += 1
        else:
            n_different += 1
            for fname, fdata in diff.items():
                for k in fdata:
                    by_key_diff[k] += 1

            if sample_printed < args.show_sample:
                sample_printed += 1
                print(f"\n{'='*60}")
                print(f"DIFF  instance_id={iid}")
                for fname, fdata in diff.items():
                    print(f"  file: {fname}")
                    for k, kdata in fdata.items():
                        if kdata["missing"]:
                            print(f"    {k}: MISSING from new: {kdata['missing']}")
                        if kdata["extra"]:
                            print(f"    {k}: EXTRA in new:    {kdata['extra']}")

    print(f"\n{'='*60}")
    print(f"Results (keys compared: {keys})")
    print(f"  identical : {n_identical} / {len(common)}")
    print(f"  different : {n_different} / {len(common)}")
    if by_key_diff:
        print(f"  diffs by key:")
        for k, cnt in sorted(by_key_diff.items()):
            print(f"    {k}: {cnt} instances")

    if n_different == 0:
        print("\nAll file_changes are identical.")
        sys.exit(0)
    else:
        print(f"\n{n_different} instance(s) differ — check diffs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
