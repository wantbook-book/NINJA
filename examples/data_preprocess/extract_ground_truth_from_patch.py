import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
# import debugpy
# debugpy.listen(5680)
# debugpy.wait_for_client()


_DEF_RE = re.compile(r"^\s*(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*\(")
_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*(?:\([^)]*\))?\s*:")


def _normalize_path(path: str) -> str:
    path = path.strip()
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


def _extract_func_from_line(line: str) -> Optional[str]:
    if not line:
        return None
    if line[0] in (" ", "+", "-"):
        line = line[1:]
    line = line.rstrip()
    match = _DEF_RE.match(line)
    if match:
        return match.group(1)
    return None


def _extract_class_from_line(line: str) -> Optional[str]:
    if not line:
        return None
    if line[0] in (" ", "+", "-"):
        line = line[1:]
    line = line.rstrip()
    match = _CLASS_RE.match(line)
    if match:
        return match.group(1)
    return None


def _strip_diff_prefix(line: str) -> tuple[str, str]:
    if not line:
        return "", line
    if line[0] in (" ", "+", "-"):
        return line[0], line[1:]
    return "", line


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _extract_func_from_hunk_header(line: str) -> Optional[str]:
    if not line.startswith("@@"):
        return None
    parts = line.split("@@")
    if len(parts) < 3:
        return None
    context = parts[2]
    return _extract_func_from_line(context)


def _extract_class_from_hunk_header(line: str) -> tuple[Optional[str], int]:
    if not line.startswith("@@"):
        return None, 0
    parts = line.split("@@")
    if len(parts) < 3:
        return None, 0
    context = parts[2]
    class_name = _extract_class_from_line(context)
    if not class_name:
        return None, 0
    return class_name, _line_indent(context)


def _scan_hunk_for_functions(
    lines: List[str],
    header_class: Optional[str] = None,
    header_class_indent: int = 0,
) -> tuple[List[str], Dict[str, set[str]], bool, bool, bool]:
    touched: List[str] = []
    touched_set: set[str] = set()
    def_signs: Dict[str, set[str]] = {}
    scopes: List[Dict[str, Any]] = []
    has_class_def = False
    has_def_line = False
    saw_change = False
    pending_decorators: Dict[int, bool] = {}

    def _current_def() -> Optional[str]:
        for scope in reversed(scopes):
            if scope["kind"] == "def":
                return scope["name"]
        return None

    def _current_first_layer_def() -> Optional[str]:
        for index in range(len(scopes) - 1, -1, -1):
            scope = scopes[index]
            if scope["kind"] != "def":
                continue
            parent_kind = scopes[index - 1]["kind"] if index > 0 else None
            if parent_kind != "def":
                return str(scope["name"])
        return None

    def _current_class_prefix() -> Optional[str]:
        for scope in reversed(scopes):
            if scope["kind"] == "class":
                return scope["name"]
        return None

    def _record(func_name: str) -> None:
        if func_name not in touched_set:
            touched_set.add(func_name)
            touched.append(func_name)

    if header_class:
        scopes.append({"indent": header_class_indent, "kind": "class", "name": header_class})

    for raw in lines:
        if raw.startswith("\\"):
            continue
        sign, content = _strip_diff_prefix(raw)
        if sign in ("+", "-"):
            saw_change = True
        line = content.rstrip("\n")
        if not line.strip():
            continue

        indent = _line_indent(line)

        if line.lstrip().startswith("@"):
            while scopes and indent <= scopes[-1]["indent"]:
                scopes.pop()
            if sign in ("+", "-"):
                pending_decorators[indent] = True
            continue

        while scopes and indent <= scopes[-1]["indent"]:
            scopes.pop()

        class_name = _extract_class_from_line(line)
        if class_name:
            has_class_def = True
            class_prefix = _current_class_prefix()
            full_class = f"{class_prefix}.{class_name}" if class_prefix else class_name
            scopes.append({"indent": indent, "kind": "class", "name": full_class})
            continue

        func_name = _extract_func_from_line(line)
        if func_name:
            has_def_line = True
            class_prefix = _current_class_prefix()
            qualified = f"{class_prefix}.{func_name}" if class_prefix else func_name
            scopes.append({"indent": indent, "kind": "def", "name": qualified})
            canonical = _current_first_layer_def() or qualified
            if sign in (" ", "+", "-"):
                def_signs.setdefault(qualified, set()).add(sign)
            if pending_decorators.pop(indent, False):
                _record(canonical)
            if sign in ("+", "-"):
                _record(canonical)
            continue

        if sign in ("+", "-"):
            current_def = _current_first_layer_def() or _current_def()
            if current_def:
                _record(current_def)

    return touched, def_signs, has_class_def, has_def_line, saw_change


def extract_edit_functions_from_patch(patch: str) -> List[str]:
    edit_functions, new_only = extract_edit_functions_with_metadata(patch)
    if not new_only:
        return edit_functions
    return [func for func in edit_functions if func not in new_only]


# diff --git a/uxarray/grid/coordinates.py b/uxarray/grid/coordinates.py
# index 45e00ba42..2d78b978a 100644
# --- a/uxarray/grid/coordinates.py
# +++ b/uxarray/grid/coordinates.py
# @@ -328,23 +328,25 @@ def _construct_face_centroids(node_x, node_y, node_z, face_nodes, n_nodes_per_fa
# tuple
# The x, y, and z coordinates of the centroids.
# """
# +
# centroid_x = np.zeros((face_nodes.shape[0]), dtype=np.float64)
# centroid_y = np.zeros((face_nodes.shape[0]), dtype=np.float64)
# centroid_z = np.zeros((face_nodes.shape[0]), dtype=np.float64)
# - n_face = n_nodes_per_face.shape[0]
# -
# - for i_face in prange(n_face):
# - n_max_nodes = n_nodes_per_face[i_face]

# - x = np.mean(node_x[face_nodes[i_face, 0:n_max_nodes]])
# - y = np.mean(node_y[face_nodes[i_face, 0:n_max_nodes]])
# - z = np.mean(node_z[face_nodes[i_face, 0:n_max_nodes]])
# + for face_idx in prange(face_nodes.shape[0]):
# + n_max_nodes = n_nodes_per_face[face_idx]
# + # Compute Cartesian Average
# + x = np.mean(node_x[face_nodes[face_idx, 0:n_max_nodes]])
# + y = np.mean(node_y[face_nodes[face_idx, 0:n_max_nodes]])
# + z = np.mean(node_z[face_nodes[face_idx, 0:n_max_nodes]])

# + # Normalize coordinates
# x, y, z = _normalize_xyz_scalar(x, y, z)
# + # Store coordinates
# + centroid_x[face_idx] = x
# + centroid_y[face_idx] = y
# + centroid_z[face_idx] = z

# - centroid_x[i_face] = x
# - centroid_y[i_face] = y
# - centroid_z[i_face] = z
# return centroid_x, centroid_y, centroid_z

def extract_edit_functions_with_metadata(
    patch: str,
    path_stats: Optional[Dict[str, int]] = None,
) -> tuple[List[str], set[str]]:
    if not isinstance(patch, str) or not patch.strip():
        return [], set()
    lines = patch.splitlines()
    current_file: Optional[str] = None
    seen: set[str] = set()
    results: List[str] = []
    new_candidates: set[str] = set()
    existing_funcs: set[str] = set()

    last_diff_paths: Optional[tuple[str, str]] = None
    last_minus: Optional[str] = None
    last_plus: Optional[str] = None
    mismatch_in_patch = False

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                a_path = _normalize_path(parts[2])
                b_path = _normalize_path(parts[3])
                current_file = b_path if b_path != "/dev/null" else a_path
                last_diff_paths = (a_path, b_path)
                last_minus = None
                last_plus = None
        elif line.startswith("--- "):
            path = _normalize_path(line[4:].strip())
            if path != "/dev/null":
                current_file = path
            last_minus = path
            if path_stats is not None and last_diff_paths is not None:
                a_path = last_diff_paths[0]
                if path != "/dev/null" and a_path != "/dev/null" and path != a_path:
                    path_stats["mismatch_git_minus"] = path_stats.get("mismatch_git_minus", 0) + 1
                    mismatch_in_patch = True
        elif line.startswith("+++ "):
            path = _normalize_path(line[4:].strip())
            if path != "/dev/null":
                current_file = path
            last_plus = path
            if path_stats is not None and last_diff_paths is not None:
                b_path = last_diff_paths[1]
                if path != "/dev/null" and b_path != "/dev/null" and path != b_path:
                    path_stats["mismatch_git_plus"] = path_stats.get("mismatch_git_plus", 0) + 1
                    mismatch_in_patch = True
            if path_stats is not None and last_minus is not None:
                if path != "/dev/null" and last_minus != "/dev/null" and path != last_minus:
                    path_stats["mismatch_minus_plus"] = path_stats.get("mismatch_minus_plus", 0) + 1
                    mismatch_in_patch = True
        elif line.startswith("@@"):
            hunk_header = line
            i += 1
            hunk_lines: List[str] = []
            while i < len(lines) and not lines[i].startswith("@@") and not lines[i].startswith("diff --git "):
                hunk_lines.append(lines[i])
                i += 1
            header_class, header_class_indent = _extract_class_from_hunk_header(hunk_header)
            (
                touched_funcs,
                def_signs,
                has_class_def,
                has_def_line,
                saw_change,
            ) = _scan_hunk_for_functions(
                hunk_lines,
                header_class=header_class,
                header_class_indent=header_class_indent,
            )
            if not touched_funcs:
                header_func = _extract_func_from_hunk_header(hunk_header)
                if header_func and saw_change and not has_class_def and not has_def_line:
                    touched_funcs = [header_func]
            if current_file and touched_funcs:
                for func_name in touched_funcs:
                    key = f"{current_file}:{func_name}"
                    if key not in seen:
                        seen.add(key)
                        results.append(key)

            if current_file and def_signs:
                for func_name, signs in def_signs.items():
                    key = f"{current_file}:{func_name}"
                    if signs <= {"+"}:
                        if key not in existing_funcs:
                            new_candidates.add(key)
                    else:
                        existing_funcs.add(key)
                        new_candidates.discard(key)
            continue
        i += 1

    if path_stats is not None and mismatch_in_patch:
        path_stats["patches_with_path_mismatch"] = path_stats.get("patches_with_path_mismatch", 0) + 1

    new_only = new_candidates - existing_funcs
    return results, new_only


def _iter_rows(df: pd.DataFrame) -> Iterable[dict]:
    for _, row in df.iterrows():
        yield row.to_dict()


def _normalize_func_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except json.JSONDecodeError:
                pass
    return [str(value)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract edit functions from patch in parquet rows.")
    parser.add_argument("--parquet", required=True, help="Input parquet file path.")
    parser.add_argument("--output", default=None, help="Output parquet file path.")
    parser.add_argument("--extra-info-key", default="extra_info", help="Extra info field name in parquet.")
    parser.add_argument("--patch-key", default="patch", help="Patch field name inside extra_info or row.")
    parser.add_argument("--edit-functions-key", default="edit_functions", help="Output field name.")
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Drop rows that have no extracted edit functions.",
    )
    parser.add_argument(
        "--drop-new-functions-only",
        action="store_true",
        help="Drop rows where all extracted edit functions are newly added (not modified existing functions).",
    )
    parser.add_argument(
        "--update-reward-model",
        action="store_true",
        help="Update reward_model.ground_truth with extracted edit functions.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Compare extracted edit functions with reward_model.ground_truth and report mismatched instance_id.",
    )
    parser.add_argument(
        "--mismatch-output",
        default=None,
        help="Output path for mismatch details (default: <input>_mismatches.jsonl).",
    )
    parser.add_argument(
        "--path-mismatch-output",
        default=None,
        help="Output path for path mismatch stats (default: <input>_path_mismatch_stats.json).",
    )
    parser.add_argument(
        "--stats-output",
        default=None,
        help="Output path for edit-function extraction stats (default: <input>_edit_function_stats.json).",
    )
    parser.add_argument("--reward-model-key", default="reward_model", help="Reward model field name.")
    parser.add_argument("--ground-truth-key", default="ground_truth", help="Ground truth key in reward model.")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    total_input_rows = len(df)
    edit_functions_column: List[List[str]] = []
    reward_model_column: Optional[List[Dict[str, Any]]] = [] if args.update_reward_model else None
    keep_mask: List[bool] = []
    empty_count = 0
    new_functions_only_count = 0
    dropped_empty = 0
    dropped_new_only = 0
    path_stats: Dict[str, int] = {}
    mismatched_instance_ids: List[str] = []
    mismatch_records: List[Dict[str, Any]] = []
    compared_rows = 0

    for row in _iter_rows(df):
        extra_info = row.get(args.extra_info_key)
        if not isinstance(extra_info, dict):
            extra_info = {}
        # if extra_info['instance_id'] == '':
        #     breakpoint()
        patch = extra_info.get(args.patch_key)
        if patch is None:
            patch = row.get(args.patch_key)
        raw_edit_functions, new_only = extract_edit_functions_with_metadata(patch, path_stats=path_stats)
        edit_functions = [func for func in raw_edit_functions if func not in new_only]
        is_empty = not edit_functions
        is_new_functions_only = bool(raw_edit_functions) and all(func in new_only for func in raw_edit_functions)
        if is_empty:
            empty_count += 1
        if is_new_functions_only:
            new_functions_only_count += 1
        edit_functions_column.append(edit_functions)
        if args.test:
            reward_model = row.get(args.reward_model_key)
            ground_truth = None
            if isinstance(reward_model, dict):
                ground_truth = reward_model.get(args.ground_truth_key)
            gt_list = ground_truth.tolist()
            if set(gt_list) != set(edit_functions):
                instance_id = extra_info.get("instance_id")
                instance_id_str = str(instance_id) if instance_id is not None else "<missing>"
                mismatched_instance_ids.append(instance_id_str)
                mismatch_records.append(
                    {
                        "instance_id": instance_id_str,
                        "ground_truth": gt_list,
                        "extracted": edit_functions,
                    }
                )
            compared_rows += 1

        keep = True
        if args.drop_empty and is_empty:
            keep = False
            dropped_empty += 1
        elif args.drop_new_functions_only and is_new_functions_only:
            keep = False
            dropped_new_only += 1

        keep_mask.append(keep)

        if reward_model_column is not None:
            reward_model = row.get(args.reward_model_key)
            if not isinstance(reward_model, dict):
                reward_model = {}
            else:
                reward_model = dict(reward_model)
            reward_model[args.ground_truth_key] = edit_functions
            reward_model_column.append(reward_model)

    df = df[keep_mask].copy()
    df[args.edit_functions_key] = [funcs for funcs, keep in zip(edit_functions_column, keep_mask) if keep]
    if reward_model_column is not None:
        df[args.reward_model_key] = [rm for rm, keep in zip(reward_model_column, keep_mask) if keep]

    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(args.parquet)
        output_path = f"{base}_with_edit_functions{ext or '.parquet'}"

    df.to_parquet(output_path, index=False)

    total = len(df)
    with_funcs = sum(1 for funcs in df[args.edit_functions_key] if funcs)
    print(f"Saved: {output_path}")
    print(
        "Rows: {total}, Rows with edit functions: {with_funcs}, "
        "Dropped empty: {dropped_empty}, Dropped new-only: {dropped_new_only}".format(
            total=total,
            with_funcs=with_funcs,
            dropped_empty=dropped_empty,
            dropped_new_only=dropped_new_only,
        )
    )
    stats_output = args.stats_output
    if stats_output is None:
        base, _ = os.path.splitext(args.parquet)
        stats_output = f"{base}_edit_function_stats.json"
    stats_payload = {
        "input_parquet": args.parquet,
        "output_parquet": output_path,
        "total_input_rows": total_input_rows,
        "total_output_rows": total,
        "rows_with_edit_functions_output": with_funcs,
        "empty_count": empty_count,
        "new_functions_only_count": new_functions_only_count,
        "dropped_empty": dropped_empty,
        "dropped_new_only": dropped_new_only,
    }
    with open(stats_output, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, ensure_ascii=False, indent=2)
    print(f"Edit-function stats saved to: {stats_output}")
    if path_stats:
        stats_output = args.path_mismatch_output
        if stats_output is None:
            base, _ = os.path.splitext(args.parquet)
            stats_output = f"{base}_path_mismatch_stats.json"
        with open(stats_output, "w", encoding="utf-8") as f:
            json.dump(path_stats, f, ensure_ascii=False, indent=2)
        print(f"Path mismatch stats saved to: {stats_output}")
    if args.test:
        mismatch_output = args.mismatch_output
        if mismatch_output is None:
            base, _ = os.path.splitext(args.parquet)
            mismatch_output = f"{base}_mismatches.jsonl"
        with open(mismatch_output, "w", encoding="utf-8") as f:
            for record in mismatch_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Mismatch details saved to: {mismatch_output}")


if __name__ == "__main__":
    main()
