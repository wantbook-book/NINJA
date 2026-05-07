import json
import math
import re

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

TRACE_LOCS_RE = re.compile(r"<trace_locs>(.*?)</trace_locs>", re.DOTALL | re.IGNORECASE)

def _convert_ndarray(obj):
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = _convert_ndarray(value)
        return obj
    if isinstance(obj, list):
        return [_convert_ndarray(value) for value in obj]
    return obj

def load_data(input_file: str):
    if pd is not None:
        try:
            df = pd.read_parquet(input_file)
            records = df.to_dict(orient="records")
            for idx, record in enumerate(records):
                records[idx] = _convert_ndarray(record)
            return records
        except Exception:
            pass
    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def ndcg_at_k(gt, preds, k):
    """
    Compute nDCG@k.
    gt: ground-truth set
    preds: model prediction list
    k: top-k positions to consider
    """
    preds_k = preds[:k]

    # Compute DCG@k
    dcg = 0.0
    for i, item in enumerate(preds_k):
        if item in gt:
            rank = i + 1
            dcg += 1.0 / (np.log2(rank + 1) if np is not None else math.log2(rank + 1))

    # Compute IDCG@k
    num_true_items = len(gt)
    ideal_ranks = min(num_true_items, k)
    
    idcg = 0.0
    for i in range(ideal_ranks):
        rank = i + 1
        idcg += 1.0 / (np.log2(rank + 1) if np is not None else math.log2(rank + 1))
        
    if idcg == 0:
        return 0.0
        
    return dcg / idcg


def parse_gt_methods(gt_entries):
    """
    Parse the entries in the ground truth and normalize them into file-level and
    function-level localizations.
    """
    files, methods = set(), set()

    for entry in gt_entries:
        parts = entry.split(':')

        if len(parts) == 2:  # file:function or file:Class.method
            file_name, method_or_class = parts
            files.add(file_name)
            methods.add(method_or_class)

        elif len(parts) == 1:  # File
            files.add(parts[0])

    return files, methods

def construct_pred_func(func_locs):
    final_funcs = []
    # for file_loc in file_locs:
    #     final_funcs.append(func_locs.get(file_loc, []))
    for key, item in func_locs.items():
        final_funcs.append(item)
    return final_funcs


def extract_locs(locs, keep_old_order=False):
    results= {}
    current_file_name = None
    for line in locs.splitlines():
        if line.strip().endswith(".py"):
            current_file_name = line.strip()
        elif line.strip() and any(
            line.startswith(w)
            for w in ["line:", "function:", "class:", "variable:"]
        ):
            if current_file_name not in results:
                results[current_file_name] = []
            results[current_file_name].append(line)

    return {fn: ["\n".join(results[fn])] for fn in results.keys()}



def extract_predicted_methods(found_related_locs):
    """
    Extract predicted function names from found_related_locs.

    Function-level evaluation only counts module-level functions and class
    methods declared via `function:` lines. Bare `class:` entries are ignored.
    """
    predicted_methods = []
    for sublist in found_related_locs:
        for loc in sublist:
            for entry in loc.split('\n'):
                if 'function:' in entry:
                    try:
                        predicted_methods.append(entry.split(': ')[1])
                    except Exception:
                        pass
    return predicted_methods

def reciprocal_rank(gt, preds):
    """
    Compute the reciprocal rank for a single query.
    """
    for i, p in enumerate(preds):
        if p in gt:
            return 1.0 / (i + 1)
    return 0.0


def top_k_accuracy(gt, preds, k):
    return any(item in gt for item in preds[:k])


def locagent_acc_at_k(gt, preds, k):
    if not gt:
        return 0.0
    # required_hits = min(len(gt), k)
    required_hits = len(gt)
    actual_hits = sum(1 for item in preds[:k] if item in gt)
    return float(actual_hits >= required_hits)


def recall_at_k(gt, preds, k):
    if len(gt) == 0:
        return 0
    hit_num = 0
    for p in preds[:k]:
        if p in gt:
            hit_num += 1
    return float(hit_num) / len(gt)


def average_precision(gt, preds):
    """
    Compute the average precision (AP) for a single query.
    """
    if not gt:
        return 0
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(preds):
        if p in gt:
            num_hits += 1.0
            score += num_hits / (i + 1)
    return score / len(gt)


def _extract_trace_locs(text: str) -> str:
    if not text:
        return ""
    matches = TRACE_LOCS_RE.findall(text)
    if not matches:
        return text
    parts = [part.strip() for part in matches if part.strip()]
    return "\n".join(parts) if parts else ""
