"""
Tools for the multi-agent Issue Localization framework.

This module provides:
- Sub Agent tools: get_callers, get_callees (based on dependency graph)
- Main Agent tools: find_files_by_content_repo, find_files_by_name_repo
- Existing tools re-exported: get_methods_of_class, get_file_functions,
  get_file_classes, get_code_of_function, get_code_of_class_method
"""

import json
import os
import subprocess
import pickle
import logging
from typing import List, Optional

import networkx as nx

from tools.repo_search_tools import (
    get_methods_of_class,
    get_file_functions,
    get_file_classes,
    get_code_of_file_function,
    get_code_of_class_function,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph loading utilities
# ---------------------------------------------------------------------------

_graph_cache = {}


def _load_graph(instance_id: str) -> nx.MultiDiGraph:
    """Load the dependency graph for the given instance_id from the graph index directory."""
    if instance_id in _graph_cache:
        return _graph_cache[instance_id]

    graph_index_dir = os.environ.get("GRAPH_INDEX_DIR", "graph_index")
    graph_file = os.path.join(graph_index_dir, f"{instance_id}.pkl")

    if not os.path.exists(graph_file):
        raise FileNotFoundError(
            f"Graph file not found: {graph_file}. "
            f"Please build the dependency graph first."
        )

    with open(graph_file, "rb") as f:
        G = pickle.load(f)

    _graph_cache[instance_id] = G
    return G


def _is_test_file(nid: str) -> bool:
    """Check if a node id belongs to a test file."""
    import re
    file_path = nid.split(':')[0]
    word_list = re.split(r" |_|\/", file_path.lower())
    return any(word.startswith('test') for word in word_list)


# ---------------------------------------------------------------------------
# Sub Agent tools: get_callers / get_callees
# ---------------------------------------------------------------------------

def get_callers(location: str, instance_id: str) -> str:
    """
    Return a list of functions that call the specified function.

    Parameters
    ----------
    location : str
        The function location in format:
        - "file_path:function_name" for module-level functions
        - "file_path:ClassName.method_name" for class methods
    instance_id : str
        The instance ID for graph lookup.

    Returns
    -------
    str
        JSON string of callers in the format:
        {"callers": ["file:function", "file:Class.method", ...]}
    """
    try:
        G = _load_graph(instance_id)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    # The node_id in the graph is the same format: "file_path:QualifiedName"
    node_id = location

    if node_id not in G:
        # Try to find a matching node
        matched = _fuzzy_match_node(G, location)
        if matched:
            node_id = matched
        else:
            return json.dumps(
                {"error": f"Node '{location}' not found in the dependency graph. "
                          f"Please check the location format (file:function or file:Class.method)."},
                ensure_ascii=False,
            )

    callers = []
    # EDGE_TYPE_INVOKES = 'invokes'
    # In the graph, caller --invokes--> callee
    # So to find callers of `node_id`, we look at predecessors with 'invokes' edge
    for predecessor in G.predecessors(node_id):
        if _is_test_file(predecessor):
            continue
        edge_data = G.get_edge_data(predecessor, node_id)
        for key, data in edge_data.items():
            if data.get('type') == 'invokes':
                callers.append(predecessor)
                break

    # Deduplicate while preserving order
    seen = set()
    unique_callers = []
    for c in callers:
        if c not in seen:
            seen.add(c)
            unique_callers.append(c)

    return json.dumps({"callers": unique_callers}, ensure_ascii=False)


def get_callees(location: str, instance_id: str) -> str:
    """
    Return a list of functions that are called by the specified function.

    Parameters
    ----------
    location : str
        The function location in format:
        - "file_path:function_name" for module-level functions
        - "file_path:ClassName.method_name" for class methods
    instance_id : str
        The instance ID for graph lookup.

    Returns
    -------
    str
        JSON string of callees in the format:
        {"callees": ["file:function", "file:Class.method", ...]}
    """
    try:
        G = _load_graph(instance_id)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    node_id = location

    if node_id not in G:
        matched = _fuzzy_match_node(G, location)
        if matched:
            node_id = matched
        else:
            return json.dumps(
                {"error": f"Node '{location}' not found in the dependency graph. "
                          f"Please check the location format (file:function or file:Class.method)."},
                ensure_ascii=False,
            )

    callees = []
    # In the graph, caller --invokes--> callee
    # So to find callees of `node_id`, we look at successors with 'invokes' edge
    for successor in G.successors(node_id):
        if _is_test_file(successor):
            continue
        edge_data = G.get_edge_data(node_id, successor)
        for key, data in edge_data.items():
            if data.get('type') == 'invokes':
                callees.append(successor)
                break

    seen = set()
    unique_callees = []
    for c in callees:
        if c not in seen:
            seen.add(c)
            unique_callees.append(c)

    return json.dumps({"callees": unique_callees}, ensure_ascii=False)


def _fuzzy_match_node(G: nx.MultiDiGraph, location: str) -> Optional[str]:
    """Try to fuzzy match a node in the graph."""
    if ':' not in location:
        return None

    file_path, func_name = location.rsplit(':', 1)

    # Try exact match first
    if location in G:
        return location

    # Try matching by suffix
    for node in G.nodes():
        if ':' in node:
            node_file, node_func = node.rsplit(':', 1)
            if node_file == file_path and node_func.endswith(func_name):
                return node
            if node_file == file_path and func_name.endswith(node_func):
                return node

    return None


# ---------------------------------------------------------------------------
# Main Agent tools: find_files_by_content_repo / find_files_by_name_repo
# ---------------------------------------------------------------------------

def find_files_by_content_repo(keywords: List[str], instance_id: str, file_pattern: str = "*.py") -> str:
    """
    Search for one or more keywords or regex patterns in the repository files,
    returning a deduplicated list of files that contain **any** of them.

    Each element of *keywords* is first tried as a regular expression
    (``re.search``).  If the regex is invalid it falls back to a plain
    case-insensitive substring match, so simple keywords still work.

    Parameters
    ----------
    keywords : list[str]
        Keywords or regex patterns to search for.
    instance_id : str
        The instance ID for locating the repository.
    file_pattern : str
        File glob pattern to filter files (default: "*.py").

    Returns
    -------
    str
        JSON string with matching files and line snippets.
    """
    project_file_loc = os.environ.get("PROJECT_FILE_LOC", None)
    if project_file_loc is None:
        return json.dumps({"error": "PROJECT_FILE_LOC not set"}, ensure_ascii=False)

    # Load the repo structure file to get all file contents
    from tools.utils.utils import load_json
    try:
        d = load_json(f"{project_file_loc}/{instance_id}.json")
    except Exception as e:
        return json.dumps({"error": f"Failed to load repo structure: {e}"}, ensure_ascii=False)

    structure = d.get("structure", {})

    from tools.RepoSearch.preprocess_data import extract_structure
    files, _, _ = extract_structure(structure)

    import fnmatch
    import re as _re

    # Pre-compile regex patterns; fall back to plain substring for invalid ones
    compiled_patterns = []
    for kw in keywords:
        try:
            compiled_patterns.append(_re.compile(kw, _re.IGNORECASE))
        except _re.error:
            # Escape and treat as literal substring
            compiled_patterns.append(_re.compile(_re.escape(kw), _re.IGNORECASE))

    matches = []
    seen_files = set()
    for file_item in files:
        file_name = file_item[0]

        # Apply file pattern filter
        if not fnmatch.fnmatch(file_name, file_pattern):
            continue

        # Skip test files
        if _is_test_file(file_name + ":dummy"):
            continue

        file_lines = file_item[-1]
        matched_lines = []
        for i, line in enumerate(file_lines, start=1):
            for pat in compiled_patterns:
                if pat.search(line):
                    matched_lines.append({
                        "line_number": i,
                        "content": line.strip()
                    })
                    break  # one match per line is enough

        if matched_lines and file_name not in seen_files:
            seen_files.add(file_name)
            matches.append({
                "file": file_name,
                "matches": matched_lines[:10]  # Limit to first 10 matches per file
            })

    # Sort by number of matches (most relevant first)
    matches.sort(key=lambda x: len(x["matches"]), reverse=True)

    # Limit total results
    matches = matches[:20]

    return json.dumps({"keywords": keywords, "results": matches}, ensure_ascii=False)


def find_files_by_name_repo(patterns: List[str], instance_id: str) -> str:
    """
    Match files by one or more name/path patterns in the repository.
    Returns a deduplicated list of file paths matching **any** of the patterns.

    Parameters
    ----------
    patterns : list[str]
        Glob patterns to match file paths (e.g., ["**/models.py", "django/db/*.py"]).
    instance_id : str
        The instance ID for locating the repository.

    Returns
    -------
    str
        JSON string with matching file paths.
    """
    project_file_loc = os.environ.get("PROJECT_FILE_LOC", None)
    if project_file_loc is None:
        return json.dumps({"error": "PROJECT_FILE_LOC not set"}, ensure_ascii=False)

    from tools.utils.utils import load_json
    try:
        d = load_json(f"{project_file_loc}/{instance_id}.json")
    except Exception as e:
        return json.dumps({"error": f"Failed to load repo structure: {e}"}, ensure_ascii=False)

    structure = d.get("structure", {})

    from tools.RepoSearch.preprocess_data import extract_structure
    files, _, _ = extract_structure(structure)

    import fnmatch
    all_file_names = [f[0] for f in files]

    matched_files = []
    seen = set()
    for file_name in all_file_names:
        if _is_test_file(file_name + ":dummy"):
            continue
        if file_name in seen:
            continue
        for pattern in patterns:
            if fnmatch.fnmatch(file_name, pattern) or fnmatch.fnmatch(file_name, f"**/{pattern}"):
                matched_files.append(file_name)
                seen.add(file_name)
                break

    # Fallback: try matching against just the filename part
    if not matched_files:
        for file_name in all_file_names:
            if _is_test_file(file_name + ":dummy"):
                continue
            if file_name in seen:
                continue
            basename = os.path.basename(file_name)
            for pattern in patterns:
                if fnmatch.fnmatch(basename, pattern):
                    matched_files.append(file_name)
                    seen.add(file_name)
                    break

    return json.dumps({"patterns": patterns, "matched_files": matched_files}, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Wrapper functions with instance_id injected (for tool dispatch)
# ---------------------------------------------------------------------------

def get_code_of_function(file_name: str, func_name: str, instance_id: str) -> str:
    """Wrapper around get_code_of_file_function for the multi-agent framework."""
    return get_code_of_file_function(file_name, func_name, instance_id)


def get_code_of_class_method(file_name: str, class_name: str, func_name: str, instance_id: str) -> str:
    """Wrapper around get_code_of_class_function for the multi-agent framework."""
    return get_code_of_class_function(file_name, class_name, func_name, instance_id)
