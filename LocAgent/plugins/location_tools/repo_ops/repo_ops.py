import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))))
import pickle
import json
import os
import re
from collections import defaultdict
from typing import List, Optional
import collections
from copy import deepcopy
import uuid
import networkx as nx
import threading
import functools
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dependency_graph import RepoEntitySearcher, RepoDependencySearcher
from dependency_graph.build_graph import (
    build_graph,
    NODE_TYPE_DIRECTORY, NODE_TYPE_FILE, NODE_TYPE_CLASS, NODE_TYPE_FUNCTION,
    EDGE_TYPE_CONTAINS, # EDGE_TYPE_INHERITS, EDGE_TYPE_INVOKES, EDGE_TYPE_IMPORTS,
    VALID_NODE_TYPES, VALID_EDGE_TYPES
)
from dependency_graph.traverse_graph import (
    is_test_file, traverse_tree_structure,
    traverse_graph_structure, traverse_json_structure,
)
from plugins.location_tools.retriever.bm25_retriever import (
    build_code_retriever_from_repo as build_code_retriever,
    build_module_retriever_from_graph as build_module_retriever,
    build_retriever_from_persist_dir as load_retriever,
)
from plugins.location_tools.retriever.fuzzy_retriever import (
    fuzzy_retrieve_from_graph_nodes as fuzzy_retrieve
)
from plugins.location_tools.utils.result_format import QueryInfo, QueryResult
from plugins.location_tools.utils.util import (
    get_meta_data,
    find_matching_files_from_list,
    merge_intervals,
    GRAPH_INDEX_DIR,
    BM25_INDEX_DIR,
)
from util.benchmark.setup_repo import setup_repo
import subprocess
import logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# Default timeout values (in seconds), can be overridden via environment variables
DEFAULT_SEARCH_TIMEOUT = int(os.environ.get('REPO_OPS_SEARCH_TIMEOUT', '60'))
DEFAULT_GRAPH_TIMEOUT = int(os.environ.get('REPO_OPS_GRAPH_TIMEOUT', '120'))
DEFAULT_CONTEXT_TIMEOUT = int(os.environ.get('REPO_OPS_CONTEXT_TIMEOUT', '300'))


class RepoOpsTimeoutError(Exception):
    """Exception raised when a repository operation times out."""
    def __init__(self, operation_name: str, timeout_seconds: float, message: str = None):
        self.operation_name = operation_name
        self.timeout_seconds = timeout_seconds
        self.message = message or f"Operation '{operation_name}' timed out after {timeout_seconds} seconds"
        super().__init__(self.message)


def with_timeout(timeout_seconds: float = None, default_timeout_env: str = None, operation_name: str = None):
    """
    Decorator to add timeout functionality to long-running functions.

    Uses ThreadPoolExecutor for thread-safe timeout handling that works across platforms.

    Args:
        timeout_seconds: Fixed timeout in seconds. If None, uses default_timeout_env or DEFAULT_SEARCH_TIMEOUT.
        default_timeout_env: Environment variable name to read timeout from (e.g., 'REPO_OPS_SEARCH_TIMEOUT').
        operation_name: Name of the operation for error messages. Defaults to function name.

    Usage:
        @with_timeout(timeout_seconds=30)
        def my_function(...):
            ...

        @with_timeout(default_timeout_env='REPO_OPS_SEARCH_TIMEOUT')
        def search_function(...):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine timeout value
            if timeout_seconds is not None:
                timeout = timeout_seconds
            elif default_timeout_env:
                timeout = int(os.environ.get(default_timeout_env, DEFAULT_SEARCH_TIMEOUT))
            else:
                timeout = DEFAULT_SEARCH_TIMEOUT

            # Allow runtime override via kwargs
            timeout = kwargs.pop('_timeout', timeout)

            # Skip timeout if set to 0 or negative
            if timeout <= 0:
                return func(*args, **kwargs)

            op_name = operation_name or func.__name__

            # Use ThreadPoolExecutor for cross-platform timeout support
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timeout=timeout)
                    return result
                except FuturesTimeoutError:
                    logger.warning(f"[TIMEOUT] {op_name} exceeded {timeout}s timeout")
                    raise RepoOpsTimeoutError(op_name, timeout)

        return wrapper
    return decorator


# Thread-safe instance context storage
# Use OrderedDict to maintain insertion order for LRU eviction
_instance_contexts = collections.OrderedDict()
_context_lock = threading.Lock()
_current_instance_id: str | None = None

# Maximum number of contexts to keep in memory (controlled by env var, default 100)
MAX_INSTANCE_CONTEXTS = int(os.environ.get('MAX_INSTANCE_CONTEXTS', '100'))

class InstanceContext:
    """Context object to store instance-specific state"""
    def __init__(self, instance_id: str, instance_data: dict):
        self.instance_id = instance_id
        self.instance_data = instance_data
        self.all_file = None
        self.all_class = None
        self.all_func = None
        self.entity_searcher = None
        self.dependency_searcher = None
        self.graph = None
        self.repo_save_dir = None


def _get_current_instance_context(required: bool = True) -> InstanceContext | None:
    global _current_instance_id
    with _context_lock:
        if _current_instance_id is None:
            if required:
                raise RuntimeError("No current issue is set. Call set_current_issue(...) before using repo ops tools.")
            return None
        ctx = _instance_contexts.get(_current_instance_id)
        if ctx is None and required:
            raise RuntimeError(
                f"Current issue context '{_current_instance_id}' is unavailable. "
                "Call set_current_issue(...) again before using repo ops tools."
            )
        return ctx


def _evict_oldest_context_if_needed():
    """Evict half of the oldest contexts if we've reached the maximum limit.

    Must be called while holding _context_lock.
    """
    if len(_instance_contexts) < MAX_INSTANCE_CONTEXTS:
        return

    # Calculate how many contexts to evict (half of current count)
    num_to_evict = len(_instance_contexts) // 2
    if num_to_evict < 1:
        num_to_evict = 1

    logging.info(f'Evicting {num_to_evict} oldest contexts (current count: {len(_instance_contexts)}, max: {MAX_INSTANCE_CONTEXTS})')

    for _ in range(num_to_evict):
        if not _instance_contexts:
            break
        # Pop the oldest (first) item from OrderedDict
        oldest_id, oldest_ctx = _instance_contexts.popitem(last=False)
        logging.debug(f'Evicted context: {oldest_id}')

        # Cleanup repo directory
        if oldest_ctx.repo_save_dir and os.path.exists(oldest_ctx.repo_save_dir):
            try:
                subprocess.run(["rm", "-rf", oldest_ctx.repo_save_dir], check=True)
                logging.debug(f'Cleaned up repo directory: {oldest_ctx.repo_save_dir}')
            except Exception as e:
                logging.error(f"Failed to cleanup {oldest_ctx.repo_save_dir}: {e}")

    logging.info(f'Eviction complete, remaining contexts: {len(_instance_contexts)}')


def get_or_create_instance_context(instance_id: str, instance_data: dict = None,
                                   dataset: str = "princeton-nlp/SWE-bench_Lite",
                                   split: str = "test", rank=0) -> InstanceContext:
    """Get existing instance context or create a new one (thread-safe).

    The number of cached contexts is limited by the MAX_INSTANCE_CONTEXTS environment
    variable (default: 100). When the limit is reached, the oldest (least recently created)
    context is evicted to make room for new ones.
    """
    dataset = os.environ['DATASET_NAME']
    with _context_lock:
        if instance_id in _instance_contexts:
            # Move to end to mark as recently used (for potential future LRU enhancement)
            _instance_contexts.move_to_end(instance_id)
            return _instance_contexts[instance_id]

        # Evict oldest context if we've reached the limit
        _evict_oldest_context_if_needed()

        # Create new context
        if not instance_data:
            instance_data = get_meta_data(instance_id, dataset, split)

        ctx = InstanceContext(instance_id, instance_data)

        # Generate a temporary folder with uuid to avoid collision
        ctx.repo_save_dir = os.path.join('playground', f"{instance_id}_{uuid.uuid4()}")
        if not os.path.exists(ctx.repo_save_dir):
            os.makedirs(ctx.repo_save_dir)

        # Setup graph traverser
        graph_index_file = f"{GRAPH_INDEX_DIR}/{instance_id}.pkl"
        if not os.path.exists(graph_index_file):
            # Pull repo
            repo_dir = setup_repo(instance_data=instance_data, repo_base_dir=ctx.repo_save_dir, dataset=None)
            # Parse the repository
            try:
                os.makedirs(GRAPH_INDEX_DIR, exist_ok=True)
                G = build_graph(repo_dir, global_import=True)
                with open(graph_index_file, 'wb') as f:
                    pickle.dump(G, f)
                logging.info(f'[{rank}] Processed {instance_id}')
            except Exception as e:
                logging.error(f'[{rank}] Error processing {instance_id}: {e}')
                raise
        else:
            with open(graph_index_file, "rb") as f:
                G = pickle.load(f)

        ctx.entity_searcher = RepoEntitySearcher(G)
        ctx.dependency_searcher = RepoDependencySearcher(G)
        ctx.graph = G

        ctx.all_file = ctx.entity_searcher.get_all_nodes_by_type(NODE_TYPE_FILE)
        ctx.all_class = ctx.entity_searcher.get_all_nodes_by_type(NODE_TYPE_CLASS)
        ctx.all_func = ctx.entity_searcher.get_all_nodes_by_type(NODE_TYPE_FUNCTION)

        _instance_contexts[instance_id] = ctx
        logging.debug(f'Rank = {rank}, created context for instance_id = {instance_id} (total contexts: {len(_instance_contexts)})')

        return ctx


def reset_instance_context(instance_id: str):
    """Remove and cleanup an instance context (thread-safe)"""
    with _context_lock:
        if instance_id in _instance_contexts:
            ctx = _instance_contexts[instance_id]
            if ctx.repo_save_dir and os.path.exists(ctx.repo_save_dir):
                try:
                    subprocess.run(["rm", "-rf", ctx.repo_save_dir], check=True)
                    logging.debug(f'Cleaned up repo directory: {ctx.repo_save_dir}')
                except Exception as e:
                    logging.error(f"Failed to cleanup {ctx.repo_save_dir}: {e}")
            del _instance_contexts[instance_id]
            logging.debug(f'Removed context for instance_id = {instance_id}')


def set_current_issue(instance_id: str = None,
                      instance_data: dict = None,
                      dataset: str = "princeton-nlp/SWE-bench_Lite",
                      split: str = "test", rank=0):
    global _current_instance_id
    assert instance_id or instance_data
    current_instance_id = instance_id or instance_data['instance_id']
    ctx = get_or_create_instance_context(
        current_instance_id,
        instance_data=instance_data,
        dataset=dataset,
        split=split,
        rank=rank,
    )
    _current_instance_id = ctx.instance_id
    return ctx


def reset_current_issue():
    global _current_instance_id
    if _current_instance_id is None:
        return
    reset_instance_context(_current_instance_id)
    _current_instance_id = None


def get_current_issue_id():
    return _current_instance_id


def get_current_issue_data():
    ctx = _get_current_instance_context(required=False)
    return ctx.instance_data if ctx else None


def get_current_repo_modules():
    ctx = _get_current_instance_context()
    return ctx.all_file, ctx.all_class, ctx.all_func


def get_graph_entity_searcher() -> RepoEntitySearcher:
    ctx = _get_current_instance_context()
    return ctx.entity_searcher


def get_graph_dependency_searcher() -> RepoDependencySearcher:
    ctx = _get_current_instance_context()
    return ctx.dependency_searcher


def get_graph():
    ctx = _get_current_instance_context()
    return ctx.graph


def get_repo_save_dir():
    ctx = _get_current_instance_context(required=False)
    return ctx.repo_save_dir if ctx else None


def get_module_name_by_line_num(file_path: str, line_num: int, ctx: InstanceContext):
    entity_searcher = ctx.entity_searcher
    dp_searcher = ctx.dependency_searcher

    cur_module = None
    LARGE_CLASS_LINE_COUNT = 100
    if entity_searcher.has_node(file_path):
        module_nids, _ = dp_searcher.get_neighbors(file_path, etype_filter=[EDGE_TYPE_CONTAINS])
        module_ndatas = entity_searcher.get_node_data(module_nids)
        for module in module_ndatas:
            if module['start_line'] <= line_num <= module['end_line']:
                cur_module = module  # ['node_id']
                break
        if cur_module and cur_module['type'] == NODE_TYPE_CLASS:
            func_nids, _ = dp_searcher.get_neighbors(cur_module['node_id'], etype_filter=[EDGE_TYPE_CONTAINS])
            func_ndatas = entity_searcher.get_node_data(func_nids, return_code_content=False)
            for func in func_ndatas:
                if func['start_line'] <= line_num <= func['end_line']:
                    cur_module = func  # ['node_id']
                    break
            else:
                class_len = cur_module['end_line'] - cur_module['start_line']
                if class_len >= LARGE_CLASS_LINE_COUNT and func_ndatas:
                    nearest = [None, None]
                    nearest_dist = [float("inf"), float("inf")]
                    for func in func_ndatas:
                        start_line = func.get('start_line', 0)
                        end_line = func.get('end_line', start_line)
                        if line_num < start_line:
                            dist = start_line - line_num
                        elif line_num > end_line:
                            dist = line_num - end_line
                        else:
                            dist = 0
                        if dist < nearest_dist[0]:
                            nearest[1], nearest_dist[1] = nearest[0], nearest_dist[0]
                            nearest[0], nearest_dist[0] = func, dist
                        elif dist < nearest_dist[1]:
                            nearest[1], nearest_dist[1] = func, dist
                    nearest_funcs = [f for f in nearest if f]
                    if nearest_funcs:
                        cur_module = sorted(nearest_funcs, key=lambda f: f.get('start_line', 0))

    if cur_module: # and cur_module['type'] in [NODE_TYPE_CLASS, NODE_TYPE_FUNCTION]
        return cur_module
        # module_ndata = entity_searcher.get_node_data([cur_module['node_id']], return_code_content=True)
        # return module_ndata[0]
    return None


def get_code_block_by_line_nums(query_info, ctx: InstanceContext, context_window=20):
    # file_path: str, line_nums: List[int]
    searcher = ctx.entity_searcher
    
    file_path = query_info.file_path_or_pattern
    line_nums = query_info.line_nums
    cur_query_results = []
    
    file_data = searcher.get_node_data([file_path], return_code_content=False)[0]
    line_intervals = []
    res_modules = []
    # res_code_blocks = None
    for line in line_nums:
        # First determine which module contains this line
        module_data = get_module_name_by_line_num(file_path, line, ctx=ctx)

        # If not within any module, search ±20 lines
        if not module_data:
            min_line_num = max(1, line - context_window)
            max_line_num = min(file_data['end_line'], line + context_window)
            line_intervals.append((min_line_num, max_line_num))
            
        else:
            module_list = module_data if isinstance(module_data, list) else [module_data]
            for module in module_list:
                if module['node_id'] in res_modules:
                    continue
                query_result = QueryResult(query_info=query_info, format_mode='preview', 
                                           nid=module['node_id'],
                                           ntype=module['type'],
                                           start_line=module['start_line'],
                                           end_line=module['end_line'],
                                           retrieve_src=f"Retrieved code context including {query_info.term}."
                                           )
                cur_query_results.append(query_result)
                res_modules.append(module['node_id'])
            
    if line_intervals:
        line_intervals = merge_intervals(line_intervals)
        for interval in line_intervals:
            start_line, end_line = interval
            query_result = QueryResult(query_info=query_info, 
                                        format_mode='code_snippet',
                                        nid=file_path,
                                        file_path=file_path,
                                        start_line=start_line,
                                        end_line=end_line,
                                        retrieve_src=f"Retrieved code context including {query_info.term}."
                                        )
            cur_query_results.append(query_result)
        # res_code_blocks = line_wrap_content('\n'.join(file_content), line_intervals)

    # return res_code_blocks, res_modules
    return cur_query_results


def parse_node_id(nid: str):
    nfile = nid.split(':')[0]
    nname = nid.split(':')[-1]
    return nfile, nname


def search_entity_in_global_dict(term: str, ctx: InstanceContext, include_files: Optional[List[str]] = None, prefix_term=None):
    searcher = ctx.entity_searcher
    
    # TODO: hard code cases like "class Migration" and "function testing"
    if term.startswith(('class ', 'Class')):
        term = term[len('class '):].strip()
    elif term.startswith(('function ', 'Function ')):
        term = term[len('function '):].strip()
    elif term.startswith(('method ', 'Method ')):
        term = term[len('method '):].strip()
    elif term.startswith('def '):
        term = term[len('def '):].strip()
    
    # TODO: lower case if not find
    # TODO: filename xxx.py as key (also lowercase if not find)
    # global_name_dict = None
    if term in searcher.global_name_dict:
        global_name_dict = searcher.global_name_dict
        nids = global_name_dict[term]
    elif term.lower() in searcher.global_name_dict_lowercase:
        term = term.lower()
        global_name_dict = searcher.global_name_dict_lowercase
        nids = global_name_dict[term]
    else:
        return None
    
    node_datas = searcher.get_node_data(nids, return_code_content=False)
    found_entities_filter_dict = collections.defaultdict(list)
    for ndata in node_datas:
        nfile, _ = parse_node_id(ndata['node_id'])
        if not include_files or nfile in include_files:
            prefix_terms = []
            # candidite_prefixes = ndata['node_id'].lower().replace('.py', '').replace('/', '.').split('.')
            candidite_prefixes = re.split(r'[./:]', ndata['node_id'].lower().replace('.py', ''))[:-1]
            if prefix_term:
                prefix_terms = prefix_term.lower().split('.')
            if not prefix_term or all([prefix in candidite_prefixes for prefix in prefix_terms]):
                found_entities_filter_dict[ndata['type']].append(ndata['node_id'])

    return found_entities_filter_dict


def search_entity(query_info, ctx: InstanceContext, include_files: List[str] = None):
    term = query_info.term
    searcher = ctx.entity_searcher
    # cur_result = ''
    continue_search = True

    cur_query_results = []
    
    # first: exact match in graph
    if searcher.has_node(term):
        continue_search = False
        query_result = QueryResult(query_info=query_info, format_mode='complete', nid=term,
                                   retrieve_src=f"Exact match found for entity name `{term}`."
                                   )
        cur_query_results.append(query_result)
    
    # TODO: __init__ not exsit
    elif term.endswith('.__init__'):
        nid = term[:-(len('.__init__'))]
        if searcher.has_node(nid):
            continue_search = False
            node_data = searcher.get_node_data([nid], return_code_content=True)[0]
            query_result = QueryResult(query_info=query_info, format_mode='preview', 
                                    nid=nid, 
                                    ntype=node_data['type'],
                                    start_line=node_data['start_line'],
                                    end_line=node_data['end_line'],
                                    retrieve_src=f"Exact match found for entity name `{nid}`."
                                    )
            cur_query_results.append(query_result)
    
    # second: search in global name dict
    if continue_search:
        found_entities_dict = search_entity_in_global_dict(term, ctx, include_files=include_files)
        if not found_entities_dict:
            found_entities_dict = search_entity_in_global_dict(term, ctx)

        use_sub_term = False
        used_term = term
        if not found_entities_dict and '.' in term:
            # for cases: class_name.method_name
            try:
                prefix_term = '.'.join(term.split('.')[:-1]).split()[-1] # incase of 'class '/ 'function '
            except IndexError:
                prefix_term = None
            split_term = term.split('.')[-1].strip()
            used_term = split_term
            found_entities_dict = search_entity_in_global_dict(split_term, ctx, include_files=include_files, prefix_term=prefix_term)
            if not found_entities_dict:
                found_entities_dict = search_entity_in_global_dict(split_term, ctx, prefix_term=prefix_term)
            if not found_entities_dict:
                use_sub_term = True
                found_entities_dict = search_entity_in_global_dict(split_term, ctx)
        
        # TODO: split the term and find in global dict
            
        if found_entities_dict:
            for ntype, nids in found_entities_dict.items():
                if not nids: continue
                # if not continue_search: break

                # procee class and function in the same way
                if ntype in [NODE_TYPE_FUNCTION, NODE_TYPE_CLASS, NODE_TYPE_FILE]:
                    if len(nids) <= 3:
                        node_datas = searcher.get_node_data(nids, return_code_content=True)
                        for ndata in node_datas:
                            query_result = QueryResult(query_info=query_info, format_mode='preview', 
                                                       nid=ndata['node_id'], 
                                                       ntype=ndata['type'],
                                                       start_line=ndata['start_line'],
                                                       end_line=ndata['end_line'],
                                                       retrieve_src=f"Match found for entity name `{used_term}`."
                                                       )
                            cur_query_results.append(query_result)
                        # continue_search = False
                    else:
                        node_datas = searcher.get_node_data(nids, return_code_content=False)
                        for ndata in node_datas:
                            query_result = QueryResult(query_info=query_info, format_mode='fold', 
                                                       nid=ndata['node_id'],
                                                       ntype=ndata['type'],
                                                       retrieve_src=f"Match found for entity name `{used_term}`."
                                                       )
                            cur_query_results.append(query_result)
                    if not use_sub_term:
                        continue_search = False
                    else:
                        continue_search = True
                                   
        
    # third: bm25 search (entity + content)
    if continue_search:
        module_nids = []

        # append the file name to keyword?
        # # if not any(symbol in file_path_or_pattern for symbol in ['*','?', '[', ']']):
        # term_with_file = f'{file_path_or_pattern}:{term}'
        # module_nids = bm25_module_retrieve(query=term_with_file, include_files=include_files)

        # search entity by keyword
        module_nids = bm25_module_retrieve(query=term, include_files=include_files, ctx=ctx)
        if not module_nids:
            module_nids = bm25_module_retrieve(query=term, ctx=ctx)
        if not module_nids:
            # result += f"No entity found using BM25 search. Try to use fuzzy search...\n"
            graph = ctx.graph
            module_nids = fuzzy_retrieve(term, graph=graph, similarity_top_k=3)

        module_datas = searcher.get_node_data(module_nids, return_code_content=True)
        showed_module_num = 0
        for module in module_datas[:5]:
            if module['type'] in [NODE_TYPE_FILE, NODE_TYPE_DIRECTORY]:
                query_result = QueryResult(query_info=query_info, format_mode='fold', 
                                        nid=module['node_id'],
                                        ntype=module['type'],
                                        retrieve_src=f"Retrieved entity using keyword search (bm25)."
                                        )
                cur_query_results.append(query_result)
            elif showed_module_num < 3:
                showed_module_num += 1
                query_result = QueryResult(query_info=query_info, format_mode='preview', 
                                        nid=module['node_id'],
                                        ntype=module['type'],
                                        start_line=module['start_line'],
                                            end_line=module['end_line'],
                                            retrieve_src=f"Retrieved entity using keyword search (bm25)."
                                        )
                cur_query_results.append(query_result)

    return (cur_query_results, continue_search)


def merge_query_results(query_results):
    priority = ['complete', 'code_snippet', 'preview', 'fold']
    merged_results = {}
    all_query_results: List[QueryResult] = []

    for qr in query_results:
        if qr.format_mode == 'code_snippet':
            all_query_results.append(qr)
        
        elif qr.nid and qr.nid in merged_results:
            # Merge query_info_list
            if qr.query_info_list[0] not in merged_results[qr.nid].query_info_list:
                merged_results[qr.nid].query_info_list.extend(qr.query_info_list)

            # Select the format_mode with the highest priority
            existing_format_mode = merged_results[qr.nid].format_mode
            if priority.index(qr.format_mode) < priority.index(existing_format_mode):
                merged_results[qr.nid].format_mode = qr.format_mode
                merged_results[qr.nid].start_line = qr.start_line
                merged_results[qr.nid].end_line = qr.end_line
                merged_results[qr.nid].retrieve_src = qr.retrieve_src
                
        elif qr.nid:
            merged_results[qr.nid] = qr
    
    all_query_results += list(merged_results.values())
    return all_query_results


def rank_and_aggr_query_results(query_results, fixed_query_info_list):
    query_info_list_dict = {}

    for qr in query_results:
        # Convert the query_info_list to a tuple so it can be used as a dictionary key
        key = tuple(qr.query_info_list)

        if key in query_info_list_dict:
            query_info_list_dict[key].append(qr)
        else:
            query_info_list_dict[key] = [qr]
            
    # for the key: sort by query
    def sorting_key(key):
        # Find the first matching element index from fixed_query_info_list in the key (tuple of query_info_list)
        for i, fixed_query in enumerate(fixed_query_info_list):
            if fixed_query in key:
                return i
        # If no match is found, assign a large index to push it to the end
        return len(fixed_query_info_list)

    sorted_keys = sorted(query_info_list_dict.keys(), key=sorting_key)
    sorted_query_info_list_dict = {key: query_info_list_dict[key] for key in sorted_keys}
    
    # for the value: sort by format priority
    priority = {'complete': 1, 'code_snippet': 2, 'preview': 3,  'fold': 4}  # Lower value indicates higher priority
    # TODO: merge the same node in 'code_snippet' and 'preview'
    
    organized_dict = {}
    for key, values in sorted_query_info_list_dict.items():
        nested_dict = {priority_key: [] for priority_key in priority.keys()}
        for qr in values:
            # Place the qr in the nested dictionary based on its format_mode
            if qr.format_mode in nested_dict:
                nested_dict[qr.format_mode].append(qr)

        # Only add keys with non-empty lists to keep the result clean
        organized_dict[key] = {k: v for k, v in nested_dict.items() if v}
    
    return organized_dict
        

@with_timeout(default_timeout_env='REPO_OPS_SEARCH_TIMEOUT', operation_name='search_code_snippets')
def search_code_snippets(
        ctx: InstanceContext = None,
        search_terms: Optional[List[str]] = None,
        line_nums: Optional[List] = None,
        file_path_or_pattern: Optional[str] = "**/*.py",
) -> str:
    """Searches the codebase to retrieve relevant code snippets based on given queries(terms or line numbers).

    This function supports retrieving the complete content of a code entity,
    searching for code entities such as classes or functions by keywords, or locating specific lines within a file.
    It also supports filtering searches based on a file path or file pattern.

    Note:
    1. If `search_terms` are provided, it searches for code snippets based on each term:
        - If a term is formatted as 'file_path:QualifiedName' (e.g., 'src/helpers/math_helpers.py:MathUtils.calculate_sum') ,
          or just 'file_path', the corresponding complete code is retrieved or file content is retrieved.
        - If a term matches a file, class, or function name, matched entities are retrieved.
        - If there is no match with any module name, it attempts to find code snippets that likely contain the term.

    2. If `line_nums` is provided, it searches for code snippets at the specified lines within the file defined by
       `file_path_or_pattern`.

    Args:
        search_terms (Optional[List[str]]): A list of names, keywords, or code snippets to search for within the codebase.
            Terms can be formatted as 'file_path:QualifiedName' to search for a specific module or entity within a file
            (e.g., 'src/helpers/math_helpers.py:MathUtils.calculate_sum') or as 'file_path' to retrieve the complete content
            of a file. This can also include potential function names, class names, or general code fragments.

        line_nums (Optional[List[int]]): Specific line numbers to locate code snippets within a specified file.
            When provided, `file_path_or_pattern` must specify a valid file path.

        file_path_or_pattern (Optional[str]): A glob pattern or specific file path used to filter search results
            to particular files or directories. Defaults to '**/*.py', meaning all Python files are searched by default.
            If `line_nums` are provided, this must specify a specific file path.

        instance_id (Optional[str]): The instance ID for thread-safe context lookup. If provided, uses
            InstanceContext instead of global variables to ensure thread safety.

    Returns:
        str: The search results, which may include code snippets, matching entities, or complete file content.


    Example Usage:
        # Search for the full content of a specific file
        result = search_code_snippets(search_terms=['src/my_file.py'])

        # Search for a specific function
        result = search_code_snippets(search_terms=['src/my_file.py:MyClass.func_name'])

        # Search for specific lines (10 and 15) within a file
        result = search_code_snippets(line_nums=[10, 15], file_path_or_pattern='src/example.py')

        # Combined search for a module name and within a specific file pattern
        result = search_code_snippets(search_terms=["MyClass"], file_path_or_pattern="src/**/*.py")
    """
    import time
    if ctx is None:
        ctx = _get_current_instance_context()

    func_start_time = time.time()
    logger.debug(f"[TIMING] search_code_snippets started - search_terms={search_terms}, line_nums={line_nums}, pattern={file_path_or_pattern}")

    # Step 1: Get repo modules
    step1_start = time.time()
    files = ctx.all_file
    all_file_paths = [file['name'] for file in files]
    step1_elapsed = time.time() - step1_start
    logger.debug(f"[TIMING] Step 1 - get_current_repo_modules: {step1_elapsed:.3f}s, found {len(all_file_paths)} files")

    result = ""
    # exclude_files = find_matching_files_from_list(all_file_paths, "**/test*/**")

    # Step 2: Filter files by pattern
    step2_start = time.time()
    if file_path_or_pattern:
        include_files = find_matching_files_from_list(all_file_paths, file_path_or_pattern)
        if not include_files:
            include_files = all_file_paths
            result += f"No files found for file pattern '{file_path_or_pattern}'. Will search all files.\n...\n"
    else:
        include_files = all_file_paths
    step2_elapsed = time.time() - step2_start
    logger.debug(f"[TIMING] Step 2 - find_matching_files: {step2_elapsed:.3f}s, matched {len(include_files)} files")

    query_info_list = []
    all_query_results = []

    # Step 3: Process search terms
    if search_terms:
        step3_start = time.time()
        logger.debug(f"[TIMING] Step 3 - Processing {len(search_terms)} search terms")

        # search all terms together
        filter_terms = []
        for term in search_terms:
            if is_test_file(term):
                result += f'No results for test files: `{term}`. Please do not search for any test files.\n\n'
            else:
                filter_terms.append(term)

        joint_terms = deepcopy(filter_terms)
        if len(filter_terms) > 1:
            filter_terms.append(' '.join(filter_terms))

        for i, term in enumerate(filter_terms):
            term_start = time.time()
            term = term.strip().strip('.')
            if not term: continue

            query_info = QueryInfo(term=term)
            query_info_list.append(query_info)

            cur_query_results = []

            # Step 3a: search entity
            entity_search_start = time.time()
            query_results, continue_search = search_entity(query_info, ctx, include_files=include_files)
            entity_search_elapsed = time.time() - entity_search_start
            logger.debug(f"[TIMING] Step 3a - search_entity for term '{term}': {entity_search_elapsed:.3f}s, found {len(query_results)} results")
            cur_query_results.extend(query_results)

            # Step 3b: search content
            if continue_search:
                content_search_start = time.time()
                query_results = bm25_content_retrieve(query_info=query_info, include_files=include_files, ctx=ctx)
                content_search_elapsed = time.time() - content_search_start
                logger.debug(f"[TIMING] Step 3b - bm25_content_retrieve for term '{term}': {content_search_elapsed:.3f}s, found {len(query_results)} results")
                cur_query_results.extend(query_results)

            elif i != (len(filter_terms)-1):
                joint_terms[i] = ''
                filter_terms[-1] = ' '.join([t for t in joint_terms if t.strip()])
                if filter_terms[-1] in filter_terms[:-1]:
                    filter_terms[-1] = ''

            term_elapsed = time.time() - term_start
            logger.debug(f"[TIMING] Term '{term}' processing completed: {term_elapsed:.3f}s, total results: {len(cur_query_results)}")
            all_query_results.extend(cur_query_results)

        step3_elapsed = time.time() - step3_start
        logger.debug(f"[TIMING] Step 3 - All search terms processed: {step3_elapsed:.3f}s, total results: {len(all_query_results)}")

    # Step 4: Process line numbers
    if file_path_or_pattern in all_file_paths and line_nums:
        step4_start = time.time()
        if isinstance(line_nums, int):
            line_nums = [line_nums]
        file_path = file_path_or_pattern
        term = file_path + ':line ' + ', '.join([str(line) for line in line_nums])
        # result += f"Search `line(s) {line_nums}` in file `{file_path}` ...\n"
        query_info = QueryInfo(term=term, line_nums=line_nums, file_path_or_pattern=file_path)

        # Search for codes based on file name and line number
        query_results = get_code_block_by_line_nums(query_info, ctx=ctx)
        all_query_results.extend(query_results)
        step4_elapsed = time.time() - step4_start
        logger.debug(f"[TIMING] Step 4 - get_code_block_by_line_nums: {step4_elapsed:.3f}s, found {len(query_results)} results")

    # Step 5: Merge and rank results
    step5_start = time.time()
    merged_results = merge_query_results(all_query_results)
    merge_elapsed = time.time() - step5_start
    logger.debug(f"[TIMING] Step 5a - merge_query_results: {merge_elapsed:.3f}s, merged to {len(merged_results)} results")

    rank_start = time.time()
    ranked_query_to_results = rank_and_aggr_query_results(merged_results, query_info_list)
    rank_elapsed = time.time() - rank_start
    logger.debug(f"[TIMING] Step 5b - rank_and_aggr_query_results: {rank_elapsed:.3f}s")

    step5_elapsed = time.time() - step5_start
    logger.debug(f"[TIMING] Step 5 - Merge and rank: {step5_elapsed:.3f}s")

    # Step 6: Format output
    step6_start = time.time()
    # format_mode: 'complete', 'preview', 'code_snippet', 'fold': 4
    searcher = ctx.entity_searcher
    
    for query_infos, format_to_results in ranked_query_to_results.items():
        term_desc = ', '.join([f'"{query.term}"' for query in query_infos])
        result += f'##Searching for term {term_desc}...\n'
        result += f'### Search Result:\n'
        cur_result = ''
        for format_mode, query_results in format_to_results.items():
            if format_mode == 'fold':
                cur_retrieve_src = ''
                for qr in query_results:
                    if not cur_retrieve_src:
                        cur_retrieve_src = qr.retrieve_src
                        
                    if cur_retrieve_src != qr.retrieve_src:
                        cur_result += "Source: " + cur_retrieve_src + '\n\n'
                        cur_retrieve_src = qr.retrieve_src
                        
                    cur_result += qr.format_output(searcher)
                    
                cur_result += "Source: " + cur_retrieve_src + '\n'
                if len(query_results) > 1:
                    cur_result += 'Hint: Use more detailed query to get the full content of some if needed.\n'
                else:
                    cur_result += f'Hint: Search `{query_results[0].nid}` for the full content if needed.\n'
                cur_result += '\n'
                
            elif format_mode == 'complete':
                for qr in query_results:
                    cur_result += qr.format_output(searcher)
                    cur_result += '\n'

            elif format_mode == 'preview':
                # Remove the small modules, leaving only the large ones
                filtered_results = []
                grouped_by_file = defaultdict(list)
                for qr in query_results:
                    if (qr.end_line - qr.start_line) < 100:
                        grouped_by_file[qr.file_path].append(qr)
                    else:
                        filtered_results.append(qr)
                
                for file_path, results in grouped_by_file.items():
                    # Sort by start_line and then by end_line in descending order
                    sorted_results = sorted(results, key=lambda qr: (qr.start_line, -qr.end_line))

                    max_end_line = -1
                    for qr in sorted_results:
                        # If the current QueryResult's range is not completely covered by the largest range seen so far, keep it
                        if qr.end_line > max_end_line:
                            filtered_results.append(qr)
                            max_end_line = max(max_end_line, qr.end_line)
                
                # filtered_results = query_results
                for qr in filtered_results:
                    cur_result += qr.format_output(searcher)
                    cur_result += '\n'
            
            elif format_mode == 'code_snippet':
                for qr in query_results:
                    cur_result += qr.format_output(searcher)
                    cur_result += '\n'

        cur_result += '\n\n'

        if cur_result.strip():
            result += cur_result
        else:
            result += 'No locations found.\n\n'

    step6_elapsed = time.time() - step6_start
    logger.debug(f"[TIMING] Step 6 - Format output: {step6_elapsed:.3f}s, output length: {len(result)} chars")

    # Total time
    total_elapsed = time.time() - func_start_time
    logger.debug(f"[TIMING] search_code_snippets COMPLETED in {total_elapsed:.3f}s")

    return result.strip()


@with_timeout(default_timeout_env='REPO_OPS_SEARCH_TIMEOUT', operation_name='get_entity_contents')
def get_entity_contents(entity_names: List[str], ctx: InstanceContext = None):
    import time
    if ctx is None:
        ctx = _get_current_instance_context()

    func_start_time = time.time()
    logger.debug(f"[TIMING] get_entity_contents started - entity_names={entity_names}")

    searcher_start = time.time()
    searcher = ctx.entity_searcher
    searcher_elapsed = time.time() - searcher_start
    logger.debug(f"[TIMING] get_graph_entity_searcher: {searcher_elapsed:.3f}s")

    result = ''
    for idx, name in enumerate(entity_names):
        entity_start = time.time()
        name = name.strip().strip('.')
        if not name: continue

        result += f'##Searching for entity `{name}`...\n'
        result += f'### Search Result:\n'
        query_info = QueryInfo(term=name)

        lookup_start = time.time()
        has_node = searcher.has_node(name)
        lookup_elapsed = time.time() - lookup_start

        if has_node:
            format_start = time.time()
            query_result = QueryResult(query_info=query_info, format_mode='complete', nid=name,
                                    retrieve_src=f"Exact match found for entity name `{name}`."
                                    )
            result += query_result.format_output(searcher)
            result += '\n\n'
            format_elapsed = time.time() - format_start
            logger.debug(f"[TIMING] Entity '{name}' found and formatted: lookup={lookup_elapsed:.3f}s, format={format_elapsed:.3f}s")
        else:
            result += 'Invalid name. \nHint: Valid entity name should be formatted as "file_path:QualifiedName" or just "file_path".'
            result += '\n\n'
            logger.debug(f"[TIMING] Entity '{name}' not found: lookup={lookup_elapsed:.3f}s")

        entity_elapsed = time.time() - entity_start
        logger.debug(f"[TIMING] Entity {idx+1}/{len(entity_names)} '{name}' processed: {entity_elapsed:.3f}s")

    total_elapsed = time.time() - func_start_time
    logger.debug(f"[TIMING] get_entity_contents COMPLETED in {total_elapsed:.3f}s")

    return result.strip()


def bm25_module_retrieve(
        query: str,
        ctx: InstanceContext,
        include_files: Optional[List[str]] = None,
        # file_pattern: Optional[str] = None,
        search_scope: str = 'all',
        similarity_top_k: int = 10,
        # sort_by_type = False
):
    # Check if query is valid (non-empty after stripping)
    if not query or not query.strip():
        logger.warning(f"[BM25] Empty or invalid query provided: '{query}'. Skipping BM25 module retrieval.")
        return []

    entity_searcher = ctx.entity_searcher
    retriever = build_module_retriever(entity_searcher=entity_searcher,
                                       search_scope=search_scope,
                                       similarity_top_k=similarity_top_k)
    try:
        retrieved_nodes = retriever.retrieve(query)
    except IndexError as e:
        logger.warning(f"[BM25] Query '{query}' could not be tokenized properly: {e}. Try using different search terms.")
        return []

    filter_nodes = []
    all_nodes = []
    for node in retrieved_nodes:
        if node.score <= 0:
            continue
        if not include_files or node.text.split(':')[0] in include_files:
            filter_nodes.append(node.text)
        all_nodes.append(node.text)

    if filter_nodes:
        return filter_nodes
    else:
        return all_nodes


def bm25_content_retrieve(
        query_info: QueryInfo,
        ctx: InstanceContext,
        # query: str,
        include_files: Optional[List[str]] = None,
        # file_pattern: Optional[str] = None,
        similarity_top_k: int = 10
) -> str:
    """Retrieves code snippets from the codebase using the BM25 algorithm based on the provided query, class names, and function names. This function helps in finding relevant code sections that match specific criteria, aiding in code analysis and understanding.

    Args:
        query (Optional[str]): A textual query to search for relevant code snippets. Defaults to an empty string if not provided.
        class_names (list[str]): A list of class names to include in the search query. If None, class names are not included.
        function_names (list[str]): A list of function names to include in the search query. If None, function names are not included.
        file_pattern (Optional[str]): A glob pattern to filter search results to specific file types or directories. If None, the search includes all files.
        similarity_top_k (int): The number of top similar documents to retrieve based on the BM25 ranking. Defaults to 15.
        ctx (InstanceContext): Thread-safe context object. If provided, uses context instead of global variables.

    Returns:
        str: A formatted string containing the search results, including file paths and the retrieved code snippets (the partial code of a module or the skeleton of the specific module).
    """

    instance = ctx.instance_data
    repo_playground = ctx.repo_save_dir
    query = query_info.term

    persist_path = os.path.join(BM25_INDEX_DIR, ctx.instance_id)
    if os.path.exists(f'{persist_path}/corpus.jsonl'):
        # TODO: if similairy_top_k > cache's setting, then regenerate
        retriever = load_retriever(persist_path)
    else:
        repo_dir = setup_repo(instance_data=instance, repo_base_dir=repo_playground, dataset=None, split=None)
        absolute_repo_dir = os.path.abspath(repo_dir)
        retriever = build_code_retriever(absolute_repo_dir, persist_path=persist_path,
                                         similarity_top_k=similarity_top_k)

    # similarity: {score}
    cur_query_results = []

    # Check if query is valid (non-empty after stripping)
    # BM25 tokenizer returns empty list for empty/whitespace-only queries,
    # which causes IndexError in bm25s library
    if not query or not query.strip():
        error_msg = f"Empty or invalid query provided: '{query}'. Please provide a non-empty search term."
        logger.warning(f"[BM25] {error_msg}")
        error_result = QueryResult(
            query_info=query_info,
            format_mode='fold',
            nid=None,
            retrieve_src=f"Error: {error_msg}"
        )
        return [error_result]

    try:
        retrieved_nodes = retriever.retrieve(query)
    except IndexError as e:
        # Handle case where query tokens are empty after tokenization
        error_msg = f"Query '{query}' could not be tokenized properly. Try using different search terms."
        logger.warning(f"[BM25] {error_msg}: {e}")
        error_result = QueryResult(
            query_info=query_info,
            format_mode='fold',
            nid=None,
            retrieve_src=f"Error: {error_msg}"
        )
        return [error_result]
    for node in retrieved_nodes:
        file = node.metadata['file_path']
        # print(node.metadata)
        if not include_files or file in include_files:
            # drop the import code
            # if len(node.metadata['span_ids']) == 1 and node.metadata['span_ids'][0] == 'imports':
            #     continue
            if all([span_id in ['docstring', 'imports', 'comments'] for span_id in node.metadata['span_ids']]):
                # TODO: drop ?
                query_result = QueryResult(query_info=query_info, 
                                           format_mode='code_snippet',
                                           nid=node.metadata['file_path'],
                                           file_path=node.metadata['file_path'],
                                           start_line=node.metadata['start_line'],
                                           end_line=node.metadata['end_line'],
                                           retrieve_src=f"Retrieved code content using keyword search (bm25)."
                                           )
                cur_query_results.append(query_result)
                
            elif any([span_id in ['docstring', 'imports', 'comments'] for span_id in node.metadata['span_ids']]):
                nids = []
                searcher = ctx.entity_searcher
                for span_id in node.metadata['span_ids']:
                    nid = f'{file}:{span_id}'
                    if searcher.has_node(nid):
                        nids.append(nid)
                    # TODO: warning if not find

                node_datas = searcher.get_node_data(nids, return_code_content=True)
                sorted_ndatas = sorted(node_datas, key=lambda x: x['start_line'])
                sorted_nids = [ndata['node_id'] for ndata in sorted_ndatas]
                
                message = ''
                if sorted_nids:
                    if sorted_ndatas[0]['start_line'] < node.metadata['start_line']:
                        nid = sorted_ndatas[0]['node_id']
                        ntype = sorted_ndatas[0]['type']
                        # The code for {ntype} {nid} is incomplete; search {nid} for the full content if needed.
                        message += f"The code for {ntype} `{nid}` is incomplete; search `{nid}` for the full content if needed.\n"
                    if sorted_ndatas[-1]['end_line'] > node.metadata['end_line']:
                        nid = sorted_ndatas[-1]['node_id']
                        ntype = sorted_ndatas[-1]['type']
                        message += f"The code for {ntype} `{nid}` is incomplete; search `{nid}` for the full content if needed.\n"
                    if message.strip():
                        message = "Hint: \n"+ message
                
                nids_str = ', '.join([f'`{nid}`' for nid in sorted_nids])
                desc = f"Found {nids_str}."
                query_result = QueryResult(query_info=query_info, 
                                           format_mode='code_snippet',
                                           nid=node.metadata['file_path'],
                                           file_path=node.metadata['file_path'],
                                           start_line=node.metadata['start_line'],
                                           end_line=node.metadata['end_line'],
                                           desc=desc,
                                           message=message,
                                           retrieve_src=f"Retrieved code content using keyword search (bm25)."
                                           )
                
                cur_query_results.append(query_result)
            else:
                searcher = ctx.entity_searcher
                for span_id in node.metadata['span_ids']:
                    nid = f'{file}:{span_id}'
                    print(nid)
                    if searcher.has_node(nid):
                        ndata = searcher.get_node_data([nid], return_code_content=True)[0]
                        query_result = QueryResult(query_info=query_info, format_mode='preview', 
                                                   nid=ndata['node_id'],
                                                   ntype=ndata['type'],
                                                   start_line=ndata['start_line'],
                                                   end_line=ndata['end_line'],
                                                   retrieve_src=f"Retrieved code content using keyword search (bm25)."
                                                   )
                        cur_query_results.append(query_result)
                    else:
                        continue
        
    cur_query_results = cur_query_results[:5]
    return cur_query_results


def _validate_graph_explorer_inputs(
        start_entities: List[str],
        ctx: InstanceContext,
        direction: str = 'downstream',
        traversal_depth: int = 1,
        node_type_filter: Optional[List[str]] = None,
        edge_type_filter: Optional[List[str]] = None,
):
    """evaluate input arguments
    """

    hints = ''

    # Validate direction
    if direction not in ['downstream', 'upstream', 'both']:
        hints += f"Invalid value for `direction`: Expected one of 'downstream', 'upstream', 'both'. Received: '{direction}'. Using default 'downstream'.\n\n"
        direction = 'downstream'

    # Validate traversal_depth
    if traversal_depth != -1 and traversal_depth < 0:
        hints += f"Invalid value for `traversal_depth`: It must be either -1 or a non-negative integer (>= 0). Received: {traversal_depth}. Using default 1.\n\n"
        traversal_depth = 1

    # Validate node_type_filter
    valid_node_type_filter = None
    if isinstance(node_type_filter, list):
        invalid_ntypes = []
        valid_ntypes = []
        for ntype in node_type_filter:
            if ntype not in VALID_NODE_TYPES:
                invalid_ntypes.append(ntype)
            else:
                valid_ntypes.append(ntype)
        if invalid_ntypes:
            hints += f"Invalid node types {invalid_ntypes} in entity_type_filter. Valid types are: {list(VALID_NODE_TYPES)}.\n"
            if valid_ntypes:
                hints += f"Using valid types only: {valid_ntypes}.\n\n"
                valid_node_type_filter = valid_ntypes
            else:
                hints += "No valid entity types provided. Will include all entity types.\n\n"
        else:
            valid_node_type_filter = node_type_filter

    # Validate edge_type_filter
    valid_edge_type_filter = None
    if isinstance(edge_type_filter, list):
        invalid_etypes = []
        valid_etypes = []
        for etype in edge_type_filter:
            if etype not in VALID_EDGE_TYPES:
                invalid_etypes.append(etype)
            else:
                valid_etypes.append(etype)
        if invalid_etypes:
            hints += f"Invalid edge types {invalid_etypes} in dependency_type_filter. Valid types are: {list(VALID_EDGE_TYPES)}.\n"
            if valid_etypes:
                hints += f"Using valid types only: {valid_etypes}.\n\n"
                valid_edge_type_filter = valid_etypes
            else:
                hints += "No valid dependency types provided. Will include all dependency types.\n\n"
        else:
            valid_edge_type_filter = edge_type_filter

    graph = ctx.graph
    entity_searcher = ctx.entity_searcher

    valid_entities = []
    for i, root in enumerate(start_entities):
        # process node name
        if root != '/':
            root = root.strip('/')
        if root.endswith('.__init__'):
            root = root[:-(len('.__init__'))]

        # validate node name
        if root not in graph:
            # search with bm25
            module_nids = bm25_module_retrieve(query=root, ctx=ctx)
            module_datas = entity_searcher.get_node_data(module_nids, return_code_content=False)
            if len(module_datas) > 0:
                hints += f'The entity name `{root}` is invalid. Based on your input, here are some candidate entities you might be referring to:\n'
                for module in module_datas[:5]:
                    ntype = module['type']
                    nid = module['node_id']
                    hints += f'{ntype}: `{nid}`\n'
                hints += "Source: Retrieved entity using keyword search (bm25).\n\n"
            else:
                hints += f'The entity name `{root}` is invalid. There are no possible candidate entities in record.\n'
        elif is_test_file(root):
            hints += f'No results for the test entity: `{root}`. Please do not include any test entities.\n\n'
        else:
            valid_entities.append(root)

    return valid_entities, hints, valid_node_type_filter, valid_edge_type_filter

@with_timeout(default_timeout_env='REPO_OPS_GRAPH_TIMEOUT', operation_name='explore_tree_structure')
def explore_tree_structure(
        start_entities: List[str],
        ctx: InstanceContext = None,
        direction: str = 'downstream',
        traversal_depth: int = 2,
        entity_type_filter: Optional[List[str]] = None,
        dependency_type_filter: Optional[List[str]] = None,
):
    """Analyzes and displays the dependency structure around specified entities in a code graph.

    This function searches and presents relationships and dependencies for the specified entities (such as classes, functions, files, or directories) in a code graph.
    It explores how the input entities relate to others, using defined types of dependencies, including 'contains', 'imports', 'invokes' and 'inherits'.
    The search can be controlled to traverse upstream (exploring dependencies that entities rely on) or downstream (exploring how entities impact others), with optional limits on traversal depth and filters for entity and dependency types.

    Example Usage:
    1. Exploring Outward Dependencies:
        ```
        get_local_structure(
            start_entities=['src/module_a.py:ClassA'],
            direction='downstream',
            traversal_depth=2,
            entity_type_filter=['class', 'function'],
            dependency_type_filter=['invokes', 'imports']
        )
        ```
        This retrieves the dependencies of `ClassA` up to 2 levels deep, focusing only on classes and functions with 'invokes' and 'imports' relationships.

    2. Exploring Inward Dependencies:
        ```
        get_local_structure(
            start_entities=['src/module_b.py:FunctionY'],
            direction='upstream',
            traversal_depth=-1
        )
        ```
        This finds all entities that depend on `FunctionY` without restricting the traversal depth.

    Notes:
    * Traversal Control: The `traversal_depth` parameter specifies how deep the function should explore the graph starting from the input entities.
    * Filtering: Use `entity_type_filter` and `dependency_type_filter` to narrow down the scope of the search, focusing on specific entity types and relationships.
    * Graph Context: The function operates on a pre-built code graph containing entities (e.g., files, classes and functions) and dependencies representing their interactions and relationships.

    Parameters:
    ----------
    start_entities : list[str]
        List of entities (e.g., class, function, file, or directory paths) to begin the search from.
        - Entities representing classes or functions must be formatted as "file_path:QualifiedName"
          (e.g., `interface/C.py:C.method_a.inner_func`).
        - For files or directories, provide only the file or directory path (e.g., `src/module_a.py` or `src/`).

    ctx : InstanceContext
        Thread-safe context object containing instance-specific state.

    direction : str, optional
        Direction of traversal in the code graph; allowed options are:
        - 'upstream': Traversal to explore dependencies that the specified entities rely on (how they depend on others).
        - 'downstream': Traversal to explore the effects or interactions of the specified entities on others
          (how others depend on them).
        - 'both': Traversal in both directions.
        Default is 'downstream'.

    traversal_depth : int, optional
        Maximum depth of traversal. A value of -1 indicates unlimited depth (subject to a maximum limit).
        Must be either `-1` or a non-negative integer (≥ 0).
        Default is 2.

    entity_type_filter : list[str], optional
        List of entity types (e.g., 'class', 'function', 'file', 'directory') to include in the traversal.
        If None, all entity types are included.
        Default is None.

    dependency_type_filter : list[str], optional
        List of dependency types (e.g., 'contains', 'imports', 'invokes', 'inherits') to include in the traversal.
        If None, all dependency types are included.
        Default is None.

    Returns:
    -------
    result : object
        An object representing the traversal results, which includes discovered entities and their dependencies.
    """
    import time
    if ctx is None:
        ctx = _get_current_instance_context()

    func_start_time = time.time()
    logger.debug(f"[TIMING] explore_tree_structure started - start_entities={start_entities}, direction={direction}, depth={traversal_depth}")

    # Step 1: Validate inputs
    validate_start = time.time()
    start_entities, hints, valid_entity_type_filter, valid_dependency_type_filter = _validate_graph_explorer_inputs(
        start_entities, ctx, direction, traversal_depth,
        entity_type_filter, dependency_type_filter)
    validate_elapsed = time.time() - validate_start
    logger.debug(f"[TIMING] _validate_graph_explorer_inputs: {validate_elapsed:.3f}s, valid_entities={len(start_entities)}")

    # Step 2: Get graph
    graph_start = time.time()
    G = ctx.graph
    graph_elapsed = time.time() - graph_start
    logger.debug(f"[TIMING] get_graph: {graph_elapsed:.3f}s")

    # Step 3: Traverse tree
    traverse_start = time.time()
    # return_json = True
    return_json = False
    if return_json:
        rtns = {node: traverse_json_structure(G, node, direction, traversal_depth, valid_entity_type_filter,
                                              valid_dependency_type_filter)
                for node in start_entities}
        rtn_str = json.dumps(rtns)
    else:
        rtns = [traverse_tree_structure(G, node, direction, traversal_depth, valid_entity_type_filter,
                                        valid_dependency_type_filter)
                for node in start_entities]
        rtn_str = "\n\n".join(rtns)
    traverse_elapsed = time.time() - traverse_start
    logger.debug(f"[TIMING] traverse_tree_structure: {traverse_elapsed:.3f}s, num_entities={len(start_entities)}, output_length={len(rtn_str)} chars")

    if hints.strip():
        rtn_str += "\n\n" + hints

    total_elapsed = time.time() - func_start_time
    logger.debug(f"[TIMING] explore_tree_structure COMPLETED in {total_elapsed:.3f}s")

    return rtn_str.strip()


__all__ = [
    'set_current_issue',
    'reset_current_issue',
    'get_current_issue_id',
    'get_current_issue_data',
    'get_current_repo_modules',
    'get_graph_entity_searcher',
    'get_graph_dependency_searcher',
    'get_graph',
    'get_repo_save_dir',
    'search_code_snippets',
    'explore_tree_structure',
    'get_entity_contents'
]
