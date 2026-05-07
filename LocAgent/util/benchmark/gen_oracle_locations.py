import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
import os
import json
import re
import subprocess
import tempfile
import collections
import argparse
import logging
import logging.handlers
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from queue import Empty
from util.utils import load_jsonl, append_to_jsonl
from util.benchmark.git_repo_manager import remove_repo_worktree
from util.benchmark.setup_repo import setup_repo_worktree
from util.benchmark.parse_patch import (
    get_oracle_filenames, parse_patch,
)
from util.benchmark.parse_python_file import (
    parse_python_file,
    parse_class_docstrings, is_docstring,
    parse_import_nodes, is_import_statement,
    parse_comment_nodes, is_comment,
    parse_global_var_from_file, is_global_var
)
from util.benchmark.extract_support_locations import (
    RepoSymbolIndex, extract_support_for_instance,
)
import torch.multiprocessing as mp
from datasets import load_dataset


def prepare_repo_for_patch(instance, repo_base_dir, dataset=None, split=None):
    os.makedirs(repo_base_dir, exist_ok=True)
    return setup_repo_worktree(
        instance_data=instance,
        repo_base_dir=repo_base_dir,
        dataset=dataset,
        split=split,
    )


def cleanup_prepared_repo(repo_source_dir, repo_dir, instance_id=None, logger=None):
    if not repo_dir:
        return

    try:
        remove_repo_worktree(repo_source_dir, repo_dir)
    except subprocess.CalledProcessError as exc:
        message = f"Failed to cleanup prepared repo for instance {instance_id}: {exc}"
        if logger is not None:
            logger.debug(message)
        else:
            logging.warning(message)


def parse_module_name(code_str: str):
    # Regular expression to match the function definition and extract the name
    match = re.search(r'\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_str)

    if match:
        function_name = match.group(1)
        return function_name
    else:
        # print("No function definition found.")
        return None


def check_moduel_existed(module, file_structure):
    s = file_structure
    module_type = module.split(':')[0].strip()
    module_name = module.split(':')[-1].strip()
    
    if module_type == 'function' and '.' not in module_name:
        for func in s['functions']:
            if func['name'] == module_name:
                return True
    elif module_type == 'function' and '.' in module_name:
        class_name = module_name.split('.')[0]
        method_name = module_name.split('.')[-1]
        cls = [cls for cls in s['classes'] if cls['name'] == class_name]
        if cls:
            method = [method for method in cls[0]['methods'] if method['name'] == method_name]
            if method:
                return True
    elif module_type == 'class':
        cls = [cls for cls in s['classes'] if cls['name'] == module_name]
        if cls:
            return True
        
    return False


# def get_module_from_line_number_with_file_structure(line, file_structure, include_class=False, merge_init=True):
def get_module_from_line_number_with_file_structure(line, file_structure, 
                                                    include_class=False, 
                                                    merge_init=False
                                                    ):
    s = file_structure
    for txt in s['classes']:
        for func in txt['methods']:
            if line >= func['start_line'] and line <= func['end_line']:
                if merge_init and func['name'] == '__init__':
                    desc = f"class: {txt['name']}"
                    return desc
                else:
                    desc = f"function: {txt['name']}.{func['name']}"
                    return desc
                
        # don't belong to any methods
        if line >= txt['start_line'] and line <= txt['end_line']:
            desc = f"class: {txt['name']}"
            # if not txt['methods'] or include_class:
            if include_class:
                return desc
            else:
                return None
            
    for txt in s['functions']:
        if line >= txt['start_line'] and line <= txt['end_line']:
            desc = f"function: {txt['name']}"
            return desc
    
    return None


def apply_patch_str(patch, apply_file_path, hunk_size):
    # Write the patch string to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_patch_file:
        temp_patch_file.write(patch)
        temp_patch_file_path = temp_patch_file.name

    # Apply the patch using
    try:
        result = subprocess.run(
            ['patch', '-p1', '-i', temp_patch_file_path, apply_file_path],
            check=True,
            text=True,
            capture_output=True
        )
        # print("Patch applied successfully.")
        # logging.debug(result.stdout)
        offsets = [0 for i in range(hunk_size)]
        for out in str(result.stdout).splitlines():
            # if out.startswith('patching file'):
            #     offsets.append(0)
            # else:
                # process offset
                # Regular expression to extract offset (including negative values)
            pattern = r"Hunk #(\d+) succeeded at (\d+) \(offset ([+-]?\d+) lines\)"
            match = re.search(pattern, str(out))
            # Extracting the values if a match is found
            if match:
                hunk_id = int(match.group(1))
                offset = int(match.group(3))
                offsets[hunk_id-1] = offset
                
        # logging.debug('offsets', offsets)
        return (True, offsets)
    except subprocess.CalledProcessError as e:
        # logging.warning(f"Error applying patch: {e.stderr}")
        return (False, [])
    finally:
        # Clean up the temporary file
        import os
        os.remove(temp_patch_file_path)


def map_import_lines(codes):
    in_import_statement = False
    open_parens = 0
    line_labels = {}  # Dictionary to store line number and its label (True/False)
    for code in codes:
        content = code['content']
        line_num = code['line']
        stripped_line = content.strip()
        if not in_import_statement:
            if stripped_line.startswith('import ') or stripped_line.startswith('from '):
                in_import_statement = True
                open_parens += stripped_line.count('(') - stripped_line.count(')')
                line_labels[line_num] = True
                if open_parens == 0:
                    in_import_statement = False
            else:
                # Not an import statement
                line_labels[line_num] = False
        else:
            # Inside a multi-line import statement
            open_parens += stripped_line.count('(') - stripped_line.count(')')
            line_labels[line_num] = True
            if open_parens == 0:
                in_import_statement = False
    return line_labels


def group_patch_by_file(patch):
    """
    Groups a patch string by file.

    Args:
        patch (str): The patch content as a string.

    Returns:
        dict: A dictionary where the keys are file paths, and the values are the corresponding patch content.
        --- a/a.py
        +++ b/a.py
        @@ -1,1 +1,1 @@
        -x = 1
        +x = 2
        --- a/b.py
        +++ b/b.py
        @@ -3,1 +3,1 @@
        -y = 3
        +y = 4

        {
            "a.py": "--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n",
            "b.py": "--- a/b.py\n+++ b/b.py\n@@ -3,1 +3,1 @@\n-y = 3\n+y = 4\n",
        }
    """
    patch_by_file = defaultdict(list)
    patch_lines = patch.splitlines()

    current_file = None
    file_header_pattern = r"^(---|\+\+\+) (.+)"

    for line in patch_lines:
        match = re.match(file_header_pattern, line)
        if match:
            current_file = re.sub(r"^(a/|b/)", "", match.group(2))
            patch_by_file[current_file].append(f"{line}\n")
        else:
            if current_file:
                patch_by_file[current_file].append(f"{line}\n")

    return {file: "".join(hunks) for file, hunks in patch_by_file.items()}


def _read_file_text(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception:
        return ''


def extract_module_from_patch(instance, repo_dir, max_edit_file_num=1,
                              logger=None,
                              include_gvar=False,
                              rank=0,
                              extract_support=True):
    edit_files = get_oracle_filenames(instance['patch'])
    # print(len(edit_files))

    # filter python files and limit the number of files
    filtered_edit_files = []
    for fle in edit_files:
        if fle.endswith('.py'):
            filtered_edit_files.append(fle)
    if not filtered_edit_files: return None
    if len(filtered_edit_files) > max_edit_file_num:
        return None

    file_changes = parse_patch(instance['patch'])
    # Group the patch by file
    patch_by_file = group_patch_by_file(instance['patch'])

    # Build the repo symbol index on the OLD repo state, before any patch is
    # applied to disk.  Used downstream by support-location extraction.
    repo_index = None
    if extract_support:
        try:
            repo_index = RepoSymbolIndex(repo_dir)
        except Exception as exc:
            if logger is not None:
                logger.debug(f"Failed to build repo symbol index: {exc}")
            repo_index = None

    file_diffs = []  # per-file metadata for support extraction
    updated_file_changes = []
    for file_change in file_changes:
        file = file_change['file']
        if not file.endswith('.py'): continue
        target_file_path = os.path.join(repo_dir, file)

        # capture the raw source BEFORE applying the patch
        old_source = _read_file_text(target_file_path)

        # initial file structure
        class_info, function_names, file_lines = parse_python_file(target_file_path)
        old_file_structure = {
            "classes": class_info,
            "functions": function_names,
            "text": file_lines,
        }
        old_global_vars = parse_global_var_from_file(target_file_path)
        old_import_nodes = parse_import_nodes(target_file_path)
        old_comment_nodes = parse_comment_nodes(target_file_path)
        old_docstring_nodes = parse_class_docstrings(target_file_path)

        # Extract the partial patch for this file
        partial_patch = patch_by_file.get(file)
        if not partial_patch:
            # logging.warning(f"No patch found for {file}")
            continue

        # Apply the patch
        success, offsets = apply_patch_str(partial_patch, target_file_path, len(file_change['hunks']))
        if not success:
            # TODO: assert
            return None

        # capture the raw source AFTER applying the patch
        new_source = _read_file_text(target_file_path)

        # new file structure
        class_info, function_names, file_lines = parse_python_file(target_file_path)
        new_file_structure = {
            "classes": class_info,
            "functions": function_names,
            "text": file_lines,
        }
        new_global_vars = parse_global_var_from_file(target_file_path)
        new_import_nodes = parse_import_nodes(target_file_path)
        new_comment_nodes = parse_comment_nodes(target_file_path)
        new_docstring_nodes = parse_class_docstrings(target_file_path)

        # collect raw delete/add line numbers for support extraction.  Old-side
        # lines are kept in pre-patch coordinates (no offset, since we parsed
        # ``old_source`` before the patch was applied); new-side lines pick up
        # the GNU-patch offsets returned by apply_patch_str.
        support_delete_lines = []
        support_add_lines = []

        changes = collections.defaultdict(list)
        for i, hunk in enumerate(file_change['hunks']):
            # if i == len(offsets): offsets.append(0) # align with hunk size
            
            # process edited lines
            delete_change = hunk['changes']['delete']
            add_change = hunk['changes']['add']
            # deleted_lines, added_lines = [], []
            
            for delete in delete_change:
                line = delete['line'] + offsets[i]
                # collect support-side line in OLD-file coordinates (pre-patch source)
                if delete.get('line') is not None:
                    support_delete_lines.append(delete['line'])
                # is_comment(line, old_comment_nodes) or \
                if is_import_statement(line, old_import_nodes) or \
                    delete['content'].strip().startswith('#') or \
                    is_docstring(line, old_docstring_nodes):
                    continue
                
                # check is global var
                variable = is_global_var(line, old_global_vars)
                if variable:
                    if include_gvar and variable not in changes['edited_modules']:
                        changes['edited_modules'].append(f'variable: {variable}')
                    continue
                
                # check is module
                module = get_module_from_line_number_with_file_structure(line, old_file_structure)
                if module and not module in changes['edited_modules']:
                    changes['edited_modules'].append(module)
                # elif not module and delete['content'].strip():
                #     deleted_lines.append(delete)
                    
            for add in add_change:
                # is_comment(line, new_comment_nodes) or \
                line = add['line'] + offsets[i]
                # collect support-side line in NEW-file coordinates (post-patch source)
                if add.get('line') is not None:
                    support_add_lines.append(line)
                if is_import_statement(line, new_import_nodes) or \
                    add['content'].strip().startswith('#') or \
                    is_docstring(line, new_docstring_nodes):
                    continue
                
                # check is global var
                variable = is_global_var(line, new_global_vars)
                if variable:
                    if not include_gvar: continue
                    if variable in old_global_vars and f'variable: {variable}' not in changes['edited_modules']:
                        changes['edited_modules'].append(f'variable: {variable}')
                    elif variable not in old_global_vars and f'variable: {variable}' not in changes['added_modules']:
                        changes['added_modules'].append(f'variable: {variable}')
                    continue
                
                # check is module
                module = get_module_from_line_number_with_file_structure(line, new_file_structure)
                if module and \
                    module not in changes['edited_modules'] and \
                    module not in changes['added_modules']:
                    
                    # check if the module in old file
                    if check_moduel_existed(module, old_file_structure):
                        changes['edited_modules'].append(module)
                    else:
                        changes['added_modules'].append(module)
        
        _changes = collections.defaultdict(list)
        for mode, change in changes.items():
            if mode in ['added_lines', 'edited_lines']:
                continue
            for c in change:
                if c.startswith("variable:"):
                    continue
                if mode in ['added_modules', 'edited_modules']:
                    _mode = mode.replace('_modules', '_entities')
                    _changes[_mode].append(f'{file}:{c.split(':')[-1].strip()}')
                
                if c.startswith("function:") and '.' in c:
                    _c = c.split(':')[-1].strip().split('.')[0]
                    if f'{file}:{_c.strip()}' not in _changes[mode]:
                        _changes[mode].append(f'{file}:{_c.strip()}')
                else:
                    if f'{file}:{c.split(':')[-1].strip()}' not in _changes[mode]:
                        _changes[mode].append(f'{file}:{c.split(':')[-1].strip()}')
        
        updated_file_changes.append({
            'file': file,
            # 'changes': changes
            'changes': _changes
        })

        file_diffs.append({
            'rel_file': file,
            'old_source': old_source,
            'new_source': new_source,
            'delete_lines': support_delete_lines,
            'add_lines': support_add_lines,
        })

    # Aggregate edited entities to subtract from the support set (complementary
    # definition: support_locations := support \ edit).
    edit_entity_strs = set()
    for fc in updated_file_changes:
        ch = fc.get('changes') or {}
        for key in ('edited_entities', 'added_entities'):
            for e in ch.get(key, []) or []:
                edit_entity_strs.add(e)

    support_entities = []
    support_stats = {}
    if extract_support and repo_index is not None and file_diffs:
        try:
            support_entities, support_stats = extract_support_for_instance(
                repo_index, file_diffs, edit_entity_strs)
        except Exception as exc:
            if logger is not None:
                logger.debug(f"Support extraction failed: {exc}")
            else:
                logging.exception("Support extraction failed")
            support_entities, support_stats = [], {}

    return {
        'file_changes': updated_file_changes,
        'support_entities': support_entities,
        'support_stats': support_stats,
    }


def _aggregate_support_stats(agg, stats):
    """Accumulate per-instance support stats into a running aggregate."""
    if not stats:
        return
    agg['total_calls'] += stats.get('total_calls', 0)
    agg['resolved'] += stats.get('resolved', 0)
    agg['n_instances'] += 1
    by_status = stats.get('by_status') or {}
    for k, v in by_status.items():
        agg['by_status'][k] = agg['by_status'].get(k, 0) + v


def _print_support_stats(agg):
    total = agg['total_calls']
    resolved = agg['resolved']
    rate = (resolved / total) if total else 0.0
    print(f"[support] instances={agg['n_instances']} "
          f"calls={total} resolved={resolved} "
          f"resolution_rate={rate:.3f}")
    if agg['by_status']:
        print(f"[support] by_status={agg['by_status']}")


def _process_records(records,
                     dataset_name,
                     split,
                     output_dir,
                     repo_base_dir,
                     max_edit_file_num,
                     selected_list=None,
                     hf_dataset_for_setup=None):
    """Common per-instance loop used by dataset / parquet / jsonl entry points.

    ``hf_dataset_for_setup`` is forwarded to ``setup_repo_worktree`` (only the
    HF-dataset path supplies it; for local files we pass None).
    """
    out_subdir = os.path.join(output_dir, dataset_name, split)
    os.makedirs(out_subdir, exist_ok=True)
    output_file = os.path.join(out_subdir, 'gt_location_with_support.jsonl')
    error_file = os.path.join(out_subdir, 'error_oracle_location_list.txt')
    empty_edit_file = os.path.join(out_subdir, 'empty_edit_location_list.txt')

    processed_instances = set()
    if os.path.exists(output_file):
        processed_instances = {
            row['instance_id'] for row in load_jsonl(output_file)
        }

    agg_stats = {'n_instances': 0, 'total_calls': 0, 'resolved': 0,
                 'by_status': {}}
    error_list, empty_edit_list = [], []

    for instance in tqdm(records):
        if instance['instance_id'] in processed_instances:
            continue
        if selected_list and instance['instance_id'] not in selected_list:
            continue

        repo_source_dir = None
        repo_dir = None
        try:
            repo_source_dir, repo_dir = prepare_repo_for_patch(
                instance,
                repo_base_dir=repo_base_dir,
                dataset=hf_dataset_for_setup,
                split=split if hf_dataset_for_setup else None,
            )

            try:
                result = extract_module_from_patch(
                    instance,
                    repo_dir,
                    max_edit_file_num=max_edit_file_num,
                )
            except Exception:
                logging.exception(
                    "Failed to extract oracle locations for instance %s",
                    instance['instance_id'],
                )
                error_list.append(instance['instance_id'])
                continue

            if not result:
                empty_edit_list.append(instance['instance_id'])
                continue

            file_changes = result.get('file_changes') or []
            support_entities = result.get('support_entities') or []
            support_stats = result.get('support_stats') or {}
            _aggregate_support_stats(agg_stats, support_stats)

            append_to_jsonl({
                'instance_id': instance['instance_id'],
                'file_changes': file_changes,
                'support_entities': support_entities,
                'support_stats': support_stats,
                'repo': instance['repo'],
                'base_commit': instance.get('base_commit'),
                'problem_statement': instance['problem_statement'],
                'patch': instance['patch'],
            }, output_file)
        except FileNotFoundError as e:
            logging.info(e)
            error_list.append(instance['instance_id'])
            continue
        except subprocess.CalledProcessError:
            logging.exception(
                "Failed to setup repo for instance %s",
                instance['instance_id'],
            )
            error_list.append(instance['instance_id'])
            continue
        finally:
            cleanup_prepared_repo(
                repo_source_dir,
                repo_dir,
                instance_id=instance['instance_id'],
            )

    with open(empty_edit_file, 'w') as f:
        for instance_id in empty_edit_list:
            f.write(f"{instance_id}\n")
    with open(error_file, 'w') as f:
        for instance_id in error_list:
            f.write(f"{instance_id}\n")

    print(empty_edit_list)
    print(error_list)
    _print_support_stats(agg_stats)
    return output_file


def generate_oracle_locations_for_dataset(dataset, split, max_edit_file_num=1,
                                     output_dir='evaluation/gt_location',
                                     repo_base_dir='playground',
                                     selected_list=None):
    bench_data = load_dataset(dataset, split=split)
    return _process_records(
        records=bench_data,
        dataset_name=dataset.split('/')[-1],
        split=split,
        output_dir=output_dir,
        repo_base_dir=repo_base_dir,
        max_edit_file_num=max_edit_file_num,
        selected_list=selected_list,
        hf_dataset_for_setup=dataset,
    )


def _load_records_from_path(data_file):
    """Load benchmark instances from a .parquet or .jsonl/.json file."""
    if data_file.endswith('.parquet'):
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError(
                "reading .parquet requires pandas; pip install pandas pyarrow"
            ) from e
        df = pd.read_parquet(data_file)
        return df.to_dict(orient='records')
    if data_file.endswith('.jsonl'):
        return load_jsonl(data_file)
    if data_file.endswith('.json'):
        with open(data_file, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = [payload]
        return payload
    raise ValueError(f"Unsupported data_file extension: {data_file}")


def generate_oracle_locations_for_path(data_file, split='test',
                                       max_edit_file_num=1,
                                       output_dir='evaluation/gt_location',
                                       repo_base_dir='playground',
                                       dataset_name=None,
                                       selected_list=None):
    """Run extraction over instances loaded from a local parquet / jsonl file.

    The output directory follows the same convention as the HF-dataset path:
    ``<output_dir>/<dataset_name>/<split>/gt_location_with_support.jsonl``.
    """
    records = _load_records_from_path(data_file)
    if dataset_name is None:
        # default: <parent_dir_name>, e.g. dataset/swe_bench_lite/test.parquet -> swe_bench_lite
        dataset_name = os.path.basename(os.path.dirname(os.path.abspath(data_file))) \
            or os.path.splitext(os.path.basename(data_file))[0]
    return _process_records(
        records=records,
        dataset_name=dataset_name,
        split=split,
        output_dir=output_dir,
        repo_base_dir=repo_base_dir,
        max_edit_file_num=max_edit_file_num,
        selected_list=selected_list,
        hf_dataset_for_setup=None,
    )


def run_extract_locations_from_patch(rank, 
                                  queue, log_queue, output_file_lock,
                                  repo_playground, output_file, max_edit_file_num
                                  ):
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(logging.getLevelName("DEBUG"))
    logger.handlers = []
    logger.addHandler(queue_handler)

    logger.debug(f"------ rank {rank} start ------")
    
    while True:
        try:
            instance = queue.get_nowait()
        except Empty:
            break
        
        repo_source_dir = None
        repo_dir = None
        try:
            # pull the repo
            repo_source_dir, repo_dir = prepare_repo_for_patch(
                instance,
                repo_base_dir=repo_playground,
                dataset=None,
                split=None,
            )
            result = extract_module_from_patch(instance, repo_dir,
                                                     logger=logger,
                                                     max_edit_file_num=max_edit_file_num, rank=rank)
            if not result:
                continue
            file_changes = result.get('file_changes') or []
            support_entities = result.get('support_entities') or []
            support_stats = result.get('support_stats') or {}
            with output_file_lock:
                with open(output_file, 'a') as f:
                    f.write(json.dumps({
                        'instance_id': instance['instance_id'],
                        'file_changes': file_changes,
                        'support_entities': support_entities,
                        'support_stats': support_stats,
                        'repo': instance['repo'],
                        'base_commit': instance.get('base_commit'),
                        'problem_statement': instance['problem_statement'],
                        'patch': instance['patch']
                    }) + '\n')
        except FileNotFoundError:
            logger.debug(f"rank {rank}: FileNotFoundError.")
            # error_list.append(instance['instance_id'])
        except subprocess.CalledProcessError as e:
            logger.debug(f"rank {rank}: {e}")
            # error_list.append(instance['instance_id'])
        except Exception as e:
            logger.debug(f"rank {rank}: {e}")
        finally:
            cleanup_prepared_repo(
                repo_source_dir,
                repo_dir,
                instance_id=instance['instance_id'],
                logger=logger,
            )
            # error_list.append(instance['instance_id'])


def generate_oracle_locations_for_data_file(dataset_file, n_limit,
                                          max_edit_file_num=1, 
                                          repo_base_dir='playground/loc_bench',
                                          num_processes=1):
    logging.basicConfig(
        # filename=f"{args.output_folder}/localize.log",
        level=logging.getLevelName('DEBUG'),
        format="%(asctime)s %(filename)s %(levelname)s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"evaluation/gt_data/LOC-bench/gen_gt.log"),
            logging.StreamHandler()
        ]
    )
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    output_file = f'evaluation/gt_data/LOC-bench/gt_modules_data_{max_edit_file_num}file_{current_date}.jsonl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    processed_instances = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                processed_instances.append(json.loads(line)['instance_id'])     
    
    bench_data = load_jsonl(dataset_file)
    manager = mp.Manager()
    queue = manager.Queue()
    output_file_lock = manager.Lock()
    
    num_instances = 0
    for instance in bench_data[:n_limit]:
        if not instance['instance_id'] in processed_instances:
            queue.put(instance)
            num_instances += 1
    
    log_queue = manager.Queue()
    queue_listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
    queue_listener.start()
    mp.spawn(
        run_extract_locations_from_patch,
        nprocs=min(num_instances, num_processes) if num_processes > 0 else num_instances,
        args=(queue, log_queue, output_file_lock,
              repo_base_dir, output_file, max_edit_file_num
              ),
        join=True
    )
    queue_listener.stop()
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_base_dir', type=str, default='playground/repo_base')
    parser.add_argument('--output_dir', type=str, default='evaluation/gt_location')
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument('--data_file', type=str, default=None,
                        help='Optional .parquet or .jsonl file to read instances from. '
                             'When set, --dataset is ignored for data loading.')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Override the dataset name used in the output path. '
                             'Defaults to the parent directory of --data_file.')
    parser.add_argument('--selected_list_file', type=str, default='playground/repo_base')
    parser.add_argument('--loc_bench', action='store_true')
    parser.add_argument("--max_edit_file_num", type=int, default=1)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--gen_n_limit", type=int, default=0)
    # parser.add_argument('--merge_init', action='store_true')
    args = parser.parse_args()

    if args.data_file:
        generate_oracle_locations_for_path(
            data_file=args.data_file,
            split=args.split,
            max_edit_file_num=args.max_edit_file_num,
            output_dir=args.output_dir,
            repo_base_dir=args.repo_base_dir,
            dataset_name=args.dataset_name,
        )
    else:
        generate_oracle_locations_for_dataset(args.dataset, args.split, args.max_edit_file_num,
                                              args.output_dir, args.repo_base_dir)

    if args.loc_bench:
        generate_oracle_locations_for_data_file(args.dataset, args.gen_n_limit,
                                              args.max_edit_file_num,
                                              args.repo_base_dir, args.num_processes)
