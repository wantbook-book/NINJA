import argparse
import os
import json
import logging
import logging.handlers
import time
import threading
import traceback
import toml
from queue import Empty, Queue
from typing import List
from tqdm import tqdm
from copy import deepcopy
from datasets import load_dataset, load_from_disk

from util.runtime.execute_ipython import execute_ipython
from util.runtime import function_calling
from util.actions.action_parser import ResponseParser
from util.actions.action import ActionType
from util.prompts.prompt import PromptManager
from util.prompts import general_prompt
from util.prompts.pipelines import (
    simple_localize_pipeline as simple_loc,
    auto_search_prompt as auto_search,
)
from util.cost_analysis import calc_cost
from util.utils import *
from util.process_output import (
    get_loc_results_from_raw_outputs,
    merge_sample_locations,
)
from plugins import LocationToolsRequirement
from plugins.location_tools.repo_ops.repo_ops import (
    get_or_create_instance_context,
    reset_instance_context,
)
import litellm
from litellm import Message as LiteLLMMessage
from openai import APITimeoutError


from time import sleep
from concurrent.futures import TimeoutError
import torch.multiprocessing as mp
from util.runtime.fn_call_converter import (
    convert_fncall_messages_to_non_fncall_messages,
    convert_non_fncall_messages_to_fncall_messages,
    STOP_WORDS as NON_FNCALL_STOP_WORDS
)
# litellm.set_verbose=True
# os.environ['LITELLM_LOG'] = 'DEBUG


def _consume_progress_events(progress_queue, total_tasks: int, stop_event: threading.Event):
    status_counts = {
        'success': 0,
        'empty': 0,
        'error': 0,
    }
    completed = 0
    with tqdm(total=total_tasks, desc='Localizing issues', dynamic_ncols=True) as pbar:
        while completed < total_tasks:
            try:
                event = progress_queue.get(timeout=0.2)
            except Empty:
                if stop_event.is_set():
                    break
                continue

            status = event.get('status', 'success')
            if status not in status_counts:
                status_counts[status] = 0
            status_counts[status] += 1
            completed += 1
            pbar.update(1)
            pbar.set_postfix(
                success=status_counts.get('success', 0),
                empty=status_counts.get('empty', 0),
                error=status_counts.get('error', 0),
            )

        if completed < total_tasks:
            pbar.set_postfix(
                success=status_counts.get('success', 0),
                empty=status_counts.get('empty', 0),
                error=status_counts.get('error', 0),
                incomplete=total_tasks - completed,
            )
            pbar.refresh()


def filter_dataset(dataset, filter_column: str, used_list: str):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml')
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = toml.load(file)
            if used_list in data:
                selected_ids = data[used_list]
                logging.info(
                    f'Filtering {len(selected_ids)} tasks from "selected_ids"...'
                )
                def filter_function(example):
                    return example[filter_column] in selected_ids  # Replace 'id' with the actual field name in the dataset
                filtered_dataset = dataset.filter(filter_function)
                # subset = dataset[dataset[filter_column].isin(selected_ids)]
                logging.info(f'Retained {len(filtered_dataset)} tasks after filtering')
                return filtered_dataset
    return dataset


def load_benchmark_dataset(dataset_spec: str, split: str):
    if os.path.isfile(dataset_spec):
        if dataset_spec.endswith(".parquet"):
            return load_dataset("parquet", data_files={split: dataset_spec}, split=split)
        raise ValueError(
            f"Unsupported local dataset file: {dataset_spec}. "
            "Only local .parquet files are supported."
        )

    if os.path.isdir(dataset_spec):
        try:
            dataset = load_from_disk(dataset_spec)
        except Exception:
            parquet_file = os.path.join(dataset_spec, f"{split}.parquet")
            if os.path.isfile(parquet_file):
                return load_dataset("parquet", data_files={split: parquet_file}, split=split)
            raise

        if hasattr(dataset, "keys"):
            if split not in dataset:
                raise ValueError(
                    f"Split '{split}' not found in local dataset directory: {dataset_spec}. "
                    f"Available splits: {list(dataset.keys())}"
                )
            return dataset[split]
        return dataset

    return load_dataset(dataset_spec, split=split)


def get_task_instruction(instance: dict, task: str = 'auto_search', include_pr=False, include_hint=False):
    output_format = None
    instruction = ""
    
    # for auto-search pipeline
    if task.strip() == 'auto_search':
        task_description = auto_search.TASK_INSTRUECTION.format(
            package_name=instance['instance_id'].split('_')[0]
        )
    
    elif task.strip() == 'simple_localize':
        task_description = simple_loc.SEARCH_LOC_TASK_INSTRUCTION
        output_format = simple_loc.OUTPUT_FORMAT_LOC
        
    else:
        return None

    instruction += task_description
        
    if include_pr:
        problem_statement = instance['problem_statement']
        instruction += general_prompt.PR_TEMPLATE.format(
            title=problem_statement.strip().split('\n')[0],
            description = '\n'.join(problem_statement.strip().split('\n')[1:]).strip()
        )
    
    if output_format:
        instruction += output_format
    
    if include_hint:
        instruction += (
            'IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
            'Don\'t include any lambda functions!\n'
            'You should NOT modify any files!\n'
        )

    # NOTE: You can actually set slightly different instruction for different task
    # instruction += AGENT_CLS_TO_INST_SUFFIX
    return instruction


def auto_search_process(result_queue,
                        model_name, messages, fake_user_msg,
                        repo_ctx,
                        tools = None,
                        traj_data=None,
                        temp=1.0,
                        max_iteration_num=20,
                        use_function_calling=True):
    model_name_lower = model_name.lower()
    use_non_fncall_compat = bool(
        tools and (
            'hosted_vllm' in model_name
            or 'qwen' in model_name_lower
            or 'deepseek' in model_name_lower
        )
    )
    if use_non_fncall_compat:
        use_function_calling = False
        
    # for LLM which do not support function calling
    if tools and not use_function_calling:
        messages = convert_fncall_messages_to_non_fncall_messages(messages, tools, add_in_context_learning_example=False)
            
    # code_history = []
    parser = ResponseParser()
    if not traj_data:
        traj_msgs = messages.copy()
        prompt_tokens = 0
        completion_tokens = 0
    else:
        # continue from last traj
        traj_msgs = traj_data['messages']
        prompt_tokens = traj_data['usage']['prompt_tokens']
        completion_tokens = traj_data['usage']['completion_tokens']
        
    cur_interation_num = 0
    last_message = None
    finish = False
    final_output = ""
    while not finish:
        cur_interation_num += 1
        if cur_interation_num > max_iteration_num:
            logging.warning("Maximum iteration limit reached without receiving a finish action.")
            break
        if cur_interation_num == max_iteration_num:
            messages.append({
                'role': 'user',
                'content': 'The Maximum number of interation has been reached, please generate your final output with required format and use <finish></finish> to exit.'
            })
            traj_msgs.append({
                'role': 'user',
                'content': 'The Maximum number of interation has been reached, please generate your final output with required format and use <finish></finish> to exit.'
            })

        try:
            # new conversation
            if use_non_fncall_compat:
                response = litellm.completion(
                    model=model_name,
                    temperature=temp, top_p=0.8, repetition_penalty=1.05, 
                    messages=messages,
                    stop=NON_FNCALL_STOP_WORDS
                )
            elif tools:
                response = litellm.completion(
                    model=model_name,
                    tools=tools,
                    messages=messages,
                    temperature=temp,
                    # stop=['</execute_ipython>'], #</finish>',
                )
            else:
                response = litellm.completion(
                    model=model_name,
                    messages=messages,
                    temperature=temp,
                    stop=['</execute_ipython>'], #</finish>',
                )
        except litellm.BadRequestError as e:
            # If there's an error, send the error info back to the parent process
            result_queue.put({'error': str(e), 'type': 'BadRequestError'})
            return
        except Exception as e:
            result_queue.put({
                'error': str(e),
                'type': type(e).__name__,
                'traceback': traceback.format_exc(),
            })
            return
        
        if last_message and response.choices[0].message.content == last_message:
            messages.append({
                "role": "user",
                "content": "OBSERVATION:\n" + "Don't repeat your response.\n" + fake_user_msg,
            })
            traj_msgs.append({
                "role": "user",
                "content": "OBSERVATION:\n" + "Don't repeat your response.\n" + fake_user_msg,
            })
            continue
        
        raw_response = deepcopy(response)
        # logging.info('response.choices[0].message')
        if use_non_fncall_compat:
            try:
                non_fncall_response_message = response.choices[0].message
                fn_call_messages_with_response = (
                    convert_non_fncall_messages_to_fncall_messages(
                        [non_fncall_response_message], tools # messages + 
                    )
                )
                fn_call_response_message = fn_call_messages_with_response[-1]
                if not isinstance(fn_call_response_message, LiteLLMMessage):
                    fn_call_response_message = LiteLLMMessage(
                        **fn_call_response_message
                    )
                response.choices[0].message = fn_call_response_message
            except:
                logging.info('convert none fncall messages failed.')
                continue 
                
        last_message = response.choices[0].message.content
        logging.info(response.choices[0].message)
        messages.append(convert_to_json(raw_response.choices[0].message))
        traj_msgs.append(convert_to_json(raw_response.choices[0].message))
        usage = getattr(response, 'usage', None)
        prompt_tokens += getattr(usage, 'prompt_tokens', 0) or 0
        completion_tokens += getattr(usage, 'completion_tokens', 0) or 0
            
        actions = parser.parse(response)
        if not isinstance(actions, List):
            actions = [actions]
        for action in actions:
            logging.debug(action.action_type)
            if action.action_type == ActionType.FINISH:
                final_output = action.thought
                logging.info('='*15)
                logging.info("\nFinal Response:=\n" + final_output)
                finish = True # break
            elif action.action_type == ActionType.MESSAGE:
                logging.debug("thought:\n" + action.content)
                # check if enough
                messages.append({"role": "user", "content": fake_user_msg})
                traj_msgs.append({"role": "user", "content": fake_user_msg})
                # continue
            elif action.action_type == ActionType.RUN_IPYTHON:
                ipython_code = action.code.strip('`')
                logging.info(f"Executing code:\n```\n{ipython_code}\n```")
                function_response = execute_ipython(ipython_code, ctx=repo_ctx)
                try:
                    function_response = eval(function_response)
                except SyntaxError:
                    function_response = function_response
                if not isinstance(function_response, str):
                    function_response = str(function_response)
                
                logging.info("OBSERVATION:\n" + function_response)
                if not tools:
                    messages.append({
                        "role": "user",
                        "content": "OBSERVATION:\n" + function_response,
                    })
                    traj_msgs.append({
                        "role": "user",
                        "content": "OBSERVATION:\n" + function_response,
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": action.tool_call_id,
                        "name": action.function_name,
                        "content": "OBSERVATION:\n" + function_response,
                    })
                    traj_msgs.append({
                        "role": "tool",
                        "tool_call_id": action.tool_call_id,
                        "name": action.function_name,
                        "content": "OBSERVATION:\n" + function_response,
                    })
            else:
                logging.warning('Error Action!')
                # return

        if cur_interation_num >= max_iteration_num and not finish:
            logging.warning("Maximum iteration limit reached and no finish action was produced.")
            break

    # save traj
    traj_data = {
        'messages': traj_msgs,
        'tools': tools,
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        }
    }
    # return final_output, messages, traj_data
    result_queue.put((final_output, messages, traj_data))


def run_localize(rank, args, bug_queue, log_queue, output_file_lock, traj_file_lock, progress_queue):
    logger = logging.getLogger()
    logger.setLevel(logging.getLevelName(args.log_level))
    if log_queue is not None:
        queue_handler = logging.handlers.QueueHandler(log_queue)
        logger.handlers = []
        logger.addHandler(queue_handler)

    logger.debug(f"------ rank {rank} start ------")
    single_process_mode = args.num_processes == 1

    while True:
        try:
            bug = bug_queue.get_nowait()
        except Empty:
            break

        instance_id = bug["instance_id"]
        progress_status = 'error'
        repo_ctx = None
        try:
            prompt_manager = PromptManager(
                prompt_dir=os.path.join(os.path.dirname(__file__), 'util/prompts'),
                agent_skills_docs=LocationToolsRequirement.documentation,
            )

            logger.info("=" * 60)
            logger.info(f"==== rank {rank} setup localize {instance_id} ====")
            repo_ctx = get_or_create_instance_context(
                instance_id=instance_id,
                instance_data=bug,
                rank=rank,
            )

            # loc result
            raw_output_loc = []
            loc_trajs = {'trajs': []}
            total_prompt_tokens, total_completion_tokens = 0, 0

            for _ in range(args.num_samples):
                logger.info("=" * 60)
                logger.info(f"==== rank {rank} begin localizing {instance_id} ====")
                max_attempt_num = args.max_attempt_num
                while max_attempt_num:
                    logger.info("=" * 60)
                    logger.info(f"==== {instance_id} Count down: attempt {max_attempt_num} ====")
                    loc_start_time = time.time()
                    try:
                        """
                        Basic instructions:
                            - CodeAct instruction
                            - Few-shot Examples
                        """
                        if args.use_function_calling:
                            system_prompt = function_calling.SYSTEM_PROMPT
                            # system_prompt = CLAUDE_THINKING_INSTRUCTION
                        else:
                            system_prompt = prompt_manager.system_message
                            
                        messages: list[dict] = [{
                            "role": "system",
                            "content": system_prompt
                        }]
                            
                        if args.use_example:
                            messages.append({
                                "role": "user",
                                "content": prompt_manager.initial_user_message
                            })

                        logger.info(f"==== {instance_id} start auto search ====")
                        messages.append({
                            "role": "user",
                            "content": get_task_instruction(bug, include_pr=True, include_hint=True),
                        })

                        tools = None
                        if args.use_function_calling:
                            tools = function_calling.get_tools(
                                codeact_enable_search_keyword=True,
                                codeact_enable_search_entity=True,
                                codeact_enable_tree_structure_traverser=True,
                                simple_desc = args.simple_desc,
                            )

                        if single_process_mode:
                            logger.info(
                                f"{instance_id} is running in single-process debug mode; "
                                "auto_search_process will run inline and timeout enforcement is disabled."
                            )
                            result_queue = Queue()
                            try:
                                auto_search_process(
                                    result_queue=result_queue,
                                    model_name=args.model,
                                    messages=messages,
                                    fake_user_msg=auto_search.FAKE_USER_MSG_FOR_LOC,
                                    repo_ctx=repo_ctx,
                                    temp=1,
                                    tools=tools,
                                    use_function_calling=args.use_function_calling,
                                )
                            except Exception as e:
                                raise RuntimeError(
                                    f"auto_search_process failed inline: {e}\n{traceback.format_exc()}".strip()
                                ) from e
                            try:
                                result = result_queue.get_nowait()
                            except Empty:
                                raise RuntimeError("auto_search_process finished inline without returning a result.")
                        else:
                            ctx = mp.get_context('fork')  # use fork to inherit context!!
                            result_queue = ctx.Manager().Queue()
                            process = ctx.Process(target=auto_search_process, kwargs={
                                'result_queue': result_queue,
                                'model_name': args.model,
                                'messages': messages,
                                'fake_user_msg': auto_search.FAKE_USER_MSG_FOR_LOC,
                                'repo_ctx': repo_ctx,
                                'temp': 1,
                                'tools': tools,
                                'use_function_calling': args.use_function_calling,
                            })
                            process.start()
                            process.join(timeout=args.timeout)
                            if process.is_alive():
                                logger.warning(f"{instance_id} attempt {max_attempt_num} execution flow "
                                                f"reconstruction exceeded timeout. Terminating.")
                                process.terminate()
                                process.join()
                                raise TimeoutError

                            try:
                                result = result_queue.get(timeout=5)
                            except Empty:
                                raise RuntimeError(
                                    f"auto_search_process exited without returning a result (exitcode={process.exitcode})."
                                )
                        if isinstance(result, dict) and 'error' in result:
                            if result['type'] == 'BadRequestError':
                                raise litellm.BadRequestError(result['error'], args.model, args.model.split('/')[0])
                            raise RuntimeError(
                                f"{result['type']}: {result['error']}\n{result.get('traceback', '')}".strip()
                            )
                        loc_result, messages, traj_data = result
                            
                    except litellm.BadRequestError as e:
                        logger.warning(f'{e}. Try again.')
                        max_attempt_num = max_attempt_num - 1
                        continue
                    except APITimeoutError:
                        logger.warning(f"APITimeoutError. Try again.")
                        sleep(10)
                        max_attempt_num = max_attempt_num - 1
                        continue
                    except TimeoutError:
                        logger.warning(f"Processing time exceeded 15 minutes. Try again.")
                        max_attempt_num = max_attempt_num - 1
                        continue
                    except litellm.exceptions.ContextWindowExceededError as e:
                        logger.warning(f'{e}. Try again.')
                        max_attempt_num = max_attempt_num - 1
                        continue
                    except RuntimeError as e:
                        logger.warning(f'{e}. Try again.')
                        max_attempt_num = max_attempt_num - 1
                        continue

                    loc_end_time = time.time()
                    if not loc_result:
                        continue # empty result

                    total_prompt_tokens += traj_data['usage']['prompt_tokens']
                    total_completion_tokens += traj_data['usage']['completion_tokens']
                    traj_data['time'] = loc_end_time - loc_start_time
                    loc_trajs['trajs'].append(traj_data)

                    # generate correct output or finish last attempt
                    raw_output_loc.append(loc_result)
                    break

            if not raw_output_loc:
                progress_status = 'empty'
                # loc generalization failed
                logger.info(f"==== localizing {instance_id} failed, save empty outputs ====")
                loc_res = {
                        "instance_id": instance_id,
                        "found_files": [[]],
                        "found_modules": [[]],
                        "found_entities": [[]],
                        "raw_output_loc": raw_output_loc,
                        "meta_data": {
                            'repo': bug['repo'],
                            'base_commit': bug['base_commit'],
                            'problem_statement': bug['problem_statement'],
                            'patch': bug['patch'],
                            # 'gt_file_changes': gt_file_changes
                        }
                    }
                with output_file_lock:
                    append_to_jsonl(loc_res, args.output_file)
            else:
                progress_status = 'success'
                # process multiple loc outputs
                logger.info(f"==== localizing {instance_id} succeed, process multiple loc outputs ====")

                # all_valid_files = get_all_valid_files()
                all_found_files, all_found_modules, all_found_entities = get_loc_results_from_raw_outputs(
                    instance_id, raw_output_loc
                )
                
                loc_res = {
                    "instance_id": instance_id,
                    "found_files": all_found_files,
                    "found_modules": all_found_modules,
                    "found_entities": all_found_entities,
                    "raw_output_loc": raw_output_loc,
                    "meta_data": {
                        'repo': bug['repo'],
                        'base_commit': bug['base_commit'],
                        'problem_statement': bug['problem_statement'],
                        'patch': bug['patch'],
                        # 'gt_file_changes': gt_file_changes
                    }
                }
                
                with output_file_lock:
                    append_to_jsonl(loc_res, args.output_file)

                cost = calc_cost(args.model, total_prompt_tokens, total_completion_tokens)
                loc_res['usage'] = {'cost($)': f'{round(cost, 5)}', 'prompt_tokens': total_prompt_tokens,
                                    'completion_tokens': total_completion_tokens}
                loc_res['loc_trajs'] = loc_trajs
                traj_file = os.path.join(args.output_folder, 'loc_trajs.jsonl')
                with traj_file_lock:
                    append_to_jsonl(loc_res, traj_file)
        finally:
            if progress_queue is not None:
                progress_queue.put({'instance_id': instance_id, 'status': progress_status})
            if repo_ctx is not None:
                reset_instance_context(instance_id)


def localize(args):
    bench_data = load_benchmark_dataset(args.dataset, args.split)
    bench_tests = filter_dataset(bench_data, 'instance_id', args.used_list)
    if args.eval_n_limit:
        eval_n_limit = min(args.eval_n_limit, len(bench_tests))
        bench_tests = bench_tests.select(range(0, eval_n_limit))
        logging.info(f'Limiting evaluation to first {eval_n_limit} instances.')

    single_process_mode = args.num_processes == 1
    if single_process_mode:
        queue = Queue()
        progress_queue = Queue()
        output_file_lock, traj_file_lock = threading.Lock(), threading.Lock()
        log_queue = None
    else:
        manager = mp.Manager()
        queue = manager.Queue()
        progress_queue = manager.Queue()
        output_file_lock, traj_file_lock = manager.Lock(), manager.Lock()
        log_queue = manager.Queue()

    # collect processed instances
    # Resume from breakpoint + optionally rerun empty results
    processed_instance = []
    backup_loc_output = None
    backup_traj_output = None
    if os.path.exists(args.output_file):
        traj_file = os.path.join(args.output_folder, 'loc_trajs.jsonl')
        locs = load_jsonl(args.output_file)        
        if args.rerun_empty_location:
            traj_datas = load_jsonl(traj_file)
            backup_loc_output = backup_file(args.output_file)
            backup_traj_output = backup_file(traj_file)
            clear_file(args.output_file)
            clear_file(traj_file)
            for loc in locs:
                if loc['found_files'] != [[]]:
                    append_to_jsonl(loc, args.output_file)
                    processed_instance.append(loc['instance_id'])
                    
            for loc_traj in traj_datas:
                if loc_traj['found_files'] != [[]]:
                    append_to_jsonl(loc_traj, traj_file)
        else:
            processed_instance = [loc['instance_id'] for loc in locs]
    
    num_bugs = 0
    for bug in bench_tests:
        instance_id = bug["instance_id"]
        if instance_id in processed_instance:
        # if instance_id in processed_instance or instance_id in filtered_instances:
            print(f"instance {instance_id} has already been processed, skip.")
        else:
            queue.put(bug)
            num_bugs += 1

    if num_bugs == 0:
        if args.rerun_empty_location and backup_loc_output and backup_traj_output:
            try:
                delete_file(backup_loc_output)
                delete_file(backup_traj_output)
            except:
                return
        logging.info('No unprocessed instances to localize.')
        return

    queue_listener = None
    if log_queue is not None:
        queue_listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
    progress_stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=_consume_progress_events,
        args=(progress_queue, num_bugs, progress_stop_event),
        daemon=True,
    )
    if queue_listener is not None:
        queue_listener.start()
    progress_thread.start()
    try:
        if single_process_mode:
            logging.info('Running localization in single-process debug mode because --num_processes=1.')
            run_localize(
                0,
                args,
                queue,
                log_queue,
                output_file_lock,
                traj_file_lock,
                progress_queue,
            )
        else:
            mp.spawn(
                run_localize,
                nprocs=min(num_bugs, args.num_processes) if args.num_processes > 0 else num_bugs,
                args=(args, queue, log_queue, output_file_lock, traj_file_lock, progress_queue),
                join=True
            )
    finally:
        progress_stop_event.set()
        progress_thread.join()
        if queue_listener is not None:
            queue_listener.stop()
    
    if args.rerun_empty_location:
        try:
            delete_file(backup_loc_output)
            delete_file(backup_traj_output)
        except:
            return


def merge(args):
    args.merge_file = os.path.join(args.output_folder, 'merged_' + os.path.basename(args.output_file))
    
    if args.ranking_method == 'mrr':
        args.merge_file = args.merge_file.replace('.jsonl', f'_{args.ranking_method}.jsonl')
        
    clear_file(args.merge_file)
    with open(args.output_file, 'r') as file:
        for line in file:
            loc_data = json.loads(line)
            if loc_data['found_files'] == [[]]:
                loc_data['found_files'] = []
                loc_data['found_modules'] = []
                loc_data['found_entities'] = []
            else:
                loc_data['found_files'] = loc_data['found_files']
                loc_data['found_modules'] = loc_data['found_modules']
                loc_data['found_entities'] = loc_data['found_entities']
                ranked_files, ranked_modules, ranked_funcs = merge_sample_locations(loc_data['found_files'], 
                                                                    loc_data['found_modules'],
                                                                    loc_data['found_entities'],
                                                                    ranking_method=args.ranking_method,
                                                                    )
                loc_data['found_files'] = ranked_files
                loc_data['found_modules'] = ranked_modules
                loc_data['found_entities'] = ranked_funcs
            with open(args.merge_file, 'a') as f:
                f.write(json.dumps(loc_data) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--localize", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--use_example", action="store_true")
    parser.add_argument("--ranking_method", type=str, default='mrr',
                        choices=['mrr', 'majority'])
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="Hugging Face dataset name, a local .parquet file, or a load_from_disk dataset directory.",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--eval_n_limit", type=int, default=0)
    parser.add_argument("--used_list", type=str, default='selected_ids')
    
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument("--merge_file", type=str, default="merged_loc_outputs.jsonl")
    
    parser.add_argument(
        "--model", type=str,
        default="openai/gpt-4o-2024-05-13",
        # choices=["gpt-4o", 
        #          "azure/gpt-4o", "openai/gpt-4o-2024-05-13",
        #          "deepseek/deepseek-chat", "deepseek-ai/DeepSeek-R1",
        #          "litellm_proxy/claude-3-5-sonnet-20241022", "litellm_proxy/gpt-4o-2024-05-13", "litellm_proxy/o3-mini-2025-01-31",
        #          # fine-tuned model
        #          "openai/qwen-7B", "openai/qwen-7B-128k", "openai/ft-qwen-7B", "openai/ft-qwen-7B-128k",
        #          "openai/qwen-32B", "openai/qwen-32B-128k", "openai/ft-qwen-32B", "openai/ft-qwen-32B-128k",
        # ]
    )
    parser.add_argument("--use_function_calling", action="store_true",
                        help='Enable function calling features of LLMs. If disabled, codeact will be used to support function calling.')
    parser.add_argument("--simple_desc", action="store_true", 
                        help="Use simplified function descriptions due to certain LLM limitations. Set to False for better performance when using Claude.")
    
    parser.add_argument("--max_attempt_num", type=int, default=1, 
                        help='Only use in generating training trajectories.')
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--num_processes", type=int, default=-1)
    
    parser.add_argument("--log_level", type=str, default='INFO')
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--rerun_empty_location", action="store_true")
    args = parser.parse_args()

    args.output_file = os.path.join(args.output_folder, args.output_file)
    os.makedirs(args.output_folder, exist_ok=True)

    # write the arguments
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    logging.basicConfig(
        level=logging.getLevelName(args.log_level),
        format="%(asctime)s %(filename)s %(levelname)s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"{args.output_folder}/localize.log"),
            logging.StreamHandler()
        ]
    )
    
    if args.localize:
        localize(args)
    
    
    if args.merge:
        merge(args)


if __name__ == "__main__":

    start_time = time.time()
    main()
    end_time = time.time()
    logging.info("Total time: {:.4f} min".format((end_time - start_time)/60))
