# Examples Data Preprocess

This directory contains small dataset-conversion scripts used to prepare data
for tool-using localization agents.

This README focuses on:

- `extract_ground_truth_from_patch.py`
- `swebench_add_edit_and_added_functions.py`
- `swebench_tool_agent_loop_reposearch.py`
- `sample_swebench_tool_agent_data.py`

## `extract_ground_truth_from_patch.py`

### Goal

Extract edited function locations from patch text stored in a parquet dataset.

The script parses unified diff content and produces function-level locations in
the form:

```text
path/to/file.py:function_name
path/to/file.py:ClassName.method_name
```

### What It Does

Given a parquet file, the script:

1. reads the patch from each row
2. parses changed hunks and function/class definitions from the diff
3. extracts edited function locations
4. filters out pure newly-added functions from the default extracted result
5. writes the extracted function list back into the parquet
6. optionally updates `reward_model.ground_truth`
7. writes summary statistics to a JSON file

Important limitation:

- extraction is based on the patch text itself
- it does not open the original repository to verify that the returned function
  really exists in the checked-out repo snapshot

### Input

Required:

- `--parquet`: input parquet file

Common optional parameters:

- `--output`: output parquet file
- `--edit-functions-key`: column name used to store extracted functions
- `--update-reward-model`: overwrite `reward_model.ground_truth`
- `--drop-empty`: drop rows whose extracted result is empty
- `--drop-new-functions-only`: drop rows whose touched functions are all newly
  added
- `--stats-output`: output JSON path for extraction statistics

Testing/debugging parameters:

- `--test`: compare extracted result with `reward_model.ground_truth`
- `--mismatch-output`: JSONL file for mismatch details
- `--path-mismatch-output`: JSON file for path mismatch stats

### Output

Main output:

- a parquet file containing the extracted `edit_functions`

Additional outputs:

- `*_edit_function_stats.json`
  Contains counts such as:
  - `empty_count`
  - `new_functions_only_count`
  - `dropped_empty`
  - `dropped_new_only`
- `*_mismatches.jsonl` when `--test` is enabled
- `*_path_mismatch_stats.json` when patch path mismatches are observed

### Usage

Direct command:

```bash
python3 -m examples.data_preprocess.extract_ground_truth_from_patch \
  --parquet /path/to/input.parquet \
  --output /path/to/output.parquet
```

Template script:

```bash
bash examples/data_preprocess/scripts_template/run_extract_ground_truth_from_patch.sh \
  /path/to/input.parquet \
  /path/to/output.parquet
```

## `swebench_add_edit_and_added_functions.py`

### Goal

Load a SWE-bench-style dataset and add two normalized fields:

- `edit_functions`
- `added_functions`

Both use the format:

```text
file1.py:func1
file1.py:ClassName.method2
file2.py:func3
```

### What It Does

For each row, the script:

1. loads the dataset from Hugging Face or a local dataset path
2. parses `patch`
3. writes modified existing functions into `edit_functions`
4. writes pure newly-added functions into `added_functions`
5. saves one parquet file per split
6. writes one sample JSON and one stats JSON

### Input

Main parameters:

- `--data_source`: Hugging Face dataset name
- `--local_dataset_path`: local dataset path, used instead of `data_source`
- `--local_save_dir`: output directory
- `--hdfs_dir`: optional HDFS copy target

### Output

For each split, the script writes:

- `<local_save_dir>/<split>.parquet`
- `<local_save_dir>/<split>_sample.json`

It also writes:

- `<local_save_dir>/function_field_stats.json`

### Usage

Direct command:

```bash
python3 -m examples.data_preprocess.swebench_add_edit_and_added_functions \
  --data_source princeton-nlp/SWE-bench_Verified \
  --local_save_dir /path/to/output_dir
```

Template script:

```bash
SWEBENCH_DATA_SOURCE=princeton-nlp/SWE-bench_Verified \
SWEBENCH_LOCAL_SAVE_DIR=/path/to/output_dir \
bash examples/data_preprocess/scripts_template/run_swebench_add_edit_and_added_functions.sh
```

If you want to use a local dataset path instead of Hugging Face:

```bash
SWEBENCH_LOCAL_DATASET_PATH=/path/to/local_dataset \
SWEBENCH_LOCAL_SAVE_DIR=/path/to/output_dir \
bash examples/data_preprocess/scripts_template/run_swebench_add_edit_and_added_functions.sh
```

## `swebench_tool_agent_loop_reposearch.py`

### Goal

Convert a SWE-bench-style dataset into parquet data for a tool-using agent that
searches repositories with the repo-search tool set.

The generated data is suitable for tool-agent style rollouts where the model is
given:

- a system prompt
- a user prompt containing the issue report and repo structure
- tool kwargs for repo-search tools

### What It Does

For each dataset row, the script:

1. loads the issue instance from Hugging Face or a local dataset path
2. reads the precomputed repo structure JSON from `project_file_loc/<instance_id>.json`
3. filters non-Python and test files from that structure
4. constructs a prompt pair:
   - system prompt from `prompts/repo_search/system_prompt.txt`
   - user prompt from `prompts/repo_search/task_prompt.txt`
5. prepares repo-search tool kwargs for the signature-based tool config
6. writes one parquet file per split
7. writes one sampled JSON record per split

### Ground Truth Behavior

`reward_model.ground_truth` is built as follows:

1. use `example["edit_functions"]` if it is non-empty
2. otherwise, fall back to `extract_edit_functions_from_patch(example["patch"])`

So this script can still produce ground truth when the raw dataset does not
already contain `edit_functions`.

### Repo-Search Tools

The generated `tools_kwargs` match the signature-based repo-search config:

- `get_methods_of_class`
- `get_file_functions`
- `get_file_classes`
- `get_code_of_class_function`
- `get_code_of_file_function`
- `exit`

This is intended to align with:

- `examples/sglang_multiturn/config/tool_config/repo_search_tool_config_signature.yaml`

### Input

Main parameters:

- `--data_source`: Hugging Face dataset name
- `--local_dataset_path`: local dataset path, used instead of `data_source`
- `--local_save_dir`: output directory for converted parquet files
- `--project_file_loc`: directory containing repo structure JSON files
- `--hdfs_dir`: optional HDFS copy target

Expected repo structure files:

```text
<project_file_loc>/<instance_id>.json
```

### Output

For each dataset split, the script writes:

- `<local_save_dir>/<split>.parquet`
- `<local_save_dir>/<split>_sample.json`

Each processed row contains fields such as:

- `data_source`
- `agent_name`
- `prompt`
- `ability`
- `reward_model.ground_truth`
- `extra_info.instance_id`
- `extra_info.repo`
- `extra_info.base_commit`
- `extra_info.tools_kwargs`

### Usage

Direct command:

```bash
python3 -m examples.data_preprocess.swebench_tool_agent_loop_reposearch \
  --data_source czlll/Loc-Bench_V1 \
  --local_save_dir /path/to/output_dir \
  --project_file_loc /path/to/structure_json_dir
```

Template script:

```bash
SWEBENCH_PROJECT_FILE_LOC=/path/to/structure_json_dir \
SWEBENCH_LOCAL_SAVE_DIR=/path/to/output_dir \
bash examples/data_preprocess/scripts_template/run_swebench_tool_agent_loop_reposearch.sh
```

If you want to use a local dataset path instead of Hugging Face:

```bash
SWEBENCH_LOCAL_DATASET_PATH=/path/to/local_dataset \
SWEBENCH_PROJECT_FILE_LOC=/path/to/structure_json_dir \
bash examples/data_preprocess/scripts_template/run_swebench_tool_agent_loop_reposearch.sh
```

## `sample_swebench_tool_agent_data.py`

### Goal

Sample and inspect parquet data generated by `swebench_tool_agent_loop.py` or
`swebench_tool_agent_loop_reposearch.py`.

### What It Does

Given a generated parquet file, the script:

1. randomly samples `N` rows per repository, default `N=2`
2. writes sampled `instance_id` values to a txt file
3. optionally extracts rows by an existing `instance_id` txt file
4. exports repository counts
5. exports `reward_model.ground_truth` location-count statistics

### Output

The output directory contains:

- `sampled_per_repo.parquet`
- `sampled_per_repo_sample.json`
- `sampled_instance_ids.txt`
- `repo_counts.json`
- `ground_truth_stats.json`
- `summary.json`

When `--instance_id_file` is provided, it also writes:

- `selected_by_instance_ids.parquet`
- `selected_by_instance_ids_sample.json`
- `selected_instance_ids.txt`
- `missing_instance_ids.txt`

### Usage

Direct command:

```bash
python3 -m examples.data_preprocess.sample_swebench_tool_agent_data \
  --input /path/to/generated_data/test.parquet \
  --output_dir /path/to/sample_output \
  --sample_per_repo 2 \
  --seed 42
```

If `--input` is a directory, pass the split name:

```bash
python3 -m examples.data_preprocess.sample_swebench_tool_agent_data \
  --input /path/to/generated_data \
  --split test \
  --output_dir /path/to/sample_output
```

Extract rows by an existing instance-id txt file:

```bash
python3 -m examples.data_preprocess.sample_swebench_tool_agent_data \
  --input /path/to/generated_data/test.parquet \
  --output_dir /path/to/sample_output \
  --instance_id_file /path/to/instance_ids.txt
```

Template script:

```bash
INPUT=/path/to/generated_data \
SPLIT=test \
OUTPUT_DIR=/path/to/sample_output \
SAMPLE_PER_REPO=2 \
bash examples/data_preprocess/scripts_template/run_sample_swebench_tool_agent_data.sh
```
