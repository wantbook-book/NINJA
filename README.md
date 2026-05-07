<div align="center">
</div>

<h1 align="center">🥷 NINJA: A Navigator-Inspector Joint Architecture for Context-Efficient Issue Localization</h1>

Official codebase for the paper **"NINJA: A Navigator-Inspector Joint Architecture for Context-Efficient Issue Localization"**.

NINJA provides the data construction, supervised fine-tuning, reinforcement learning, inference, evaluation, and trajectory visualization pipeline for context-efficient issue localization. The training stack is built on [verl](https://github.com/verl-project/verl), while parts of the repository indexing, code graph, and localization utilities are adapted from [LocAgent](https://github.com/gersteinlab/LocAgent) and [RepoSearcher](https://github.com/Mizersy/RepoDeepSearch).

<a id="overview"></a>
## 🧭 Overview

**Abstract:** Recent advances in agent-based methods have demonstrated strong promise for issue localization, a critical prerequisite for software issue resolution. However, most existing agent-based methods rely on a single agent with a growing context, where long-context accumulation compresses the effective reasoning space. Meanwhile, the flat exploration structure hinders the balance between file-level breadth and function-level depth. To address these limitations, we propose NINJA, a hierarchical Navigator-INspector Joint Architecture for context-efficient issue localization. NINJA decomposes repository exploration into global navigation and local inspection: the navigator maintains the global search state, performs file-level search, and dispatches selected entry files, while inspectors independently conduct function-level exploration around the assigned files in separate contexts. Through multi-round interactions, inspectors return suspicious locations as feedback, and the navigator updates the global state to decide whether to continue exploration or finalize localization. This hierarchical design balances file-level breadth with function-level depth, while independent inspector contexts prevent local exploration traces from accumulating in a single growing context. To further strengthen both global coordination and local exploration, we introduce a two-stage agentic fine-tuning strategy. Extensive experiments across multiple benchmarks and LLM backbones show that NINJA consistently outperforms competitive baselines. Notably, after fine-tuning, Qwen3-Coder-30B-A3B-Instruct surpasses the strong closed-source Claude-Haiku-4.5 model.


<a id="contents"></a>
## 📚 Contents

- [🧭 Overview](#overview)
- [🗂️ Directory Overview](#directory-overview)
- [⚙️ Installation](#installation)
- [🏗️ Data Construction](#data-construction)
- [🧪 Agentic Fine-Tuning](#agentic-fine-tuning)
- [📊 Evaluation](#evaluation)
- [🔎 Trajectory Visualization](#trajectory-visualization)
- [⚖️ License](#license)
- [🙏 Acknowledgments](#acknowledgments)

<a id="directory-overview"></a>
## 🗂️ Directory Overview

- `verl/`: training framework code used for SFT and RL, based on verl.
- `LocAgent/`: LocAgent-derived utilities for code graph construction, ground-truth edited-location extraction, and support-location extraction.
- `tools/`: tool implementations used by the agents, including RepoSearch-style repository structure parsing under `tools/RepoSearch`.
- `examples/code_localization/`: training data builders, RL scripts, and tool configuration files for Navigator and Inspector training.
- `examples/data_preprocess/`: benchmark conversion and sampling scripts for SWE-bench-style tool-agent data.
- `evaluation/`: multi-agent inference, evaluation, SFT data construction, data validation, and analysis utilities.
- `prompts/`: prompt templates for the multi-agent and repository-search workflows.
- `visual_analysis/`: browser-based viewer for inspecting Navigator and Inspector inference trajectories.
- `scripts/`: installation, model merging, diagnostics, and trainer utility scripts.

<a id="installation"></a>
## ⚙️ Installation

Python 3.12 is recommended. The environment has been tested on Ubuntu 22.04 with CUDA-enabled GPUs.

```bash
uv venv --python 3.12
source .venv/bin/activate

uv pip install pre-commit hydra-core
pre-commit install

USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
uv pip install -e .
uv pip install -r requirements-code.txt
uv pip install flash-attn --no-build-isolation
```

Adjust the CUDA, vLLM, SGLang, and FlashAttention versions if your hardware or driver stack requires a different build.

<a id="data-construction"></a>
## 🏗️ Data Construction

NINJA uses several intermediate artifacts: repository structure JSON files, code graph indexes, ground-truth locations, SFT trajectories, and RL parquet datasets. The scripts below are templates, so update their paths or pass the documented environment variables before running them.

### 🧱 Repository Structure Construction

The tools `get_file_functions`, `get_file_classes`, `get_methods_of_class`, `get_code_of_function`, and `get_code_of_class_method` rely on repository structure JSON files. Build these files with:

```bash
INPUT_FILE=/path/to/input.parquet \
PLAYGROUND_DIR=/path/to/playground \
OUTPUT_DIR=/path/to/repo_strucs \
bash tools/RepoSearch/scripts_template/run_get_repo_structure_batch.sh
```

The input parquet should contain the benchmark instances to parse. If `base_commit` is available, the parser checks out the corresponding version before extracting repository structure.

### 🕸️ Code Graph Construction

The `get_callers` and `get_callees` tools use LocAgent-style dependency graphs. Build graph indexes with:

```bash
DATASET=czlll/SWE-bench_Lite \
REPO_PATH=/path/to/repos \
INDEX_DIR=/path/to/index_data \
bash LocAgent/scripts_template/run_batch_build_graph.sh
```

Use `INPUT_FILE=/path/to/input.parquet` instead of `DATASET=...` when you want to build graphs from a local benchmark file.

### 🎯 Ground-Truth Location Extraction

Evaluation uses edited locations extracted from patches as ground-truth locations. Edit the variables at the top of the script, then run:

```bash
bash LocAgent/scripts_template/run_gen_oracle_locations.sh
```

The extraction pipeline can also produce support locations. Support locations are not treated as ground-truth locations for now.

### 🧾 SFT Data Construction

Build Navigator and Inspector SFT datasets from collected trajectories:

```bash
INPUT_FILE=/path/to/navigator_trajs.jsonl \
OUTPUT_DIR=/path/to/navigator_sft \
bash evaluation/scripts_template/run_build_navigator_sft_data.sh

INPUT_FILE=/path/to/inspector_trajs.jsonl \
OUTPUT_DIR=/path/to/inspector_sft \
bash evaluation/scripts_template/run_build_inspector_sft_data.sh
```

After construction, validate the message format:

```bash
INPUT_FILE=/path/to/inspector_sft/train.parquet \
OUTPUT_FILE=/path/to/wrong_format_contents.jsonl \
bash evaluation/scripts_template/run_check_sft_format.sh
```

For production-quality runs, manually inspect representative samples after the rule-based format check.

### 🏋️ RL Data Construction

Build Navigator and Inspector RL datasets from preprocessed SWE-bench-style data:

```bash
INPUT_FILE=/path/to/preprocessed_swebench.parquet \
OUTPUT_DIR=/path/to/code_loc/navigator_from_swebench \
bash examples/code_localization/scripts_template/run_build_navigator_training_data_from_swebench.sh

INPUT_FILE=/path/to/preprocessed_swebench.parquet \
OUTPUT_DIR=/path/to/code_loc/inspector_from_swebench \
bash examples/code_localization/scripts_template/run_build_inspector_training_data_from_swebench.sh
```

To cap the number of samples per repository:

```bash
INPUT=/path/to/generated_swebench_tool_agent_data \
SPLIT=train \
OUTPUT_DIR=/path/to/sample_output \
SAMPLE_PER_REPO=2 \
bash examples/data_preprocess/scripts_template/run_sample_swebench_tool_agent_data.sh
```

### 🧩 Evaluation Data Construction

Convert a benchmark dataset into the tool-agent format used by NINJA:

```bash
bash examples/data_preprocess/scripts_template/run_swebench_tool_agent_loop_reposearch.sh
```

Before running the script, set the benchmark source, output directory, repository structure directory, split, and ground-truth location file in the script template.

<a id="agentic-fine-tuning"></a>
## 🧪 Agentic Fine-Tuning

### 📘 SFT

After constructing the SFT datasets, run the multi-turn code-agent SFT script:

```bash
bash examples/sft/multiturn/run_code_agent_sft.sh
```

Update the dataset paths, model path, output directory, and distributed training options in the script before launching a full run.

### 🧬 RL with DAPO

Start a Ray cluster:

```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
```

Train the Inspector first:

```bash
RUNTIME_ENV=examples/code_localization/scripts_template/inspector_dapo_runtime_env.yaml \
bash examples/code_localization/scripts_template/run_train_inspector_dapo.sh
```

Then serve the trained Inspector checkpoint with vLLM, and make sure the serving port, model name, and `REMOTE_API_BASE` used by the Navigator script are consistent:

```bash
bash evaluation/scripts_template/deploy_vllm.sh
```

Train the Navigator:

```bash
RUNTIME_ENV=examples/code_localization/scripts_template/navigator_dapo_runtime_env.yaml \
REMOTE_API_BASE=http://localhost:8001/v1 \
REMOTE_API_KEY=EMPTY \
bash examples/code_localization/scripts_template/run_train_navigator_dapo.sh
```

### 🧠 RL with PPO

Start Ray as above, then train the Inspector and Navigator with PPO:

```bash
RUNTIME_ENV=examples/code_localization/scripts_template/inspector_runtime_env.yaml \
bash examples/code_localization/scripts_template/run_train_inspector.sh

RUNTIME_ENV=examples/code_localization/scripts_template/navigator_runtime_env.yaml \
REMOTE_API_BASE=http://localhost:8001/v1 \
REMOTE_API_KEY=EMPTY \
bash examples/code_localization/scripts_template/run_train_navigator.sh
```

As with DAPO, the Navigator RL stage requires a running Inspector endpoint.

<a id="evaluation"></a>
## 📊 Evaluation

Deploy the Navigator and Inspector models with vLLM. The template below should be copied or edited for each model so that the model path, served name, tensor parallelism, data parallelism, and port match your environment:

```bash
bash evaluation/scripts_template/deploy_vllm.sh
```

Run multi-agent inference:

```bash
INPUT_FILE=/path/to/eval_data.parquet \
BASE_OUTPUT_DIR=/path/to/ninja_outputs \
RUN_NAME=ninja_eval \
MODEL_NAME=navigator_model_name \
MODEL_BACKEND=openai \
BASE_URL=http://localhost:8000/v1 \
API_KEY=EMPTY \
SUB_AGENT_MODEL_NAME=inspector_model_name \
SUB_AGENT_MODEL_BACKEND=openai \
SUB_AGENT_BASE_URL=http://localhost:8001/v1 \
SUB_AGENT_API_KEY=EMPTY \
GRAPH_INDEX_DIR=/path/to/graph_index \
PROJECT_FILE_LOC=/path/to/repo_strucs \
bash evaluation/scripts_template/run_multi_agent_inference.sh
```

Evaluate inference outputs:

```bash
bash evaluation/scripts_template/run_evaluate_multifiles.sh
```

Edit the `INPUT_FILES` array in `run_evaluate_multifiles.sh` to point to the generated `traj/trajs.jsonl` files before running the evaluation script.

<a id="trajectory-visualization"></a>
## 🔎 Trajectory Visualization

Use the browser viewer to inspect Navigator and Inspector trajectories from an inference output directory.

Open `visual_analysis/index.html` directly in a browser, or serve it locally:

```bash
python -m http.server -d visual_analysis 8000
```

Then open `http://127.0.0.1:8000/` and select an inference output directory such as:

```text
<base_output_dir>/<run_name>/<timestamp>/
```

<a id="license"></a>
## ⚖️ License

This repository is released under the [Apache-2.0 License](LICENSE). See [Notice.txt](Notice.txt) for upstream attribution notices.

| Component | License |
| --- | --- |
| NINJA codebase | [Apache-2.0](LICENSE) |
| LocAgent-derived components | [Apache-2.0](LocAgent/LICENSE) |
| verl-derived components | Apache-2.0, with upstream notices preserved |

<a id="acknowledgments"></a>
## 🙏 Acknowledgments

- [verl](https://github.com/verl-project/verl) for the RL training framework.
- [LocAgent](https://github.com/gersteinlab/LocAgent) for graph-guided localization utilities and data construction references.
- [RepoSearcher](https://github.com/Mizersy/RepoDeepSearch) for structure-aware code search and tool implementation support.
