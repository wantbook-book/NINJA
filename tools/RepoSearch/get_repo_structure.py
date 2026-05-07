import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

GIT_POST_BUFFER = "5242880000"


def repo_to_top_folder(repo: str) -> str:
    return repo.rsplit("/", maxsplit=1)[-1]


def normalize_commit_id(commit_id: object) -> Optional[str]:
    if commit_id is None:
        return None

    normalized_commit_id = str(commit_id).strip()
    if normalized_commit_id.lower() in {"", "nan", "none", "null", "<na>"}:
        return None

    return normalized_commit_id


def _configure_git_post_buffer(env=None) -> None:
    for protocol in ("https", "http"):
        subprocess.run(
            ["git", "config", "--global", f"{protocol}.postBuffer", GIT_POST_BUFFER],
            check=True,
            env=env,
        )


def add_detached_worktree(repo_path: str, worktree_path: str, commit_id: str) -> None:
    """Create a detached worktree for the specified commit."""
    print(
        f"Adding detached worktree for commit {commit_id} at {worktree_path} "
        f"from repository at {repo_path}..."
    )
    subprocess.run(
        [
            "git",
            "-C",
            repo_path,
            "worktree",
            "add",
            "--detach",
            worktree_path,
            commit_id,
        ],
        check=True,
    )
    print("Detached worktree added successfully.")


def remove_worktree(repo_path: str, worktree_path: str) -> None:
    """Remove a previously added git worktree."""
    print(f"Removing worktree at {worktree_path} from repository at {repo_path}...")
    subprocess.run(
        ["git", "-C", repo_path, "worktree", "remove", "--force", worktree_path],
        check=True,
    )
    print("Worktree removed successfully.")


def clone_repo(repo_name: str, repo_playground: str) -> None:
    repo_folder = Path(repo_playground) / repo_to_top_folder(repo_name)
    print(
        f"Cloning repository from https://github.com/{repo_name}.git to {repo_folder}..."
    )
    env = os.environ.copy()
    _configure_git_post_buffer(env=env)
    subprocess.run(
        [
            "git",
            "clone",
            f"https://github.com/{repo_name}.git",
            str(repo_folder),
        ],
        check=True,
        env=env,
    )
    print("Repository cloned successfully.")


def copy_repo_checkout(repo_path: str, worktree_path: str) -> None:
    print(f"Copying repository from {repo_path} to {worktree_path}...")
    shutil.copytree(
        repo_path,
        worktree_path,
        ignore=shutil.ignore_patterns(".git"),
        symlinks=True,
    )
    print("Repository copied successfully.")


def get_project_structure_from_scratch(
    repo_name: str,
    commit_id: Optional[str],
    instance_id: str,
    repo_playground: str,
):
    repo_playground_path = Path(repo_playground)
    repo_playground_path.mkdir(parents=True, exist_ok=True)

    repo_folder = repo_playground_path / repo_to_top_folder(repo_name)
    if not repo_folder.exists():
        clone_repo(repo_name, repo_playground)

    normalized_commit_id = normalize_commit_id(commit_id)

    with tempfile.TemporaryDirectory(dir=repo_playground) as temp_dir:
        working_repo = Path(temp_dir) / repo_to_top_folder(repo_name)
        worktree_added = False
        try:
            if normalized_commit_id:
                add_detached_worktree(
                    str(repo_folder), str(working_repo), normalized_commit_id
                )
                worktree_added = True
            else:
                copy_repo_checkout(str(repo_folder), str(working_repo))
            structure = create_structure(str(working_repo))
        finally:
            if worktree_added:
                remove_worktree(str(repo_folder), str(working_repo))

    return {
        "repo": repo_name,
        "base_commit": normalized_commit_id,
        "structure": structure,
        "instance_id": instance_id,
    }


def parse_python_file(file_path, file_content=None):
    """Parse a Python file to extract class and function definitions."""
    if file_content is None:
        try:
            with open(file_path, "r") as file:
                file_content = file.read()
                parsed_data = ast.parse(file_content)
        except Exception as exc:
            print(f"Error in file {file_path}: {exc}")
            return [], [], ""
    else:
        try:
            parsed_data = ast.parse(file_content)
        except Exception as exc:
            print(f"Error in file {file_path}: {exc}")
            return [], [], ""

    file_lines = file_content.splitlines()
    class_info = []
    function_names = []
    class_methods = set()

    for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = []
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    methods.append(
                        {
                            "name": child.name,
                            "start_line": child.lineno,
                            "end_line": child.end_lineno,
                            "text": file_lines[child.lineno - 1 : child.end_lineno],
                        }
                    )
                    class_methods.add(child.name)
            class_info.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": file_lines[node.lineno - 1 : node.end_lineno],
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.FunctionDef):
            if node.name not in class_methods:
                function_names.append(
                    {
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "text": file_lines[node.lineno - 1 : node.end_lineno],
                    }
                )

    return class_info, function_names, file_lines


def create_structure(directory_path):
    """Create the structure of the repository directory by parsing Python files."""
    structure = {}
    repo_name = os.path.basename(directory_path)

    for root, _, files in os.walk(directory_path):
        relative_root = os.path.relpath(root, directory_path)
        if relative_root == ".":
            relative_root = repo_name

        curr_struct = structure
        for part in relative_root.split(os.sep):
            if part not in curr_struct:
                curr_struct[part] = {}
            curr_struct = curr_struct[part]

        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_python_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            else:
                curr_struct[file_name] = {}

    return structure


def get_directory_size_gb(directory_path):
    """Return directory size in GB using du -sk."""
    output = subprocess.check_output(["du", "-sk", directory_path], text=True)
    size_kb = int(output.split()[0])
    return size_kb / (1024 * 1024)


def _load_allowed_instance_ids(
    instance_id_list_file: Optional[str],
) -> Optional[set[str]]:
    if not instance_id_list_file:
        return None

    with open(instance_id_list_file, "r") as file:
        return {line.strip() for line in file if line.strip()}


def _write_structure_record(record: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(record, file)


def _write_structure_record_to_dir(record: dict, output_dir: Optional[str]) -> None:
    if not output_dir:
        return
    output_path = Path(output_dir) / f"{record['instance_id']}.json"
    _write_structure_record(record, output_path)


def process_batch(
    *,
    input_file: str,
    playground_dir: str,
    output_dir: Optional[str] = None,
    error_file: Optional[str] = None,
    instance_id_list_file: Optional[str] = None,
    limit: Optional[int] = None,
) -> None:
    import pandas as pd
    from tqdm import tqdm

    allowed_instance_ids = _load_allowed_instance_ids(instance_id_list_file)
    df = pd.read_parquet(input_file)
    processed = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        instance_id = row["instance_id"]
        if allowed_instance_ids is not None and instance_id not in allowed_instance_ids:
            continue
        if limit is not None and processed >= limit:
            break

        repo_name = row["repo"]
        base_commit = row.get("base_commit")
        try:
            record = get_project_structure_from_scratch(
                repo_name=repo_name,
                commit_id=base_commit,
                instance_id=instance_id,
                repo_playground=playground_dir,
            )
            if os.path.exists(playground_dir):
                cumulative_size_gb = get_directory_size_gb(playground_dir)
                print(
                    f"{instance_id} cumulative_repo_playground_gb: "
                    f"{cumulative_size_gb:.2f}"
                )
            _write_structure_record_to_dir(record, output_dir)
            processed += 1
        except Exception as exc:
            print(f"Failed to build structure for {instance_id}: {exc}")
            if error_file:
                error_path = Path(error_file)
                error_path.parent.mkdir(parents=True, exist_ok=True)
                with open(error_path, "a") as file:
                    file.write(f"{instance_id}\n")


def process_single(
    *,
    repo_name: str,
    commit_id: Optional[str],
    instance_id: str,
    playground_dir: str,
    output_file: Optional[str] = None,
) -> dict:
    record = get_project_structure_from_scratch(
        repo_name=repo_name,
        commit_id=commit_id,
        instance_id=instance_id,
        repo_playground=playground_dir,
    )
    if output_file:
        _write_structure_record(record, Path(output_file))
    else:
        print(json.dumps(record))
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build Python repository structure snapshots for one repository or a "
            "batch of instances."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    batch_parser = subparsers.add_parser(
        "batch",
        help="Build structures for all instances in a parquet file.",
    )
    batch_parser.add_argument(
        "--input-file",
        default=os.environ.get("INPUT_FILE"),
        help="Input parquet file. Defaults to environment variable INPUT_FILE.",
    )
    batch_parser.add_argument(
        "--playground-dir",
        default=os.environ.get("PLAYGROUND_DIR"),
        help=(
            "Directory used to cache cloned repos and temporary working copies. "
            "Defaults to PLAYGROUND_DIR."
        ),
    )
    batch_parser.add_argument(
        "--output-dir",
        default=os.environ.get("PROJECT_FILE_LOC"),
        help="Directory used to store generated JSON files. Defaults to PROJECT_FILE_LOC.",
    )
    batch_parser.add_argument(
        "--error-file",
        default=os.environ.get("ERROR_FILE_LOC"),
        help="Append failed instance IDs to this file. Defaults to ERROR_FILE_LOC.",
    )
    batch_parser.add_argument(
        "--instance-id-list-file",
        default=os.environ.get("INSTANCE_ID_LIST_FILE"),
        help=(
            "Optional file containing allowed instance IDs, one per line. "
            "Defaults to INSTANCE_ID_LIST_FILE."
        ),
    )
    batch_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for the number of selected instances to process.",
    )

    single_parser = subparsers.add_parser(
        "single",
        help="Build the structure for one repo/commit pair.",
    )
    single_parser.add_argument(
        "--repo",
        required=True,
        help="GitHub repo name, e.g. psf/requests.",
    )
    single_parser.add_argument(
        "--commit",
        default=None,
        help=(
            "Optional commit SHA to checkout. If omitted, parse the cloned "
            "repository's current checkout."
        ),
    )
    single_parser.add_argument(
        "--instance-id",
        required=True,
        help="Identifier used in the JSON output.",
    )
    single_parser.add_argument(
        "--playground-dir",
        default=os.environ.get("PLAYGROUND_DIR"),
        help=(
            "Directory used to cache cloned repos and temporary working copies. "
            "Defaults to PLAYGROUND_DIR."
        ),
    )
    single_parser.add_argument(
        "--output-file",
        default=None,
        help="Optional JSON output file. If omitted, the JSON is printed to stdout.",
    )

    return parser


def _validate_required_args(args: argparse.Namespace) -> None:
    if args.command == "batch":
        if not args.input_file:
            raise SystemExit(
                "batch mode requires --input-file or environment variable INPUT_FILE"
            )
        if not args.playground_dir:
            raise SystemExit(
                "batch mode requires --playground-dir or environment variable PLAYGROUND_DIR"
            )
    elif args.command == "single":
        if not args.playground_dir:
            raise SystemExit(
                "single mode requires --playground-dir or environment variable PLAYGROUND_DIR"
            )


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if not argv and os.environ.get("INPUT_FILE"):
        argv = ["batch"]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        raise SystemExit(1)

    _validate_required_args(args)

    if args.command == "batch":
        process_batch(
            input_file=args.input_file,
            playground_dir=args.playground_dir,
            output_dir=args.output_dir,
            error_file=args.error_file,
            instance_id_list_file=args.instance_id_list_file,
            limit=args.limit,
        )
    elif args.command == "single":
        process_single(
            repo_name=args.repo,
            commit_id=args.commit,
            instance_id=args.instance_id,
            playground_dir=args.playground_dir,
            output_file=args.output_file,
        )


if __name__ == "__main__":
    main()
