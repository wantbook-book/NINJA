import logging
import os
import subprocess
import time
import shutil
import uuid
import hashlib
import fcntl
from contextlib import contextmanager
logger = logging.getLogger(__name__)


def normalize_commit_hash(commit_hash: object) -> str | None:
    if commit_hash is None:
        return None

    normalized_commit_hash = str(commit_hash).strip()
    if normalized_commit_hash.lower() in {"", "nan", "none", "null", "<na>"}:
        return None

    return normalized_commit_hash


def get_repo_dir_name(repo: str):
    return repo.split("/")[-1]
    # return repo.replace("/", "_")

# def repo_to_top_folder(repo):
#     return repo.split("/")[-1]


def _run_git_command(args, cwd=None):
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def _log_completed_process(result):
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.stderr:
        logger.info(result.stderr.strip())


def _repo_lock_path(repo_dir: str) -> str:
    parent_dir = os.path.dirname(os.path.abspath(repo_dir))
    lock_dir = os.path.join(parent_dir, ".locks")
    os.makedirs(lock_dir, exist_ok=True)
    repo_name = os.path.basename(os.path.abspath(repo_dir)) or "repo"
    digest = hashlib.sha1(os.path.abspath(repo_dir).encode("utf-8")).hexdigest()[:12]
    return os.path.join(lock_dir, f"{repo_name}_{digest}.lock")


@contextmanager
def repo_admin_lock(repo_dir: str):
    lock_path = _repo_lock_path(repo_dir)
    with open(lock_path, "w", encoding="utf-8") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def setup_github_repo(repo: str, base_commit: str, base_dir: str = "/tmp/repos") -> str:
    repo_name = get_repo_dir_name(repo)
    repo_url = f"https://github.com/{repo}.git"
    path = f"{base_dir}/{repo_name}"
    normalized_base_commit = normalize_commit_hash(base_commit)
    logger.info(
        "Clone Github repo %s to %s and checkout commit %s",
        repo_url,
        path,
        normalized_base_commit,
    )
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Directory '{path}' was created.")
    maybe_clone(repo_url, path)
    if normalized_base_commit:
        with repo_admin_lock(path):
            checkout_commit(path, normalized_base_commit)
    else:
        logger.info(
            "No base commit provided for %s; using the cloned repository's current checkout at %s",
            repo,
            path,
        )
    return path


def setup_github_repo_worktree(
    repo: str,
    base_commit: str,
    base_dir: str = "/tmp/repos",
) -> tuple[str, str]:
    repo_name = get_repo_dir_name(repo)
    repo_url = f"https://github.com/{repo}.git"
    repo_cache_root = base_dir
    repo_dir = os.path.join(repo_cache_root, repo_name)
    worktree_root = os.path.join(base_dir, "worktrees")
    normalized_base_commit = normalize_commit_hash(base_commit)

    logger.info(
        "Clone Github repo %s into cache %s and prepare checkout for commit %s",
        repo_url,
        repo_dir,
        normalized_base_commit,
    )
    os.makedirs(repo_cache_root, exist_ok=True)
    os.makedirs(worktree_root, exist_ok=True)
    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir)
        logger.info("Directory '%s' was created.", repo_dir)

    maybe_clone(repo_url, repo_dir)
    if not normalized_base_commit:
        copy_dir = os.path.join(
            worktree_root,
            f"{repo_name}_head_{uuid.uuid4().hex[:8]}",
        )
        logger.info(
            "No base commit provided for %s; copying cloned repository from %s to %s",
            repo,
            repo_dir,
            copy_dir,
        )
        shutil.copytree(
            repo_dir,
            copy_dir,
            ignore=shutil.ignore_patterns(".git"),
            symlinks=True,
        )
        return repo_dir, copy_dir

    ensure_commit_available(repo_dir, normalized_base_commit)

    worktree_dir = os.path.join(
        worktree_root,
        f"{repo_name}_{normalized_base_commit[:12]}_{uuid.uuid4().hex[:8]}",
    )
    with repo_admin_lock(repo_dir):
        result = _run_git_command(
            ["git", "worktree", "add", "--detach", worktree_dir, normalized_base_commit],
            cwd=repo_dir,
        )
        _log_completed_process(result)
    return repo_dir, worktree_dir


def maybe_clone(repo_url, repo_dir, max_retries=3, retry_delay=2):
    with repo_admin_lock(repo_dir):
        if os.path.exists(f"{repo_dir}/.git"):
            return

        for attempt in range(1, max_retries + 1):
            logger.info(
                "Cloning repo '%s' to '%s' (attempt %s/%s)",
                repo_url,
                repo_dir,
                attempt,
                max_retries,
            )
            try:
                result = subprocess.run(
                    ["git", "clone", repo_url, repo_dir],
                    check=True,
                    text=True,
                    capture_output=True,
                )
                if result.stdout:
                    logger.info(result.stdout.strip())
                if result.stderr:
                    logger.info(result.stderr.strip())
                logger.info(f"Repo '{repo_url}' was cloned to '{repo_dir}'")
                return
            except subprocess.CalledProcessError as e:
                logger.error(
                    "git clone failed for '%s' to '%s' on attempt %s/%s: %s",
                    repo_url,
                    repo_dir,
                    attempt,
                    max_retries,
                    e,
                )
                if e.stdout:
                    logger.error("git clone stdout: %s", e.stdout.strip())
                if e.stderr:
                    logger.error("git clone stderr: %s", e.stderr.strip())

                # Remove a partially created repo so the next retry starts clean.
                if os.path.exists(repo_dir) and not os.path.exists(f"{repo_dir}/.git"):
                    shutil.rmtree(repo_dir, ignore_errors=True)

                if attempt == max_retries:
                    raise

                time.sleep(retry_delay)


def pull_latest(repo_dir):
    subprocess.run(
        ["git", "pull"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def clean_and_reset_state(repo_dir):
    subprocess.run(
        ["git", "clean", "-fd"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "reset", "--hard"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def create_branch(repo_dir, branch_name):
    try:
        subprocess.run(
            ["git", "branch", branch_name],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr)
        raise e


def create_and_checkout_branch(repo_dir, branch_name):
    try:
        branches = subprocess.run(
            ["git", "branch"],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.split("\n")
        branches = [branch.strip() for branch in branches]
        if branch_name in branches:
            subprocess.run(
                ["git", "checkout", branch_name],
                cwd=repo_dir,
                check=True,
                text=True,
                capture_output=True,
            )
        else:
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=repo_dir,
                check=True,
                text=True,
                capture_output=True,
            )
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr)
        raise e


def commit_changes(repo_dir, commit_message):
    subprocess.run(
        ["git", "commit", "-m", commit_message, "--no-verify"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def checkout_branch(repo_dir, branch_name):
    subprocess.run(
        ["git", "checkout", branch_name],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def push_branch(repo_dir, branch_name):
    subprocess.run(
        ["git", "push", "origin", branch_name, "--no-verify"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def get_diff(repo_dir):
    output = subprocess.run(
        ["git", "diff"], cwd=repo_dir, check=True, text=True, capture_output=True
    )

    return output.stdout


def stage_all_files(repo_dir):
    subprocess.run(
        ["git", "add", "."], cwd=repo_dir, check=True, text=True, capture_output=True
    )


def fetch_all_refs(repo_dir):
    return subprocess.run(
        ["git", "fetch", "--all", "--tags", "--prune"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def ensure_commit_available(repo_dir, commit_hash):
    normalized_commit_hash = normalize_commit_hash(commit_hash)
    if not normalized_commit_hash:
        return

    with repo_admin_lock(repo_dir):
        try:
            _run_git_command(
                ["git", "cat-file", "-e", f"{normalized_commit_hash}^{{commit}}"],
                cwd=repo_dir,
            )
            return
        except subprocess.CalledProcessError:
            logger.info(
                "Fetching refs for '%s' because commit %s is not available locally",
                repo_dir,
                normalized_commit_hash,
            )
            fetch_result = fetch_all_refs(repo_dir)
            _log_completed_process(fetch_result)
            _run_git_command(
                ["git", "cat-file", "-e", f"{normalized_commit_hash}^{{commit}}"],
                cwd=repo_dir,
            )


def remove_repo_worktree(repo_dir: str, worktree_dir: str):
    if not worktree_dir or not os.path.exists(worktree_dir):
        return
    if not os.path.exists(os.path.join(worktree_dir, ".git")):
        shutil.rmtree(worktree_dir, ignore_errors=True)
        return
    with repo_admin_lock(repo_dir):
        result = _run_git_command(
            ["git", "worktree", "remove", "--force", worktree_dir],
            cwd=repo_dir,
        )
        _log_completed_process(result)


def checkout_commit(repo_dir, commit_hash):
    normalized_commit_hash = normalize_commit_hash(commit_hash)
    if not normalized_commit_hash:
        return

    try:
        subprocess.run(
            ["git", "reset", "--hard", normalized_commit_hash],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(
            "git reset --hard %s failed in '%s': %s",
            normalized_commit_hash,
            repo_dir,
            e,
        )
        if e.stdout:
            logger.error("git reset stdout: %s", e.stdout.strip())
        if e.stderr:
            logger.error("git reset stderr: %s", e.stderr.strip())

        try:
            logger.info(
                "Fetching refs for '%s' and retrying checkout of commit %s",
                repo_dir,
                normalized_commit_hash,
            )
            fetch_result = fetch_all_refs(repo_dir)
            if fetch_result.stdout:
                logger.info(fetch_result.stdout.strip())
            if fetch_result.stderr:
                logger.info(fetch_result.stderr.strip())

            subprocess.run(
                ["git", "reset", "--hard", normalized_commit_hash],
                cwd=repo_dir,
                check=True,
                text=True,
                capture_output=True,
            )
            return
        except subprocess.CalledProcessError as retry_error:
            logger.error(
                "Retry git reset --hard %s failed in '%s': %s",
                normalized_commit_hash,
                repo_dir,
                retry_error,
            )
            if retry_error.stdout:
                logger.error("retry stdout: %s", retry_error.stdout.strip())
            if retry_error.stderr:
                logger.error("retry stderr: %s", retry_error.stderr.strip())
            raise retry_error


def create_and_checkout_new_branch(repo_dir: str, branch_name: str):
    try:
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr)
        raise e


def setup_repo(repo_url, repo_dir, branch_name="master"):
    maybe_clone(repo_url, repo_dir)
    clean_and_reset_state(repo_dir)
    checkout_branch(repo_dir, branch_name)
    pull_latest(repo_dir)


def clean_and_reset_repo(repo_dir, branch_name="master"):
    clean_and_reset_state(repo_dir)
    checkout_branch(repo_dir, branch_name)
    pull_latest(repo_dir)
