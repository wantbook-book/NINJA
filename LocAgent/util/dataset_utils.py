import os
from pathlib import Path

from datasets import load_dataset, load_from_disk

_LOCAL_FILE_LOADERS = {
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".parquet": "parquet",
}


def _load_local_files(file_paths: list[str], split: str):
    if not file_paths:
        raise ValueError("file_paths must not be empty")

    suffixes = {Path(file_path).suffix.lower() for file_path in file_paths}
    if len(suffixes) != 1:
        raise ValueError(
            "All local dataset files must share the same extension. "
            f"Got: {sorted(suffixes)}"
        )

    suffix = next(iter(suffixes))
    loader_name = _LOCAL_FILE_LOADERS.get(suffix)
    if loader_name is None:
        raise ValueError(
            f"Unsupported local dataset file extension: {suffix}. "
            f"Supported extensions: {sorted(_LOCAL_FILE_LOADERS)}"
        )

    return load_dataset(
        loader_name,
        data_files={split: sorted(file_paths)},
        split=split,
    )


def _matches_split_file(path: Path, split: str) -> bool:
    suffix = path.suffix.lower()
    if suffix not in _LOCAL_FILE_LOADERS:
        return False

    stem = path.stem.lower()
    split = split.lower()
    if stem == split or stem.startswith(f"{split}-") or stem.startswith(f"{split}_"):
        return True

    parent_parts = {part.lower() for part in path.parts[:-1]}
    return split in parent_parts


def _find_local_split_files(dataset_dir: str, split: str) -> list[str]:
    root = Path(dataset_dir)
    matches = [
        str(path)
        for path in root.rglob("*")
        if path.is_file() and _matches_split_file(path, split)
    ]
    return sorted(matches)


def dataset_spec_to_name(dataset_spec: str) -> str:
    dataset_spec = str(dataset_spec or "").strip()
    if not dataset_spec:
        return "dataset"

    if os.path.isdir(dataset_spec):
        return os.path.basename(os.path.normpath(dataset_spec))

    if os.path.isfile(dataset_spec):
        return Path(dataset_spec).stem

    return dataset_spec.split("/")[-1]


def load_benchmark_dataset(dataset_spec: str, split: str):
    dataset_spec = str(dataset_spec or "").strip()
    if not dataset_spec:
        raise ValueError("dataset must not be empty")

    if os.path.isfile(dataset_spec):
        return _load_local_files([dataset_spec], split)

    if os.path.isdir(dataset_spec):
        try:
            loaded = load_from_disk(dataset_spec)
        except Exception:
            try:
                return load_dataset(dataset_spec, split=split)
            except Exception:
                split_files = _find_local_split_files(dataset_spec, split)
                if split_files:
                    return _load_local_files(split_files, split)
                raise

        if hasattr(loaded, "keys"):
            if split not in loaded:
                raise ValueError(
                    f"Split '{split}' not found in local dataset directory: {dataset_spec}. "
                    f"Available splits: {list(loaded.keys())}"
                )
            return loaded[split]
        return loaded

    return load_dataset(dataset_spec, split=split)
