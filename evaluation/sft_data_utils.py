from __future__ import annotations

import json
import os
import random
from typing import Any

import pandas as pd

from evaluation.utils import load_data


CHAT_MESSAGE_KEYS = ("role", "content", "tool_calls", "name", "tool_call_id")


def load_records(path: str) -> list[dict[str, Any]]:
    records = load_data(path)
    return [record for record in records if isinstance(record, dict)]


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def coerce_messages(value: Any) -> list[dict[str, Any]]:
    value = parse_jsonish(value)
    if not isinstance(value, list):
        return []
    return [message for message in value if isinstance(message, dict)]


def sanitize_messages(
    messages: list[dict[str, Any]],
    *,
    keep_message_metadata: bool = False,
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "") or "").strip()
        if not role:
            continue
        content = message.get("content", "")
        if content is None:
            content = ""

        if keep_message_metadata:
            clean_message = dict(message)
            clean_message["role"] = role
            clean_message["content"] = content
        else:
            clean_message = {
                key: message[key]
                for key in CHAT_MESSAGE_KEYS
                if key in message and message[key] is not None
            }
            clean_message["role"] = role
            clean_message["content"] = content
        sanitized.append(clean_message)
    return sanitized


def has_assistant_turn(messages: list[dict[str, Any]]) -> bool:
    return any(
        message.get("role") == "assistant" and str(message.get("content", "") or "").strip()
        for message in messages
    )


def split_records(
    records: list[dict[str, Any]],
    *,
    val_size: int | None,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if val_size is not None and val_size < 0:
        raise ValueError("--val-size must be non-negative")
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("--val-ratio must be in [0, 1]")
    if val_size is None:
        val_size = int(len(records) * val_ratio)
    if val_size > len(records):
        raise ValueError(f"--val-size ({val_size}) cannot exceed selected records ({len(records)})")

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    val_indices = set(indices[:val_size])

    train_data: list[dict[str, Any]] = []
    val_data: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        item = dict(record)
        item["extra_info"] = dict(record.get("extra_info", {}))
        item["extra_info"]["index"] = index
        if index in val_indices:
            item["extra_info"]["split"] = "test"
            val_data.append(item)
        else:
            item["extra_info"]["split"] = "train"
            train_data.append(item)
    return train_data, val_data


def _dataframe(records: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if records:
        return pd.DataFrame(records)
    return pd.DataFrame(columns=columns)


def write_dataset_outputs(
    records: list[dict[str, Any]],
    *,
    output_dir: str,
    val_size: int | None,
    val_ratio: float,
    seed: int,
    metadata: dict[str, Any],
) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    train_data, val_data = split_records(records, val_size=val_size, val_ratio=val_ratio, seed=seed)

    columns = (
        list(records[0].keys())
        if records
        else ["messages", "data_source", "ability", "reward_model", "extra_info"]
    )
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    _dataframe(train_data, columns).to_parquet(train_path, index=False)
    _dataframe(val_data, columns).to_parquet(val_path, index=False)

    if train_data:
        with open(os.path.join(output_dir, "train_sample.json"), "w", encoding="utf-8") as f:
            json.dump(train_data[0], f, ensure_ascii=False, indent=2)
    if val_data:
        with open(os.path.join(output_dir, "val_sample.json"), "w", encoding="utf-8") as f:
            json.dump(val_data[0], f, ensure_ascii=False, indent=2)

    metadata_payload = dict(metadata)
    metadata_payload.update(
        {
            "train_records": len(train_data),
            "val_records": len(val_data),
            "train_path": os.path.abspath(train_path),
            "val_path": os.path.abspath(val_path),
        }
    )
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    return train_path, val_path
