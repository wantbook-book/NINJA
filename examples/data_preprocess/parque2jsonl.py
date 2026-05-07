import argparse
import json
import os
from typing import Iterable, Optional

import pyarrow as pa
import pyarrow.parquet as pq


def _iter_rows(
    parquet_path: str, columns: Optional[list[str]] = None, batch_size: int = 1024
) -> Iterable[dict]:
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        if isinstance(batch, pa.RecordBatch):
            rows = batch.to_pylist()
        else:
            rows = pa.Table.from_batches([batch]).to_pylist()
        for row in rows:
            yield row


def parquet_to_jsonl(
    input_path: str,
    output_path: str,
    columns: Optional[list[str]] = None,
    batch_size: int = 1024,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f_out:
        for row in _iter_rows(input_path, columns=columns, batch_size=batch_size):
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a parquet file to JSONL.")
    parser.add_argument("--input", required=True, help="Path to parquet file.")
    parser.add_argument("--output", default=None, help="Path to output JSONL file.")
    parser.add_argument(
        "--columns",
        default=None,
        help="Comma-separated list of columns to keep (optional).",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Record batch size.")
    args = parser.parse_args()

    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}.jsonl" if ext else f"{args.input}.jsonl"

    columns = None
    if args.columns:
        columns = [col.strip() for col in args.columns.split(",") if col.strip()]

    parquet_to_jsonl(args.input, output_path, columns=columns, batch_size=args.batch_size)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
