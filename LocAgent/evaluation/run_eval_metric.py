import argparse
import os

from eval_metric import eval_w_file, evaluate_results


def parse_csv(raw: str, cast=str):
    values = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    return values


def load_selected_list(path: str):
    path = str(path or "").strip()
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        values = [line.strip() for line in f if line.strip()]
    return values or None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LocAgent evaluation metrics.")
    parser.add_argument("--mode", choices=["dataset", "file"], default="dataset")
    parser.add_argument("--loc-file", required=True, help="Localization output jsonl file.")
    parser.add_argument(
        "--dataset",
        default="czlll/SWE-bench_Lite",
        help=(
            "Dataset source for dataset mode. Supports a HuggingFace dataset name, "
            "a local save_to_disk directory, or a local parquet/json/jsonl file."
        ),
    )
    parser.add_argument("--split", default="test", help="Dataset split name for dataset mode.")
    parser.add_argument("--gt-file", default="", help="Ground-truth jsonl file for file mode.")
    parser.add_argument(
        "--selected-instance-id-file",
        default="",
        help="Optional newline-delimited instance-id file used to filter evaluation rows.",
    )
    parser.add_argument("--output-csv", default="", help="Optional CSV output path.")
    parser.add_argument("--metrics", default="acc,ndcg,precision,recall,map")
    parser.add_argument("--file-k-values", default="1,3,5")
    parser.add_argument("--module-k-values", default="5,10")
    parser.add_argument("--function-k-values", default="5,10")
    parser.add_argument("--file-key", default="found_files")
    parser.add_argument("--module-key", default="found_modules")
    parser.add_argument("--function-key", default="found_entities")
    return parser


def main():
    args = build_arg_parser().parse_args()
    selected_list = load_selected_list(args.selected_instance_id_file)
    level2key_dict = {
        "file": args.file_key.strip(),
        "module": args.module_key.strip(),
        "function": args.function_key.strip(),
    }
    k_values_list = [
        parse_csv(args.file_k_values, int),
        parse_csv(args.module_k_values, int),
        parse_csv(args.function_k_values, int),
    ]
    metrics = parse_csv(args.metrics, str)

    if args.mode == "dataset":
        df = evaluate_results(
            loc_file=args.loc_file.strip(),
            level2key_dict=level2key_dict,
            dataset=args.dataset.strip(),
            split=args.split.strip(),
            selected_list=selected_list,
            metrics=metrics,
            k_values_list=k_values_list,
        )
    else:
        gt_file = args.gt_file.strip()
        if not gt_file:
            raise ValueError("--gt-file must be set when --mode=file.")
        df = eval_w_file(
            gt_file=gt_file,
            loc_file=args.loc_file.strip(),
            level2key_dict=level2key_dict,
            selected_list=selected_list,
            k_values_list=k_values_list,
        )

    print(df.to_string(index=False))

    output_csv = args.output_csv.strip()
    if output_csv:
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved evaluation table to: {output_csv}")


if __name__ == "__main__":
    main()
