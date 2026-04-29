import argparse
import json
import os
from typing import Dict, List


MODEL_ALIASES = {
    "Myplan（不加自适应和后处理）": "noSmoothNoStream",
    "Myplan（不加自适应）": "noSmooth",
    "Myplan（不加后处理）": "noStream",
    "Myplan": "full",
}

METRIC_ALIASES = {
    "AUC-PR": "aucpr",
    "AUC-ROC": "aucroc",
    "F1 socre": "f1",
    "Accuracy": "acc",
    "Recall": "recall",
}

DATASET_ALIASES = {
    "NYC": "nyc",
    "Chicago": "chicago",
}


def parse_table_row(line: str) -> List[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_metrics_markdown(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    sections = {
        "models": {"header": None, "rows": []},
        "ablation": {"header": None, "rows": []},
    }

    current = None
    for line in lines:
        if not line.startswith("|"):
            continue
        row = parse_table_row(line)
        if not row or row[0].startswith(":"):
            continue
        if row[0] == "方法" and "GSNet" in row:
            current = "models"
            sections[current]["header"] = row
            continue
        if row[0] == "方法" and "Myplan（不加自适应和后处理）" in row:
            current = "ablation"
            sections[current]["header"] = row
            continue
        if current and row[0] != "**数据集**":
            sections[current]["rows"].append(row)

    payload = {dataset: {"models": {}, "ablation": {}} for dataset in DATASET_ALIASES.values()}

    model_header = sections["models"]["header"]
    if model_header:
        model_names = model_header[2:]
        current_dataset = None
        for row in sections["models"]["rows"]:
            dataset_cell = row[0].replace("*", "").strip()
            if dataset_cell:
                current_dataset = DATASET_ALIASES.get(dataset_cell)
            if not current_dataset:
                continue
            metric_key = METRIC_ALIASES.get(row[1])
            if not metric_key:
                continue
            for model_name, value in zip(model_names, row[2:]):
                payload[current_dataset]["models"].setdefault(model_name, {})[metric_key] = float(value)

    ablation_header = sections["ablation"]["header"]
    if ablation_header:
        ablation_names = ablation_header[2:]
        current_dataset = None
        for row in sections["ablation"]["rows"]:
            dataset_cell = row[0].replace("*", "").strip()
            if dataset_cell:
                current_dataset = DATASET_ALIASES.get(dataset_cell)
            if not current_dataset:
                continue
            metric_key = METRIC_ALIASES.get(row[1])
            if not metric_key:
                continue
            for model_name, value in zip(ablation_names, row[2:]):
                alias = MODEL_ALIASES.get(model_name)
                if not alias:
                    continue
                payload[current_dataset]["ablation"].setdefault(alias, {})[metric_key] = float(value)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frontend-ready metrics JSON from markdown tables")
    parser.add_argument("--input", default="p1结果展示.md")
    parser.add_argument("--output", default="results/frontend_metrics.json")
    args = parser.parse_args()

    payload = parse_metrics_markdown(args.input)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Frontend-Metrics-Saved-To: {args.output}")


if __name__ == "__main__":
    main()
