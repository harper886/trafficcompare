import argparse
import json
import os
from datetime import datetime, timezone


def load_latest_runtime_rows(metrics_file: str):
    latest = {}
    if not metrics_file or not os.path.exists(metrics_file):
        return latest

    with open(metrics_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            dataset = str(row.get("dataset") or "")
            model = str(row.get("model") or "").lower()
            if dataset not in {"nyc", "chicago"} or model != "myplan":
                continue
            ts = float(row.get("timestamp") or 0.0)
            current = latest.get(dataset)
            if current is None or ts >= float(current.get("timestamp") or 0.0):
                latest[dataset] = row
    return latest


def build_payload(rows):
    payload = {}
    for dataset, row in rows.items():
        ts = float(row.get("timestamp") or 0.0)
        payload[dataset] = {
            "dataset": dataset,
            "model": row.get("model", "myplan"),
            "timestamp": ts,
            "updatedAt": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts > 0 else None,
            "thresholdSelected": row.get("threshold_selected"),
            "thresholdF1": row.get("threshold_f1"),
            "thresholdAccu": row.get("threshold_accu"),
            "thresholdStreamOn": row.get("threshold_stream_on"),
            "thresholdStreamOff": row.get("threshold_stream_off"),
            "streamingAlpha": row.get("streaming_alpha"),
            "streamingEnabled": row.get("streaming_enabled"),
            "evolutionSmooth": row.get("evolution_smooth"),
            "attentionMode": row.get("attention_mode"),
            "maxNeigh": row.get("max_neigh"),
            "metrics": {
                "ap": row.get("ap"),
                "auc": row.get("auc"),
                "f1": row.get("f1"),
                "recall": row.get("recall"),
                "precision": row.get("precision"),
                "accuracy": row.get("accuracy"),
            },
        }
    return payload


def main():
    parser = argparse.ArgumentParser(description="Export frontend runtime meta from metrics.jsonl")
    parser.add_argument("--input", default="results/metrics.jsonl")
    parser.add_argument("--output", default="results/frontend_runtime_meta.json")
    args = parser.parse_args()

    payload = build_payload(load_latest_runtime_rows(args.input))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Frontend-Runtime-Meta-Saved-To: {args.output}")


if __name__ == "__main__":
    main()
