import argparse
import os
import subprocess
import sys

import numpy as np
import tensorflow as tf

from configs.params import nyc_params, chicago_params
from lib.utils import get_neigh_index, prepare_data, streaming_postprocess
from model import MYPLAN


def load_params(dataset: str):
    return nyc_params if dataset == "nyc" else chicago_params


def load_threshold_from_metrics(dataset: str, metrics_file: str):
    if not metrics_file or not os.path.exists(metrics_file):
        return 0.31
    import json
    best = None
    best_ts = -1
    with open(metrics_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if str(row.get("dataset")) != dataset:
                continue
            if str(row.get("model", "")).lower() != "myplan":
                continue
            ts = float(row.get("timestamp", 0))
            if ts >= best_ts:
                best_ts = ts
                best = row
    if not best:
        return 0.31
    return float(best.get("threshold_selected") or best.get("threshold_stream_on") or best.get("threshold_f1") or 0.31)


def run_inference(dataset: str, weights: str, max_neigh: int, use_streaming: bool, threshold_on: float):
    params = load_params(dataset)
    all_data_path = f"{dataset}/{params.all_data}"
    threshold_path = f"{dataset}/{params.threshold_nc}"

    all_data = np.load(all_data_path)
    threshold_nc = np.load(threshold_path)
    all_data = np.asarray(prepare_data(all_data, params.len_recent_time), dtype=np.float32)
    threshold_nc = np.asarray(prepare_data(threshold_nc, params.len_recent_time), dtype=np.float32)

    neigh_road_index = get_neigh_index(f"{dataset}/road_ad.txt", max_neigh=max_neigh)
    neigh_record_index = get_neigh_index(f"{dataset}/record_ad.txt", max_neigh=max_neigh)
    neigh_poi_index = get_neigh_index(f"{dataset}/poi_ad.txt", max_neigh=max_neigh)

    model = MYPLAN(
        params.dr,
        params.len_recent_time,
        params.number_sp,
        params.number_region,
        neigh_poi_index,
        neigh_road_index,
        neigh_record_index,
        attention_mode="scaled_dot",
        evolution_smooth=False,
    )

    y_dynamic = tf.ones((params.len_recent_time, params.number_region, 2 * params.dr), dtype=tf.float32)
    _ = model(tf.convert_to_tensor(all_data[:1]), tf.convert_to_tensor(threshold_nc[:1]), y_dynamic)
    model.load_weights(weights)

    num_windows = int(all_data.shape[0])
    pred = np.full((num_windows, int(params.number_region)), np.nan, dtype=np.float32)
    for i in range(num_windows):
        x = tf.convert_to_tensor(all_data[i:i + 1])
        th_nc = tf.convert_to_tensor(threshold_nc[i:i + 1])
        y_pred, y_dynamic_now, _ = model(x, th_nc, y_dynamic)
        pred[i] = y_pred.numpy().reshape((-1,))
        y_dynamic = y_dynamic_now

    pred_flat = pred.reshape((-1, 1))
    if use_streaming:
        smooth_flat, _ = streaming_postprocess(pred_flat, alpha=0.0, th_on=threshold_on, th_off=max(0.0, threshold_on - 0.01))
    else:
        smooth_flat = pred_flat.copy()
    return pred_flat.reshape(-1), smooth_flat.reshape(-1)


def main():
    parser = argparse.ArgumentParser(description="Run MYPLAN inference from weights and export frontend JSON")
    parser.add_argument("--dataset", required=True, choices=["nyc", "chicago"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--max_neigh", type=int, default=4)
    parser.add_argument("--metrics_file", default="results/metrics.jsonl")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--no_streaming", action="store_true")
    parser.add_argument("--pred_out", default="")
    parser.add_argument("--smooth_out", default="")
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    threshold_on = float(args.threshold) if args.threshold is not None else load_threshold_from_metrics(args.dataset, args.metrics_file)
    pred, smooth = run_inference(
        dataset=args.dataset,
        weights=args.weights,
        max_neigh=int(args.max_neigh),
        use_streaming=not args.no_streaming,
        threshold_on=threshold_on,
    )

    pred_out = args.pred_out or f"results/{args.dataset}_pred.npy"
    smooth_out = args.smooth_out or f"results/{args.dataset}_smooth.npy"
    json_out = args.json_out or f"results/frontend_predictions_{args.dataset}.json"

    os.makedirs(os.path.dirname(pred_out) or ".", exist_ok=True)
    np.save(pred_out, pred)
    np.save(smooth_out, smooth)

    cmd = [
        sys.executable,
        "export_frontend_predictions.py",
        "--dataset", args.dataset,
        "--pred", pred_out,
        "--smooth", smooth_out,
        "--threshold", str(threshold_on),
        "--output", json_out,
    ]
    subprocess.check_call(cmd)

    print(f"Pred-Saved-To: {pred_out}")
    print(f"Smooth-Saved-To: {smooth_out}")
    print(f"Frontend-Json-Saved-To: {json_out}")


if __name__ == "__main__":
    main()
