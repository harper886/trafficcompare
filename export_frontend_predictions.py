import argparse
import json
import os

import numpy as np


def safe_load_dict_xy(path: str):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    if isinstance(obj, dict):
        if len(obj) > 0:
            k0 = next(iter(obj.keys()))
            v0 = obj[k0]
            if isinstance(k0, (tuple, list, np.ndarray)) and not isinstance(v0, (tuple, list, np.ndarray)):
                try:
                    return {int(v): [int(k[0]), int(k[1])] for k, v in obj.items()}
                except Exception:
                    pass
        return {int(k): [int(v[0]), int(v[1])] for k, v in obj.items()}
    if isinstance(obj, np.ndarray) and obj.ndim == 2 and obj.shape[1] == 2:
        return {int(i): [int(obj[i, 0]), int(obj[i, 1])] for i in range(obj.shape[0])}
    raise ValueError(f"Unsupported dict_xy format: {path}")


def flatten_label(label):
    label = np.asarray(label)
    if label.ndim == 1:
        return label.astype(int)
    return label.reshape(-1).astype(int)


def build_frames(dataset, data_path, label_path, dict_xy_path, pred_path=None, smooth_path=None, threshold=0.5):
    all_data = np.load(data_path, allow_pickle=True)
    label = flatten_label(np.load(label_path, allow_pickle=True))
    dict_xy = safe_load_dict_xy(dict_xy_path)

    time_len = int(all_data.shape[0])
    region_count = int(all_data.shape[1])
    feature_dim = int(all_data.shape[2]) if all_data.ndim >= 3 else 0

    pred = None
    if pred_path and os.path.exists(pred_path):
        pred = np.load(pred_path, allow_pickle=True).reshape(-1)
    smooth = None
    if smooth_path and os.path.exists(smooth_path):
        smooth = np.load(smooth_path, allow_pickle=True).reshape(-1)

    window_count = time_len - 5
    expected_len = window_count * region_count
    if pred is not None and len(pred) != expected_len:
        raise ValueError(f"pred length mismatch: got {len(pred)}, expected {expected_len}")
    if smooth is not None and len(smooth) != expected_len:
        raise ValueError(f"smooth length mismatch: got {len(smooth)}, expected {expected_len}")
    if len(label) != expected_len:
        raise ValueError(f"label length mismatch: got {len(label)}, expected {expected_len}")

    frames = []
    for t in range(window_count):
        items = []
        for region_id in range(region_count):
            idx = t * region_count + region_id
            x, y = dict_xy.get(region_id, [0, 0])
            current_flow = float(all_data[t + 5, region_id, 0]) if feature_dim > 0 else 0.0
            history_slice = all_data[max(0, t):t + 5, region_id, 0] if feature_dim > 0 else np.array([0])
            avg_flow = float(np.mean(history_slice)) if len(history_slice) else 0.0
            raw_prob = float(pred[idx]) if pred is not None else None
            smooth_prob = float(smooth[idx]) if smooth is not None else raw_prob
            truth = int(label[idx])
            pred_positive = (smooth_prob is not None and smooth_prob >= threshold)
            truth_status = "tp" if truth == 1 and pred_positive else "fn" if truth == 1 and not pred_positive else "fp" if truth == 0 and pred_positive else None
            items.append({
                "regionId": int(region_id),
                "x": int(x),
                "y": int(y),
                "currentFlow": round(current_flow, 4),
                "avgFlow": round(avg_flow, 4),
                "label": truth,
                "rawProb": None if raw_prob is None else round(raw_prob, 6),
                "smoothProb": None if smooth_prob is None else round(smooth_prob, 6),
                "truthStatus": truth_status,
            })
        frames.append({"timeIndex": t, "items": items})

    return {
        "dataset": dataset,
        "lenRecentTime": 5,
        "regionCount": region_count,
        "timeLength": time_len,
        "windowCount": window_count,
        "threshold": threshold,
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(description="Export frontend-ready prediction JSON for dashboard")
    parser.add_argument("--dataset", required=True, choices=["nyc", "chicago"])
    parser.add_argument("--data")
    parser.add_argument("--label")
    parser.add_argument("--dict_xy")
    parser.add_argument("--pred", default="")
    parser.add_argument("--smooth", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = args.dataset
    data_path = args.data or os.path.join(base, "data_nyc.npy" if base == "nyc" else "data_chicago.npy")
    label_path = args.label or os.path.join(base, "label.npy")
    dict_xy_path = args.dict_xy or os.path.join(base, "dict_xy.npy")

    payload = build_frames(
        dataset=args.dataset,
        data_path=data_path,
        label_path=label_path,
        dict_xy_path=dict_xy_path,
        pred_path=args.pred or None,
        smooth_path=args.smooth or None,
        threshold=args.threshold,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
