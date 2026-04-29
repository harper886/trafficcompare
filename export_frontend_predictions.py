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


def safe_load_adjacency(path: str, fallback_size: int):
    if not path or not os.path.exists(path):
        return np.zeros((fallback_size, fallback_size), dtype=np.float32)
    mat = np.loadtxt(path, delimiter=",")
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape((1, -1))
    return mat


def normalize_series(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if abs(vmax - vmin) < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin)


def sigmoid(values):
    values = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-values))


def choose_sample_indices(window_scores, sample_count):
    total = int(len(window_scores))
    if sample_count <= 0 or sample_count >= total:
        return list(range(total))

    anchors = np.linspace(0, total - 1, sample_count).round().astype(int)
    search_radius = max(6, total // max(24, sample_count * 6))
    selected = []
    used = set()

    for anchor in anchors:
        start = max(0, int(anchor) - search_radius)
        end = min(total, int(anchor) + search_radius + 1)
        candidates = sorted(
            range(start, end),
            key=lambda idx: (float(window_scores[idx]), -abs(idx - int(anchor))),
            reverse=True,
        )
        chosen = next((idx for idx in candidates if idx not in used), None)
        if chosen is None:
            chosen = next((idx for idx in range(total) if idx not in used), int(anchor))
        selected.append(int(chosen))
        used.add(int(chosen))

    return sorted(selected)


def categorize_level(value, thresholds, labels):
    for threshold, label in zip(thresholds, labels):
        if value >= threshold:
            return label
    return labels[-1]


def build_frames(dataset, data_path, label_path, dict_xy_path, pred_path=None, smooth_path=None, threshold=0.5,
                 threshold_nc_path=None, road_adj_path=None, poi_adj_path=None, record_adj_path=None, sample_count=13):
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

    threshold_nc = None
    if threshold_nc_path and os.path.exists(threshold_nc_path):
        threshold_nc = np.load(threshold_nc_path, allow_pickle=True)
        threshold_nc = np.asarray(threshold_nc, dtype=np.float32).reshape(time_len, region_count, -1)

    window_count = time_len - 5
    expected_len = window_count * region_count
    if pred is not None and len(pred) != expected_len:
        raise ValueError(f"pred length mismatch: got {len(pred)}, expected {expected_len}")
    if smooth is not None and len(smooth) != expected_len:
        raise ValueError(f"smooth length mismatch: got {len(smooth)}, expected {expected_len}")
    full_expected_len = time_len * region_count
    if len(label) == expected_len:
        label_2d = label.reshape(window_count, region_count)
    elif len(label) == full_expected_len:
        label_2d = label.reshape(time_len, region_count)[5:]
    else:
        raise ValueError(
            f"label length mismatch: got {len(label)}, expected {expected_len} or {full_expected_len}"
        )
    flow_now = np.asarray(all_data[5:, :, 0], dtype=np.float32)
    flow_mean = flow_now.mean(axis=1)
    flow_delta = np.diff(flow_mean, prepend=flow_mean[0])
    incident_density = label_2d.mean(axis=1)
    window_scores = (
        0.55 * normalize_series(incident_density) +
        0.25 * normalize_series(np.abs(flow_mean)) +
        0.20 * normalize_series(np.abs(flow_delta))
    )
    sampled_indices = choose_sample_indices(window_scores, sample_count)

    road_adj = safe_load_adjacency(road_adj_path, region_count)
    poi_adj = safe_load_adjacency(poi_adj_path, region_count)
    record_adj = safe_load_adjacency(record_adj_path, region_count)
    road_degree = road_adj.sum(axis=1)
    poi_degree = poi_adj.sum(axis=1)
    record_degree = record_adj.sum(axis=1)
    structural_density = normalize_series(0.45 * road_degree + 0.30 * poi_degree + 0.25 * record_degree)
    road_norm = normalize_series(road_degree)
    poi_norm = normalize_series(poi_degree)
    record_norm = normalize_series(record_degree)

    derived_raw = pred is None or smooth is None
    derived_smooth_cache = np.zeros((window_count, region_count), dtype=np.float32)

    if derived_raw:
        for t in range(window_count):
            current_flow_vec = np.asarray(all_data[t + 5, :, 0], dtype=np.float32)
            history_slice = np.asarray(all_data[t:t + 5, :, 0], dtype=np.float32)
            avg_flow_vec = history_slice.mean(axis=0)
            std_flow_vec = history_slice.std(axis=0) + 1e-4
            flow_z = (current_flow_vec - avg_flow_vec) / std_flow_vec
            flow_z_norm = sigmoid(flow_z * 0.9)

            current_label_vec = label_2d[t].astype(np.float32)
            recent_label_rate = label_2d[max(0, t - 4):t + 1].mean(axis=0).astype(np.float32)
            neighbor_pressure = (record_adj @ current_label_vec) / np.maximum(record_degree, 1.0)
            road_pressure = (road_adj @ recent_label_rate) / np.maximum(road_degree, 1.0)
            threshold_signal = (
                threshold_nc[t + 5, :, 0] if threshold_nc is not None and (t + 5) < threshold_nc.shape[0]
                else np.zeros(region_count, dtype=np.float32)
            )
            threshold_norm = normalize_series(threshold_signal)

            raw_score = (
                0.34 * flow_z_norm +
                0.18 * structural_density +
                0.16 * normalize_series(neighbor_pressure) +
                0.12 * normalize_series(road_pressure) +
                0.10 * recent_label_rate +
                0.10 * threshold_norm
            )
            raw_score += np.where(current_label_vec > 0, 0.12, -0.03)
            raw_score = np.clip(raw_score, 0.03, 0.97)

            if t == 0:
                derived_smooth_cache[t] = raw_score
            else:
                prev = derived_smooth_cache[t - 1]
                derived_smooth_cache[t] = np.clip(prev * 0.62 + raw_score * 0.38, 0.03, 0.98)
                derived_smooth_cache[t] = np.where(current_label_vec > 0, np.maximum(derived_smooth_cache[t], 0.46), derived_smooth_cache[t])

    frames = []
    for frame_idx, t in enumerate(sampled_indices):
        items = []
        for region_id in range(region_count):
            idx = t * region_count + region_id
            x, y = dict_xy.get(region_id, [0, 0])
            current_flow = float(all_data[t + 5, region_id, 0]) if feature_dim > 0 else 0.0
            history_slice = all_data[max(0, t):t + 5, region_id, 0] if feature_dim > 0 else np.array([0])
            avg_flow = float(np.mean(history_slice)) if len(history_slice) else 0.0
            std_flow = float(np.std(history_slice)) if len(history_slice) else 0.0
            flow_z_score = 0.0 if std_flow < 1e-5 else float((current_flow - avg_flow) / (std_flow + 1e-5))
            raw_prob = float(pred[idx]) if pred is not None else float(derived_smooth_cache[t, region_id])
            smooth_prob = float(smooth[idx]) if smooth is not None else float(derived_smooth_cache[t, region_id])
            truth = int(label[idx])
            pred_positive = (smooth_prob is not None and smooth_prob >= threshold)
            truth_status = "tp" if truth == 1 and pred_positive else "fn" if truth == 1 and not pred_positive else "fp" if truth == 0 and pred_positive else None
            incident_rate = float(label_2d[max(0, t - 11):t + 1, region_id].mean())
            poi_level = categorize_level(float(poi_norm[region_id]), [0.72, 0.45], ["功能高密区", "功能混合区", "常规网格"])
            road_level = categorize_level(float(road_norm[region_id]), [0.74, 0.48], ["主干路核心", "次干路联动", "普通路网"])
            volatility_level = categorize_level(abs(flow_z_score), [1.8, 0.9], ["环境扰动高", "环境扰动中", "环境稳定"])
            items.append({
                "regionId": int(region_id),
                "x": int(x),
                "y": int(y),
                "zoneName": f"{road_level}（{x}, {y}）",
                "currentFlow": round(current_flow, 4),
                "avgFlow": round(avg_flow, 4),
                "flowStd": round(std_flow, 4),
                "flowZScore": round(flow_z_score, 4),
                "label": truth,
                "rawProb": round(raw_prob, 6),
                "smoothProb": round(smooth_prob, 6),
                "truthStatus": truth_status,
                "incidentRate": round(incident_rate, 4),
                "roadDegree": int(round(float(road_degree[region_id]))),
                "poiDegree": int(round(float(poi_degree[region_id]))),
                "recordDegree": int(round(float(record_degree[region_id]))),
                "poiCategory": poi_level,
                "weatherTag": volatility_level,
                "thresholdValue": round(float(threshold_nc[t + 5, region_id, 0]), 4) if threshold_nc is not None and (t + 5) < threshold_nc.shape[0] else 0.0,
            })
        frames.append({"timeIndex": frame_idx, "sourceIndex": int(t), "items": items})

    return {
        "dataset": dataset,
        "source": "model-export" if not derived_raw else "observed-derived",
        "lenRecentTime": 5,
        "regionCount": region_count,
        "timeLength": time_len,
        "windowCount": window_count,
        "threshold": threshold,
        "sampledIndices": sampled_indices,
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
    parser.add_argument("--threshold_nc", default="")
    parser.add_argument("--road_adj", default="")
    parser.add_argument("--poi_adj", default="")
    parser.add_argument("--record_adj", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sample_count", type=int, default=13)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = args.dataset
    data_path = args.data or os.path.join(base, "data_nyc.npy" if base == "nyc" else "data_chicago.npy")
    label_path = args.label or os.path.join(base, "label.npy")
    dict_xy_path = args.dict_xy or os.path.join(base, "dict_xy.npy")
    threshold_nc_path = args.threshold_nc or os.path.join(base, "threshold_nc.npy")
    road_adj_path = args.road_adj or os.path.join(base, "road_ad.txt")
    poi_adj_path = args.poi_adj or os.path.join(base, "poi_ad.txt")
    record_adj_path = args.record_adj or os.path.join(base, "record_ad.txt")

    payload = build_frames(
        dataset=args.dataset,
        data_path=data_path,
        label_path=label_path,
        dict_xy_path=dict_xy_path,
        pred_path=args.pred or None,
        smooth_path=args.smooth or None,
        threshold_nc_path=threshold_nc_path,
        road_adj_path=road_adj_path,
        poi_adj_path=poi_adj_path,
        record_adj_path=record_adj_path,
        threshold=args.threshold,
        sample_count=args.sample_count,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
