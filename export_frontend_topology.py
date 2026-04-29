import argparse
import json
import os
from typing import Dict, List

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
                return {int(v): [int(k[0]), int(k[1])] for k, v in obj.items()}
        return {int(k): [int(v[0]), int(v[1])] for k, v in obj.items()}
    if isinstance(obj, np.ndarray) and obj.ndim == 2 and obj.shape[1] == 2:
        return {int(i): [int(obj[i, 0]), int(obj[i, 1])] for i in range(obj.shape[0])}
    raise ValueError(f"Unsupported dict_xy format: {path}")


def load_adjacency(path: str) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter=",")
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape((1, -1))
    return matrix


def neighbor_entries(matrix: np.ndarray, dict_xy: Dict[int, List[int]], region_id: int, prefix: str, limit: int):
    weights = matrix[region_id]
    order = np.argsort(weights)[::-1]
    items = []
    for idx in order:
        idx = int(idx)
        weight = float(weights[idx])
        if idx == region_id or weight <= 0:
            continue
        x, y = dict_xy.get(idx, [0, 0])
        items.append({
            "regionId": idx,
            "gridId": f"{prefix}-{idx:02d}",
            "x": int(x),
            "y": int(y),
            "weight": round(weight, 4),
        })
        if len(items) >= limit:
            break
    return items


def export_dataset(dataset: str, limit: int):
    prefix = "NYC" if dataset == "nyc" else "CHI"
    dict_xy = safe_load_dict_xy(os.path.join(dataset, "dict_xy.npy"))
    road = load_adjacency(os.path.join(dataset, "road_ad.txt"))
    poi = load_adjacency(os.path.join(dataset, "poi_ad.txt"))
    record = load_adjacency(os.path.join(dataset, "record_ad.txt"))

    payload = {}
    region_count = road.shape[0]
    for region_id in range(region_count):
        x, y = dict_xy.get(region_id, [0, 0])
        payload[str(region_id)] = {
            "regionId": region_id,
            "gridId": f"{prefix}-{region_id:02d}",
            "x": int(x),
            "y": int(y),
            "road": neighbor_entries(road, dict_xy, region_id, prefix, limit),
            "poi": neighbor_entries(poi, dict_xy, region_id, prefix, limit),
            "record": neighbor_entries(record, dict_xy, region_id, prefix, limit),
        }
    return payload


def main():
    parser = argparse.ArgumentParser(description="Export frontend topology summary from adjacency matrices")
    parser.add_argument("--output", default="results/frontend_topology.json")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    payload = {
        "nyc": export_dataset("nyc", args.limit),
        "chicago": export_dataset("chicago", args.limit),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Frontend-Topology-Saved-To: {args.output}")


if __name__ == "__main__":
    main()
