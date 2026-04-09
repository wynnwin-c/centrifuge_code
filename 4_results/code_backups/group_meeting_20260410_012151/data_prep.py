# 1_data/data_prep.py
import os
import re
import json
import numpy as np
import pandas as pd
import yaml

with open("1_data/config_dataset.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

PROC_PATH = cfg["processed_path"]
SAMPLING_RATE = cfg["sampling_rate"]
SLICE_TIME = cfg["slice_time"]
WINDOW = int(SAMPLING_RATE * cfg["window"])
EPOCHS_PER_FILE = int(SLICE_TIME * SAMPLING_RATE)
ALL_SPEEDS = [560, 840, 1120, 1400, 2170, 5320, 7500, 9710]
SPEED_MIN = min(ALL_SPEEDS)
SPEED_MAX = max(ALL_SPEEDS)
TARGET_N_KEYS = ("水水30g", "水水 ")
TARGET_FAULT_KEY = "水蔗"
SOURCE_NORMAL_MULTIPLIER = 3
SOURCE_STEP = 7168     # 50 windows/file
TARGET_N_STEP = 3584       # 100 windows/file from 水水N and 水水30g => 200 total
TARGET_FAULT_STEP = 3584   # 100 windows/file from e1 and e2 => 200 total

os.makedirs(PROC_PATH, exist_ok=True)


def slice_signal(sig, window, step):
    segments = []
    for i in range(0, len(sig) - window, step):
        segments.append(sig[i:i + window])
    return np.array(segments)


def infer_label(dataset_name: str, filename: str) -> int:
    upper_name = filename.upper()
    lower_name = filename.lower()
    if "EF" not in upper_name and "e1" not in lower_name and "e2" not in lower_name and "N" in upper_name:
        return 0
    if dataset_name == "experiment":
        if "EF1" in upper_name:
            return 1
        if "EF2" in upper_name:
            return 2
        if "EF3" in upper_name:
            return 3
        raise ValueError(f"无法从源域文件名识别标签: {filename}")
    if dataset_name.startswith("centrifuge_"):
        if "e1" in lower_name or "e2" in lower_name:
            return 1
        raise ValueError(f"无法从目标域文件名识别标签: {filename}")
    raise ValueError(f"未知数据集: {dataset_name}")


def infer_speed(dataset_name: str, filename: str) -> int:
    if dataset_name == "experiment":
        match = re.search(r"[（(]([^）)]+)[）)]", filename)
        if not match:
            raise ValueError(f"无法从源域文件名识别转速: {filename}")
        inside = match.group(1)
        rpm_match = re.search(r"(560|840|1120|1400)", inside)
        if not rpm_match:
            raise ValueError(f"无法从源域文件名识别转速: {filename}")
        return int(rpm_match.group(1))
    if dataset_name.startswith("centrifuge_"):
        match = re.search(r"centrifuge_(\d+)", dataset_name)
        if not match:
            raise ValueError(f"无法从目标域名称识别转速: {dataset_name}")
        return int(match.group(1))
    raise ValueError(f"未知数据集: {dataset_name}")


def normalize_speed(speed_rpm: int) -> float:
    return float((speed_rpm - SPEED_MIN) / (SPEED_MAX - SPEED_MIN))


def keep_file(dataset_name: str, filename: str, label: int) -> bool:
    if dataset_name == "experiment":
        return True
    lower_name = filename.lower()
    if dataset_name.startswith("centrifuge_"):
        if label == 0:
            return any(key in filename for key in TARGET_N_KEYS) and " n" in lower_name
        if label == 1:
            return TARGET_FAULT_KEY in filename and ("e1" in lower_name or "e2" in lower_name)
        return False
    return True


def step_for_sample(dataset_name: str, label: int) -> int:
    if dataset_name == "experiment":
        return SOURCE_STEP
    if dataset_name.startswith("centrifuge_"):
        if label == 0:
            return TARGET_N_STEP
        if label == 1:
            return TARGET_FAULT_STEP
    return int(WINDOW * (1 - float(cfg.get("overlap", 0.5))))


def process_single_folder(folder_path, save_name):
    print(f"Processing folder: {folder_path}")
    all_segments = []
    all_labels = []
    all_speeds = []
    all_groups = []
    group_map = {}
    next_gid = 0
    kept_files = []
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            if not f.endswith(".csv"):
                continue
            label = infer_label(save_name, f)
            if not keep_file(save_name, f, label):
                continue
            step = step_for_sample(save_name, label)
            speed = normalize_speed(infer_speed(save_name, f))
            file_key = os.path.relpath(os.path.join(root, f), folder_path)
            if file_key not in group_map:
                group_map[file_key] = next_gid
                next_gid += 1
            gid = group_map[file_key]
            df = pd.read_csv(os.path.join(root, f), encoding="iso-8859-1")
            sig = df.iloc[:, 1].values.astype(np.float32)
            sig = sig[:EPOCHS_PER_FILE]
            segments = slice_signal(sig, WINDOW, step)
            if len(segments) == 0:
                continue
            if save_name == "experiment" and label == 0 and SOURCE_NORMAL_MULTIPLIER > 1:
                segments = np.concatenate([segments] * SOURCE_NORMAL_MULTIPLIER, axis=0)
            all_segments.append(segments)
            all_labels.extend([label] * len(segments))
            all_speeds.extend([speed] * len(segments))
            all_groups.extend([gid] * len(segments))
            kept_files.append(file_key)
    if not all_segments:
        raise ValueError(f"目录中没有可用样本: {folder_path}")
    X = np.vstack(all_segments).astype(np.float32)
    y = np.array(all_labels, dtype=np.int64)
    s = np.array(all_speeds, dtype=np.float32)
    g = np.array(all_groups, dtype=np.int64)
    np.save(os.path.join(PROC_PATH, f"{save_name}_X.npy"), X)
    np.save(os.path.join(PROC_PATH, f"{save_name}_y.npy"), y)
    np.save(os.path.join(PROC_PATH, f"{save_name}_s.npy"), s)
    np.save(os.path.join(PROC_PATH, f"{save_name}_g.npy"), g)
    with open(os.path.join(PROC_PATH, f"{save_name}_groups.json"), "w", encoding="utf-8") as f:
        json.dump({str(v): k for k, v in group_map.items()}, f, ensure_ascii=False, indent=2)
    unique, counts = np.unique(y, return_counts=True)
    print(f"[OK] {save_name}: {X.shape[0]} samples saved. labels={dict(zip(unique.tolist(), counts.tolist()))} groups={len(group_map)} kept_files={kept_files} speed_range=({s.min():.4f}, {s.max():.4f})")


if __name__ == "__main__":
    for name, info in cfg["datasets"].items():
        process_single_folder(info["path"], name)
    print("数据预处理完成，已保存到 1_data/processed/")
