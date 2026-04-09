# 1_data/data_prep.py
import os
import re
import numpy as np
import pandas as pd
import yaml

with open("1_data/config_dataset.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

PROC_PATH = cfg["processed_path"]
SAMPLING_RATE = cfg["sampling_rate"]
SLICE_TIME = cfg["slice_time"]
OVERLAP = cfg["overlap"]
WINDOW = int(SAMPLING_RATE * cfg["window"])
EPOCHS_PER_FILE = int(SLICE_TIME * SAMPLING_RATE)
ALL_SPEEDS = [560, 840, 1120, 1400, 2170, 5320, 7500, 9710]
SPEED_MIN = min(ALL_SPEEDS)
SPEED_MAX = max(ALL_SPEEDS)

os.makedirs(PROC_PATH, exist_ok=True)


def slice_signal(sig, window, overlap):
    step = int(window * (1 - overlap))
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


def process_single_folder(folder_path, save_name):
    print(f"Processing folder: {folder_path}")
    all_segments = []
    all_labels = []
    all_speeds = []
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            if not f.endswith(".csv"):
                continue
            label = infer_label(save_name, f)
            speed = normalize_speed(infer_speed(save_name, f))
            df = pd.read_csv(os.path.join(root, f), encoding="iso-8859-1")
            sig = df.iloc[:, 1].values
            sig = sig[:EPOCHS_PER_FILE]
            segments = slice_signal(sig, WINDOW, OVERLAP)
            if len(segments) == 0:
                continue
            all_segments.append(segments)
            all_labels.extend([label] * len(segments))
            all_speeds.extend([speed] * len(segments))

    if not all_segments:
        raise ValueError(f"目录中没有可用样本: {folder_path}")

    X = np.vstack(all_segments)
    y = np.array(all_labels, dtype=np.int64)
    s = np.array(all_speeds, dtype=np.float32)
    np.save(os.path.join(PROC_PATH, f"{save_name}_X.npy"), X)
    np.save(os.path.join(PROC_PATH, f"{save_name}_y.npy"), y)
    np.save(os.path.join(PROC_PATH, f"{save_name}_s.npy"), s)
    unique, counts = np.unique(y, return_counts=True)
    print(f"[OK] {save_name}: {X.shape[0]} samples saved. labels={dict(zip(unique.tolist(), counts.tolist()))} speed_range=({s.min():.4f}, {s.max():.4f})")


if __name__ == "__main__":
    for name, info in cfg["datasets"].items():
        process_single_folder(info["path"], name)
    print("数据预处理完成，已保存到 1_data/processed/")
