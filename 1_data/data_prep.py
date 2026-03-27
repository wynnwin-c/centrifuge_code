# 1_data/data_prep.py
import os
import numpy as np
import pandas as pd
from scipy.io import savemat
import yaml

# 读取配置文件，cfg是一个字典
with open("1_data/config_dataset.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

RAW_PATH = cfg["raw_path"]
PROC_PATH = cfg["processed_path"]
SAMPLING_RATE = cfg["sampling_rate"]
SLICE_TIME = cfg["slice_time"]
OVERLAP = cfg["overlap"]
EPOCHS_PER_FILE = int(SLICE_TIME * SAMPLING_RATE)#每一个切片包含多少个数据点 15s*25600hz=384000

os.makedirs(PROC_PATH, exist_ok=True)

def slice_signal(sig, window, overlap):
    step = int(window * (1 - overlap))#step=25600*0.5=12800
    segments = []
    for i in range(0, len(sig) - window, step):
        seg = sig[i:i + window]
        segments.append(seg)
    return np.array(segments)#(28, 25600)

def process_single_folder(folder_path, save_name):
    print(f"Processing folder: {folder_path}")
    all_segments = []
    all_labels = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if not f.endswith(".csv"):
                continue
            label = 0 if "N" in f or "Normal" in f else 1 #！！这里label是不是不对啊？？不只有1啊
            df = pd.read_csv(os.path.join(root, f), encoding='iso-8859-1')
            sig = df.iloc[:, 1].values  # 默认第二列是信号
            sig = sig[:EPOCHS_PER_FILE] #取前15秒的数据点
            segments = slice_signal(sig, int(SAMPLING_RATE * cfg["window"]), cfg["overlap"])#window=1s*25600
            all_segments.append(segments)
            all_labels.extend([label] * len(segments)) #每一个离心机：（28*6（6个csv文件），25600）
    X = np.vstack(all_segments)
    y = np.array(all_labels)
    np.save(os.path.join(PROC_PATH, f"{save_name}_X.npy"), X)
    np.save(os.path.join(PROC_PATH, f"{save_name}_y.npy"), y)
    print(f"[OK] {save_name}: {X.shape[0]} samples saved.")

if __name__ == "__main__":
    for name, info in cfg["datasets"].items():
        process_single_folder(info["path"], name)
    print("✅ 数据预处理完成，已保存到 1_data/processed/")
