import numpy as np
import os
from pathlib import Path

# ================= 配置区域 =================
# 1. 数据所在的文件夹路径 (根据你的项目结构)
DATA_DIR = Path("1_data/processed")

# 2. 源域（训练集）的名字
SOURCE_NAME = "experiment"

# 3. 目标域（测试集）的名字 (你可以改成 5320, 7500 等)
TARGET_NAME = "centrifuge_7500" 
# ===========================================

def get_sample_count(domain_name, type_label):
    """
    读取 .npy 文件并返回样本数量
    """
    # 拼接文件路径：例如 1_data/processed/experiment_X.npy
    file_path = DATA_DIR / f"{domain_name}_X.npy"
    
    if not file_path.exists():
        print(f"❌ 错误：找不到文件 -> {file_path}")
        print(f"   请检查 1_data/processed 文件夹里有没有这个文件。")
        return None
    
    try:
        # 只加载元数据，不加载整个文件到内存 (mmap_mode='r')，速度极快
        data = np.load(file_path, mmap_mode='r')
        count = data.shape[0] # 第一个维度通常是样本数量 (N, C, L)
        print(f"✅ {type_label} [{domain_name}]: \t{count} 个样本")
        return count
    except Exception as e:
        print(f"⚠️ 读取文件出错: {e}")
        return None

if __name__ == "__main__":
    print(f"{'='*30}")
    print(f"正在检查数据样本数量...")
    print(f"数据目录: {DATA_DIR.resolve()}")
    print(f"{'='*30}\n")

    # 1. 检查训练集 (源域)
    train_count = get_sample_count(SOURCE_NAME, "训练集 (Source)")

    # 2. 检查测试集 (目标域)
    test_count = get_sample_count(TARGET_NAME, "测试集 (Target)")

    print(f"\n{'='*30}")
    if train_count and test_count:
        print(f"📊 总结:")
        print(f"   - 训练样本: {train_count}")
        print(f"   - 测试样本: {test_count}")
        print(f"   - 总样本数: {train_count + test_count}")
    print(f"{'='*30}")