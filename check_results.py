import os
import json
import re
from pathlib import Path

# ================= 你的配置区域 =================
# 1. 结果都在 logs 文件夹里
LOGS_DIR = Path("4_results/logs")

# 2. 你想看哪个模型的？(根据截图，你有 dann 和 pdan)
# 修改这里：改成 "dann" 或者 "pdan"
MODEL_NAME = "dann" 

# 3. 你的四个任务后缀
TARGETS = ["2170", "5320", "7500", "9710"]

# 4. 源域名字
SOURCE = "experiment"
# ===============================================

def find_best_accuracy_in_folder(folder_path):
    """
    在一个文件夹里翻箱倒柜，寻找准确率数字
    """
    if not folder_path.exists():
        return "❌ 文件夹不存在"

    # 获取文件夹里所有文件
    files = list(folder_path.iterdir())
    
    # --- 策略 A: 优先找 .json 文件 (通常是最准的) ---
    json_files = [f for f in files if f.suffix == '.json']
    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 常见字段名探测
                for key in ['acc', 'accuracy', 'test_acc', 'val_acc']:
                    if key in data:
                        return f"{data[key]*100:.2f}% (JSON)"
                # 深度探测 report 字段
                if 'report' in data and 'accuracy' in data['report']:
                    return f"{data['report']['accuracy']*100:.2f}% (JSON)"
        except:
            pass # json读不出来就继续往下试

    # --- 策略 B: 如果没 JSON，就去扒 .log 或 .txt 文件 ---
    log_files = [f for f in files if f.suffix in ['.log', '.txt', '.out']]
    # 按修改时间排序，找最新的日志
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for lf in log_files:
        try:
            with open(lf, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # 使用正则表达式寻找 "Accuracy: 0.xxxx" 或者 "acc: 0.xxxx"
                # 这里的正则会匹配最后一次出现的准确率
                matches = re.findall(r'(?:ccuracy|acc)[a-zA-Z\s_]*[:=]\s*(\d+\.?\d+)', content, re.IGNORECASE)
                
                if matches:
                    # 取最后一个找到的数值（通常是训练结束时的）
                    acc_val = float(matches[-1])
                    # 如果数值小于1，比如0.95，就乘100；如果是95，就不乘
                    if acc_val <= 1.0: 
                        acc_val *= 100
                    return f"{acc_val:.2f}% (Log)"
        except:
            pass

    return "⚠️ 未找到数据"

# ================= 主程序 =================
print(f"\n{'='*55}")
print(f"📊 批量提取准确率报告 | 模型: {MODEL_NAME}")
print(f"{'='*55}")
print(f"{'任务文件夹 (Task)':<40} | {'准确率 (Accuracy)':<15}")
print(f"{'-'*55}")

for tgt in TARGETS:
    # 拼凑文件夹名字: pdan_experiment_to_centrifuge_2170
    folder_name = f"{MODEL_NAME}_{SOURCE}_to_centrifuge_{tgt}"
    folder_path = LOGS_DIR / folder_name
    
    # 获取结果
    result = find_best_accuracy_in_folder(folder_path)
    
    # 打印表格行
    print(f"{folder_name:<40} | {result:<15}")

print(f"{'='*55}\n")