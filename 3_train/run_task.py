# 3_train/run_task.py
import os
import sys
import argparse
import subprocess
import yaml
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def find_task(cfg: dict, target_rpm: int) -> str:
    name = f"exp_to_{target_rpm}"
    for t in cfg["tasks"]:
        if t["name"] == name:
            return t["name"]
    raise ValueError(f"任务未找到: {name}")

def run_one(task_name: str, model: str, extra_args=None):
    cmd = [sys.executable, str(ROOT / "3_train" / "train.py"), "--task", task_name, "--model", model]    # python 3_train/train.py --task <task_name> --model <model> [extra_args...]
    if extra_args:
        cmd += extra_args
    print(" ".join(cmd)) #拼成python 3_train/train.py --task <task_name> --model <model> [extra_args...]
    subprocess.run(cmd, check=True) #运行cmd程序，并检查是否成功

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--config", type=str, default=str(ROOT / "3_train" / "config_train.yaml"))
    parser.add_argument("--models", type=str, nargs="+", default=["ddc", "dann", "edann", "san"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    #cfg是一个config字典
    cfg = load_config(args.config)
    task_name = find_task(cfg, args.target)

    extra = []
    if args.epochs is not None:
        extra += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        extra += ["--batch_size", str(args.batch_size)]
    if args.lr is not None:
        extra += ["--lr", str(args.lr)]

    for m in args.models:
        run_one(task_name, m, extra_args=extra)

if __name__ == "__main__":
    main()
