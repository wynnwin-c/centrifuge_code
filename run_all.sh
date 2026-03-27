#!/usr/bin/env bash
# run_all.sh  — Linux 一键环境 + 预处理 + 训练（根目录执行）

set -Eeuo pipefail

# --- 基本设置 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/chenjingwen/miniconda3/envs/evolvepro/bin/python}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/venv}"
REQ_FILE="${REQ_FILE:-requirements.txt}"

RPM_LIST=(2170 5320 7500 9710)
MODELS_DEFAULT=("pdan" "dann")

TARGET=""
MODELS=()
EPOCHS=""
BATCH=""
LR=""
SKIP_PIP="0"
SKIP_PREP="0"

usage() {
  cat <<EOF
Usage:
  bash run_all.sh [options]

Options:
  --target <rpm>        只跑一个目标转速（默认跑全部：2170/5320/7500/9710）
  --models "<list>"     模型列表，空格分隔（默认: "pdan dann"）
  --epochs <int>        训练轮数
  --batch_size <int>    batch 大小
  --lr <float>          学习率
  --skip-pip            跳过 pip 安装
  --skip-prep           跳过数据预处理
  --python <path>       指定 Python 可执行文件
  --help                帮助
示例：
  bash run_all.sh --target 2170 --models "pdan dann" --epochs 20
EOF
}

# --- 解析参数 ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)       TARGET="$2"; shift 2 ;;
    --models)       IFS=' ' read -r -a MODELS <<< "$2"; shift 2 ;;
    --epochs)       EPOCHS="$2"; shift 2 ;;
    --batch_size)   BATCH="$2"; shift 2 ;;
    --lr)           LR="$2"; shift 2 ;;
    --skip-pip)     SKIP_PIP="1"; shift ;;
    --skip-prep)    SKIP_PREP="1"; shift ;;
    --python)       PYTHON_BIN="$2"; shift 2 ;;
    --help|-h)      usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# --- 选择模型 ---
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=("${MODELS_DEFAULT[@]}")
fi

# --- 选择目标 ---
if [[ -n "$TARGET" ]]; then
  RPM_LIST=("$TARGET")
fi

# --- Python/虚拟环境 ---
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "找不到 Python，可通过 --python 指定；当前值: $PYTHON_BIN"
  exit 1
fi

# if [[ ! -d "$VENV_DIR" ]]; then
#   "$PYTHON_BIN" -m venv "$VENV_DIR"
# fi
# # shellcheck disable=SC1091
# source "$VENV_DIR/bin/activate"

echo "使用 Python 解释器: $PYTHON_BIN"

if [[ "$SKIP_PIP" != "1" ]]; then
  python -m pip install --upgrade pip
  if [[ -f "$REQ_FILE" ]]; then
    pip install -r "$REQ_FILE"
  else
    pip install numpy pandas scipy pyyaml
  fi
fi

# --- 数据预处理 ---
if [[ "$SKIP_PREP" != "1" ]]; then
  python 1_data/data_prep.py
fi

# --- 训练参数拼接 ---
EXTRA=()
[[ -n "$EPOCHS" ]] && EXTRA+=("--epochs" "$EPOCHS")
[[ -n "$BATCH"  ]] && EXTRA+=("--batch_size" "$BATCH")
[[ -n "$LR"     ]] && EXTRA+=("--lr" "$LR")

# --- 执行任务 ---
for rpm in "${RPM_LIST[@]}"; do
  echo "==== 运行目标转速: $rpm ===="
  python 3_train/run_task.py --target "$rpm" --models "${MODELS[@]}" "${EXTRA[@]}"
done

echo "✅ 全部完成。结果在 4_results/ 下（logs / weights / plots）。"
