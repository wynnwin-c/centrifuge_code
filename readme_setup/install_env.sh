#!/bin/bash
echo "============================================"
echo "  创建虚拟环境并安装依赖"
echo "============================================"

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级 pip
python -m pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt

echo
echo "✅ 环境安装完成！"
echo "请执行 ./run_all.sh 运行全部实验。"
