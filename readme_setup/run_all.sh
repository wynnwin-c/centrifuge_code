#!/bin/bash
echo "============================================"
echo "  Bearing Transfer - 全部迁移实验启动"
echo "============================================"



cd 3_train || exit

echo "[1/4] 运行 2170rpm 任务 ..."
python run_task.py --target 2170

echo "[2/4] 运行 5320rpm 任务 ..."
python run_task.py --target 5320

echo "[3/4] 运行 7500rpm 任务 ..."
python run_task.py --target 7500

echo "[4/4] 运行 9710rpm 任务 ..."
python run_task.py --target 9710

cd ..
echo
echo "✅ 所有实验运行完成！"
echo "结果保存在 4_results/ 目录下。"
