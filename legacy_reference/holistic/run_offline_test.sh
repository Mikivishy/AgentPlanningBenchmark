#!/bin/bash

# ================= 配置区域 =================

# 1. 基础路径 (你的 Selected_benchs 所在位置)
BASE_DIR="/path/to/project"

# 2. 设置你要处理的数据集名称
TARGET_BENCHMARKS=("agentnetbench" "framethinker" "gaia" "gta" "skywork" "tool_bench")

# 如果你想通过命令行传参覆盖上面的设置 (例如: sh run.sh gta)，可以保留下面这行逻辑
if [ $# -gt 0 ]; then
    TARGET_BENCHMARKS=("$@")
fi

# ================= 环境准备 =================

echo "正在加载环境..."
source ~/.bashrc
source ~/miniconda3/bin/activate vllm2

# ================= 开始执行 =================

for bench_name in "${TARGET_BENCHMARKS[@]}"; do
    WORK_DIR="${BASE_DIR}/${bench_name}"
    
    echo "----------------------------------------------------------------"
    echo "准备处理数据集: ${bench_name}"
    echo "工作目录: ${WORK_DIR}"
    
    # 检查目录是否存在
    if [ ! -d "$WORK_DIR" ]; then
        echo "❌ 错误: 找不到目录 $WORK_DIR，跳过..."
        continue
    fi

    # 进入目录 (很多 python 脚本依赖相对路径引用，所以最好 cd 进去)
    cd "$WORK_DIR" || exit

    # 检查脚本是否存在
    if [ ! -f "offline_test.py" ]; then
        echo "❌ 错误: 在 ${bench_name} 中找不到 offline_test.py，跳过..."
        cd ".."
        continue
    fi

    echo "🚀 开始提交任务..."
    
    # 执行 srun 命令
    # 注意：这里使用了 job-name 动态包含数据集名称，方便你在 squeue 中查看
    srun --partition=ai_moe2 \
         --job-name="test_${bench_name}" \
         --mpi=pmi2 \
         -n1 \
         --ntasks-per-node=1 \
         -c 2 \
         --kill-on-bad-exit=1 \
         --quotatype=reserved \
         python offline_test.py

    echo "✅ ${bench_name} 任务提交/运行完成"
    
    # 返回上一级/基础目录，以免影响下一次循环（虽然直接用绝对路径cd也可以）
    cd "$BASE_DIR" || exit
    
    # (可选) 如果是排队提交，可以sleep几秒防止提交过快
    sleep 2
done

echo "----------------------------------------------------------------"
echo "🎉 所有指定的 Benchmark 流程已结束。"