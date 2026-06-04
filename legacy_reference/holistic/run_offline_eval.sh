#!/bin/bash

# ================= 配置区域 =================

# 1. 基础路径
BASE_DIR="/path/to/project"

# 2. 默认要处理的数据集 (可以按需修改)
# 例如: ("gaia" "agentnetbench" "framethinker")
# TARGET_BENCHMARKS=("agentnetbench" "framethinker" "gaia" "gta" "skywork" "tool_bench")
TARGET_BENCHMARKS=("agentnetbench" "framethinker" "gaia" "gta" "tool_bench")

# 允许命令行传参覆盖默认列表 (使用方式: sh run_eval.sh gaia gta)
if [ $# -gt 0 ]; then
    TARGET_BENCHMARKS=("$@")
fi

# ================= 环境准备 =================

echo "正在加载环境..."
source ~/.bashrc
source ~/miniconda3/bin/activate vllm2

# ================= 开始执行评测 =================

for bench_name in "${TARGET_BENCHMARKS[@]}"; do
    WORK_DIR="${BASE_DIR}/${bench_name}"
    
    echo "----------------------------------------------------------------"
    echo "🔍 准备评测数据集: ${bench_name}"
    
    if [ ! -d "$WORK_DIR" ]; then
        echo "❌ 错误: 找不到目录 $WORK_DIR，跳过..."
        continue
    fi

    cd "$WORK_DIR" || exit

    if [ ! -f "offline_eval.py" ]; then
        echo "❌ 错误: 在 ${bench_name} 中找不到 offline_eval.py，跳过..."
        cd ".."
        continue
    fi

    echo "🚀 提交评测任务..."
    
    # 注意：这里将 job-name 修改为了 eval_前缀
    srun --partition=ai_moe2 \
         --job-name="eval_${bench_name}" \
         --mpi=pmi2 \
         -n1 \
         --ntasks-per-node=1 \
         -c 2 \
         --kill-on-bad-exit=1 \
         --quotatype=reserved \
         python offline_eval.py

    echo "✅ ${bench_name} 评测任务已运行"
    
    # 切回基础目录
    cd "$BASE_DIR" || exit
    
    sleep 1
done


for bench_name in "${TARGET_BENCHMARKS[@]}"; do
    WORK_DIR="${BASE_DIR}/${bench_name}"
    
    echo "----------------------------------------------------------------"
    echo "🔍 准备评测数据集: ${bench_name}"
    
    if [ ! -d "$WORK_DIR" ]; then
        echo "❌ 错误: 找不到目录 $WORK_DIR，跳过..."
        continue
    fi

    cd "$WORK_DIR" || exit

    if [ ! -f "offline_eval.py" ]; then
        echo "❌ 错误: 在 ${bench_name} 中找不到 offline_eval.py，跳过..."
        cd ".."
        continue
    fi

    echo "🚀 提交评测任务..."
    
    # 注意：这里将 job-name 修改为了 eval_前缀
    srun --partition=ai_moe2 \
         --job-name="eval_${bench_name}" \
         --mpi=pmi2 \
         -n1 \
         --ntasks-per-node=1 \
         -c 2 \
         --kill-on-bad-exit=1 \
         --quotatype=reserved \
         python offline_eval.py

    echo "✅ ${bench_name} 评测任务已运行"
    
    # 切回基础目录
    cd "$BASE_DIR" || exit
    
    sleep 1
done


for bench_name in "${TARGET_BENCHMARKS[@]}"; do
    WORK_DIR="${BASE_DIR}/${bench_name}"
    
    echo "----------------------------------------------------------------"
    echo "🔍 准备评测数据集: ${bench_name}"
    
    if [ ! -d "$WORK_DIR" ]; then
        echo "❌ 错误: 找不到目录 $WORK_DIR，跳过..."
        continue
    fi

    cd "$WORK_DIR" || exit

    if [ ! -f "offline_eval.py" ]; then
        echo "❌ 错误: 在 ${bench_name} 中找不到 offline_eval.py，跳过..."
        cd ".."
        continue
    fi

    echo "🚀 提交评测任务..."
    
    # 注意：这里将 job-name 修改为了 eval_前缀
    srun --partition=ai_moe2 \
         --job-name="eval_${bench_name}" \
         --mpi=pmi2 \
         -n1 \
         --ntasks-per-node=1 \
         -c 2 \
         --kill-on-bad-exit=1 \
         --quotatype=reserved \
         python offline_eval.py

    echo "✅ ${bench_name} 评测任务已运行"
    
    # 切回基础目录
    cd "$BASE_DIR" || exit
    
    sleep 1
done


for bench_name in "${TARGET_BENCHMARKS[@]}"; do
    WORK_DIR="${BASE_DIR}/${bench_name}"
    
    echo "----------------------------------------------------------------"
    echo "🔍 准备评测数据集: ${bench_name}"
    
    if [ ! -d "$WORK_DIR" ]; then
        echo "❌ 错误: 找不到目录 $WORK_DIR，跳过..."
        continue
    fi

    cd "$WORK_DIR" || exit

    if [ ! -f "offline_eval.py" ]; then
        echo "❌ 错误: 在 ${bench_name} 中找不到 offline_eval.py，跳过..."
        cd ".."
        continue
    fi

    echo "🚀 提交评测任务..."
    
    # 注意：这里将 job-name 修改为了 eval_前缀
    srun --partition=ai_moe2 \
         --job-name="eval_${bench_name}" \
         --mpi=pmi2 \
         -n1 \
         --ntasks-per-node=1 \
         -c 2 \
         --kill-on-bad-exit=1 \
         --quotatype=reserved \
         python offline_eval.py

    echo "✅ ${bench_name} 评测任务已运行"
    
    # 切回基础目录
    cd "$BASE_DIR" || exit
    
    sleep 1
done


for bench_name in "${TARGET_BENCHMARKS[@]}"; do
    WORK_DIR="${BASE_DIR}/${bench_name}"
    
    echo "----------------------------------------------------------------"
    echo "🔍 准备评测数据集: ${bench_name}"
    
    if [ ! -d "$WORK_DIR" ]; then
        echo "❌ 错误: 找不到目录 $WORK_DIR，跳过..."
        continue
    fi

    cd "$WORK_DIR" || exit

    if [ ! -f "offline_eval.py" ]; then
        echo "❌ 错误: 在 ${bench_name} 中找不到 offline_eval.py，跳过..."
        cd ".."
        continue
    fi

    echo "🚀 提交评测任务..."
    
    # 注意：这里将 job-name 修改为了 eval_前缀
    srun --partition=ai_moe2 \
         --job-name="eval_${bench_name}" \
         --mpi=pmi2 \
         -n1 \
         --ntasks-per-node=1 \
         -c 2 \
         --kill-on-bad-exit=1 \
         --quotatype=reserved \
         python offline_eval.py

    echo "✅ ${bench_name} 评测任务已运行"
    
    # 切回基础目录
    cd "$BASE_DIR" || exit
    
    sleep 1
done


for bench_name in "${TARGET_BENCHMARKS[@]}"; do
    WORK_DIR="${BASE_DIR}/${bench_name}"
    
    echo "----------------------------------------------------------------"
    echo "🔍 准备评测数据集: ${bench_name}"
    
    if [ ! -d "$WORK_DIR" ]; then
        echo "❌ 错误: 找不到目录 $WORK_DIR，跳过..."
        continue
    fi

    cd "$WORK_DIR" || exit

    if [ ! -f "offline_eval.py" ]; then
        echo "❌ 错误: 在 ${bench_name} 中找不到 offline_eval.py，跳过..."
        cd ".."
        continue
    fi

    echo "🚀 提交评测任务..."
    
    # 注意：这里将 job-name 修改为了 eval_前缀
    srun --partition=ai_moe2 \
         --job-name="eval_${bench_name}" \
         --mpi=pmi2 \
         -n1 \
         --ntasks-per-node=1 \
         -c 2 \
         --kill-on-bad-exit=1 \
         --quotatype=reserved \
         python offline_eval.py

    echo "✅ ${bench_name} 评测任务已运行"
    
    # 切回基础目录
    cd "$BASE_DIR" || exit
    
    sleep 1
done


for bench_name in "${TARGET_BENCHMARKS[@]}"; do
    WORK_DIR="${BASE_DIR}/${bench_name}"
    
    echo "----------------------------------------------------------------"
    echo "🔍 准备评测数据集: ${bench_name}"
    
    if [ ! -d "$WORK_DIR" ]; then
        echo "❌ 错误: 找不到目录 $WORK_DIR，跳过..."
        continue
    fi

    cd "$WORK_DIR" || exit

    if [ ! -f "offline_eval.py" ]; then
        echo "❌ 错误: 在 ${bench_name} 中找不到 offline_eval.py，跳过..."
        cd ".."
        continue
    fi

    echo "🚀 提交评测任务..."
    
    # 注意：这里将 job-name 修改为了 eval_前缀
    srun --partition=ai_moe2 \
         --job-name="eval_${bench_name}" \
         --mpi=pmi2 \
         -n1 \
         --ntasks-per-node=1 \
         -c 2 \
         --kill-on-bad-exit=1 \
         --quotatype=reserved \
         python offline_eval.py

    echo "✅ ${bench_name} 评测任务已运行"
    
    # 切回基础目录
    cd "$BASE_DIR" || exit
    
    sleep 1
done


for bench_name in "${TARGET_BENCHMARKS[@]}"; do
    WORK_DIR="${BASE_DIR}/${bench_name}"
    
    echo "----------------------------------------------------------------"
    echo "🔍 准备评测数据集: ${bench_name}"
    
    if [ ! -d "$WORK_DIR" ]; then
        echo "❌ 错误: 找不到目录 $WORK_DIR，跳过..."
        continue
    fi

    cd "$WORK_DIR" || exit

    if [ ! -f "offline_eval.py" ]; then
        echo "❌ 错误: 在 ${bench_name} 中找不到 offline_eval.py，跳过..."
        cd ".."
        continue
    fi

    echo "🚀 提交评测任务..."
    
    # 注意：这里将 job-name 修改为了 eval_前缀
    srun --partition=ai_moe2 \
         --job-name="eval_${bench_name}" \
         --mpi=pmi2 \
         -n1 \
         --ntasks-per-node=1 \
         -c 2 \
         --kill-on-bad-exit=1 \
         --quotatype=reserved \
         python offline_eval.py

    echo "✅ ${bench_name} 评测任务已运行"
    
    # 切回基础目录
    cd "$BASE_DIR" || exit
    
    sleep 1
done


for bench_name in "${TARGET_BENCHMARKS[@]}"; do
    WORK_DIR="${BASE_DIR}/${bench_name}"
    
    echo "----------------------------------------------------------------"
    echo "🔍 准备评测数据集: ${bench_name}"
    
    if [ ! -d "$WORK_DIR" ]; then
        echo "❌ 错误: 找不到目录 $WORK_DIR，跳过..."
        continue
    fi

    cd "$WORK_DIR" || exit

    if [ ! -f "offline_eval.py" ]; then
        echo "❌ 错误: 在 ${bench_name} 中找不到 offline_eval.py，跳过..."
        cd ".."
        continue
    fi

    echo "🚀 提交评测任务..."
    
    # 注意：这里将 job-name 修改为了 eval_前缀
    srun --partition=ai_moe2 \
         --job-name="eval_${bench_name}" \
         --mpi=pmi2 \
         -n1 \
         --ntasks-per-node=1 \
         -c 2 \
         --kill-on-bad-exit=1 \
         --quotatype=reserved \
         python offline_eval.py

    echo "✅ ${bench_name} 评测任务已运行"
    
    # 切回基础目录
    cd "$BASE_DIR" || exit
    
    sleep 1
done

for bench_name in "${TARGET_BENCHMARKS[@]}"; do
    WORK_DIR="${BASE_DIR}/${bench_name}"
    
    echo "----------------------------------------------------------------"
    echo "🔍 准备评测数据集: ${bench_name}"
    
    if [ ! -d "$WORK_DIR" ]; then
        echo "❌ 错误: 找不到目录 $WORK_DIR，跳过..."
        continue
    fi

    cd "$WORK_DIR" || exit

    if [ ! -f "offline_eval.py" ]; then
        echo "❌ 错误: 在 ${bench_name} 中找不到 offline_eval.py，跳过..."
        cd ".."
        continue
    fi

    echo "🚀 提交评测任务..."
    
    # 注意：这里将 job-name 修改为了 eval_前缀
    srun --partition=ai_moe2 \
         --job-name="eval_${bench_name}" \
         --mpi=pmi2 \
         -n1 \
         --ntasks-per-node=1 \
         -c 2 \
         --kill-on-bad-exit=1 \
         --quotatype=reserved \
         python offline_eval.py

    echo "✅ ${bench_name} 评测任务已运行"
    
    # 切回基础目录
    cd "$BASE_DIR" || exit
    
    sleep 1
done

echo "----------------------------------------------------------------"
echo "🎉 所有评测流程执行完毕。"