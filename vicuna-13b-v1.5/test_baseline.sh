#!/bin/bash

# Vicuna模型基准测试脚本
# 测试未压缩模型的性能

set -e

# 设置GPU
export CUDA_VISIBLE_DEVICES=2,3

# 配置
MODEL="/mnt/share/HuggingfaceModels/lmsys/vicuna-13b-v1.5"

echo "=== Vicuna-13B 基准性能测试 ==="
echo "模型: $MODEL"
echo "精度: FP16 (未压缩)"
echo "GPU: 2,3"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 运行基准测试
echo "开始测试..."
echo ""

/media/user/data3/toky/CondaEnvs/SparseGPT/bin/python "$SCRIPT_DIR/test_baseline.py" \
    "$MODEL" \
    2>&1 | tee "$SCRIPT_DIR/baseline_test.log"

echo ""
echo "=== 测试完成 ==="
echo "日志保存在: $SCRIPT_DIR/baseline_test.log"
echo "结果保存在: $SCRIPT_DIR/baseline_results.json"

