#!/bin/bash

# Vicuna模型测试脚本
# 使用GPU 2和3

set -e

# 设置GPU
export CUDA_VISIBLE_DEVICES=2,3

# 配置
MODEL="/mnt/share/HuggingfaceModels/lmsys/vicuna-13b-v1.5"  # 或者你本地的模型路径
DATASET="c4"
NSAMPLES=128
SPARSITY=0.5
WBITS=4
TARGET_AVG_BITS=4.0

echo "=== Vicuna MI量化测试 ==="
echo "模型: $MODEL"
echo "数据集: $DATASET"
echo "样本数: $NSAMPLES"
echo "稀疏度: $SPARSITY"
echo "量化位宽: $WBITS"
echo "目标平均位宽: $TARGET_AVG_BITS"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 运行测试
python "$SCRIPT_DIR/vicuna_mi.py" \
    "$MODEL" \
    "$DATASET" \
    --nsamples "$NSAMPLES" \
    --sparsity "$SPARSITY" \
    --wbits "$WBITS" \
    --target_avg_bits "$TARGET_AVG_BITS" \
    --use_mi_grouping 1 \
    --n_groups 10 \
    2>&1 | tee "$SCRIPT_DIR/vicuna_test.log"

echo ""
echo "=== 测试完成 ==="
echo "日志保存在: $SCRIPT_DIR/vicuna_test.log"

