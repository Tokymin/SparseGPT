#!/bin/bash
# 互信息量化快速测试脚本

PYTHON=/media/user/data3/toky/CondaEnvs/SparseGPT/bin/python
MODEL="facebook/opt-125m"
DATASET="c4"

# 仅使用 GPU 2和3
export CUDA_VISIBLE_DEVICES=2,3

# 结果目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULT_DIR="$SCRIPT_DIR/test_results"
mkdir -p $RESULT_DIR

echo "========================================================================"
echo "互信息量化测试"
echo "========================================================================"
echo ""
echo "配置:"
echo "  模型: $MODEL"
echo "  数据集: $DATASET"
echo "  稀疏度: 0.5"
echo "  目标比特: 4.0"
echo ""
echo "========================================================================"
echo ""

# ========================================================================
# 测试1: 基准（原版）
# ========================================================================
echo "测试 1/3: 原版 SparseGPT (基准)"
echo "开始时间: $(date)"
echo "------------------------------------------------------------------------"

cd /media/user/data3/toky/Projects/SparseGPT

$PYTHON opt.py $MODEL $DATASET \
    --sparsity 0.5 --wbits 4 \
    2>&1 | tee $RESULT_DIR/baseline_original.log

echo "完成时间: $(date)"
echo ""

# ========================================================================
# 测试2: 增强版（不使用MI）
# ========================================================================
echo "测试 2/3: 增强版 SparseGPT (无MI分组)"
echo "开始时间: $(date)"
echo "------------------------------------------------------------------------"

$PYTHON mutual_info_quantization/opt_mi.py $MODEL $DATASET \
    --sparsity 0.5 --wbits 4 \
    --target_avg_bits 4.0 \
    --use_mi_grouping 0 \
    2>&1 | tee $RESULT_DIR/enhanced_no_mi.log

echo "完成时间: $(date)"
echo ""

# ========================================================================
# 测试3: MI改进版
# ========================================================================
echo "测试 3/3: MI改进版 SparseGPT"
echo "开始时间: $(date)"
echo "------------------------------------------------------------------------"

$PYTHON mutual_info_quantization/opt_mi.py $MODEL $DATASET \
    --sparsity 0.5 --wbits 4 \
    --target_avg_bits 4.0 \
    --use_mi_grouping 1 \
    --n_groups 10 \
    2>&1 | tee $RESULT_DIR/mi_enhanced.log

echo "完成时间: $(date)"
echo ""

# ========================================================================
# 结果提取和对比
# ========================================================================
echo "========================================================================"
echo "结果对比"
echo "========================================================================"
echo ""

# 提取PPL结果
echo "WikiText2 PPL:"
echo "------------------------------------------------------------------------"
echo -n "  原版:      "
grep "Perplexity on wikitext2:" $RESULT_DIR/baseline_original.log 2>/dev/null | tail -1 | awk '{print $NF}'

echo -n "  增强版:    "
grep "Perplexity on wikitext2:" $RESULT_DIR/enhanced_no_mi.log 2>/dev/null | tail -1 | awk '{print $NF}'

echo -n "  MI改进版:  "
grep "Perplexity on wikitext2:" $RESULT_DIR/mi_enhanced.log 2>/dev/null | tail -1 | awk '{print $NF}'

echo ""
echo "PTB PPL:"
echo "------------------------------------------------------------------------"
echo -n "  原版:      "
grep "Perplexity on ptb:" $RESULT_DIR/baseline_original.log 2>/dev/null | tail -1 | awk '{print $NF}'

echo -n "  增强版:    "
grep "Perplexity on ptb:" $RESULT_DIR/enhanced_no_mi.log 2>/dev/null | tail -1 | awk '{print $NF}'

echo -n "  MI改进版:  "
grep "Perplexity on ptb:" $RESULT_DIR/mi_enhanced.log 2>/dev/null | tail -1 | awk '{print $NF}'

echo ""
echo "C4 PPL:"
echo "------------------------------------------------------------------------"
echo -n "  原版:      "
grep "Perplexity on c4:" $RESULT_DIR/baseline_original.log 2>/dev/null | tail -1 | awk '{print $NF}'

echo -n "  增强版:    "
grep "Perplexity on c4:" $RESULT_DIR/enhanced_no_mi.log 2>/dev/null | tail -1 | awk '{print $NF}'

echo -n "  MI改进版:  "
grep "Perplexity on c4:" $RESULT_DIR/mi_enhanced.log 2>/dev/null | tail -1 | awk '{print $NF}'

echo ""
echo "========================================================================"
echo "测试完成！"
echo "详细日志保存在: $RESULT_DIR/"
echo "========================================================================"

