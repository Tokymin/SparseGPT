#!/bin/bash
# 完整基准测试脚本

PYTHON=/media/user/data3/toky/CondaEnvs/SparseGPT/bin/python
MODEL="facebook/opt-125m"
DATASET="c4"
SPARSITY="0.5"

echo "================================================================================"
echo "SparseGPT 完整基准测试"
echo "================================================================================"
echo ""
echo "模型: $MODEL"
echo "数据集: $DATASET"
echo "剪枝率: $SPARSITY (50%)"
echo ""
echo "================================================================================"
echo ""

# 创建日志目录
mkdir -p baseline_logs

# 测试 1: 原版 Dense（无压缩）
echo "测试 1/5: 原版 Dense（无压缩）"
echo "预期: PPL ≈ 27-28"
echo "开始时间: $(date)"
$PYTHON opt.py $MODEL $DATASET 2>&1 | tee baseline_logs/01_original_dense.log
echo "完成时间: $(date)"
echo ""
echo "--------------------------------------------------------------------------------"
echo ""

# 测试 2: 原版 + 50%剪枝（无量化）
echo "测试 2/5: 原版 + 50%剪枝（无量化）"
echo "预期: PPL ≈ 28-30"
echo "开始时间: $(date)"
$PYTHON opt.py $MODEL $DATASET --sparsity $SPARSITY 2>&1 | tee baseline_logs/02_original_prune_only.log
echo "完成时间: $(date)"
echo ""
echo "--------------------------------------------------------------------------------"
echo ""

# 测试 3: 原版 + 50%剪枝 + 4bit量化 ⭐ 最重要
echo "测试 3/5: 原版 + 50%剪枝 + 4bit量化 ⭐ 关键基准"
echo "预期: PPL ≈ 32-38"
echo "开始时间: $(date)"
$PYTHON opt.py $MODEL $DATASET --sparsity $SPARSITY --wbits 4 2>&1 | tee baseline_logs/03_original_prune_quant4.log
echo "完成时间: $(date)"
echo ""
echo "--------------------------------------------------------------------------------"
echo ""

# 测试 4: Toky版 + 50%剪枝 + 4bit量化
echo "测试 4/5: Toky版 + 50%剪枝 + 4bit量化"
echo "预期: PPL ≈ 32-40"
echo "开始时间: $(date)"
$PYTHON opt_toky.py $MODEL $DATASET --sparsity $SPARSITY --wbits 4 2>&1 | tee baseline_logs/04_toky_prune_quant4.log
echo "完成时间: $(date)"
echo ""
echo "--------------------------------------------------------------------------------"
echo ""

# 测试 5: 增强版（当前有bug，仅作对比）
echo "测试 5/5: 增强版（当前版本，已知有bug）"
echo "预期: 需要修复后才能评估"
echo "开始时间: $(date)"
cd enhanced_version
$PYTHON opt_enhanced.py $MODEL $DATASET --sparsity $SPARSITY --wbits 4 \
    --target_avg_bits 4.0 --bit_method quantile 2>&1 | tee ../baseline_logs/05_enhanced_current.log
cd ..
echo "完成时间: $(date)"
echo ""

# 提取关键结果
echo "================================================================================"
echo "测试完成！提取关键结果..."
echo "================================================================================"
echo ""

for i in {1..5}; do
    logfile=$(ls baseline_logs/0${i}_*.log 2>/dev/null | head -1)
    if [ -f "$logfile" ]; then
        echo "测试 $i: $(basename $logfile)"
        echo "  C4:        $(grep 'Perplexity on c4:' $logfile | tail -1)"
        echo "  WikiText2: $(grep 'Perplexity on wikitext2:' $logfile | tail -1)"
        echo "  PTB:       $(grep 'Perplexity on ptb:' $logfile | tail -1)"
        echo ""
    fi
done

echo "================================================================================"
echo "所有测试完成！"
echo "日志保存在: baseline_logs/"
echo "================================================================================"

