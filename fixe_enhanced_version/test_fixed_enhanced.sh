#!/bin/bash
# 快速测试修复后的增强版

# 仅使用 2号和3号 GPU
export CUDA_VISIBLE_DEVICES=2,3

PYTHON=/media/user/data3/toky/CondaEnvs/SparseGPT/bin/python
MODEL="facebook/opt-125m"
DATASET="c4"

echo "================================================================================"
echo "测试修复后的增强版"
echo "================================================================================"
echo ""
echo "基准对比:"
echo "  原版 50%+4bit: C4=37.23, WikiText2=38.86, PTB=61.13"
echo "  目标: PPL < 50 (不崩溃), 理想 PPL ≈ 37-40"
echo ""
echo "================================================================================"
echo ""

# 创建测试日志目录
mkdir -p test_logs

echo "测试 1: 增强版 Quantile方法 (4.0 bits)"
echo "开始时间: $(date)"
cd /media/user/data3/toky/Projects/SparseGPT
$PYTHON enhanced_version/opt_enhanced.py $MODEL $DATASET \
    --sparsity 0.5 --wbits 4 \
    --target_avg_bits 4.0 --bit_method quantile \
    2>&1 | tee test_logs/fixed_quantile_4bit.log
echo "完成时间: $(date)"
echo ""

echo "================================================================================"
echo "提取结果:"
echo "================================================================================"
echo ""
echo "WikiText2: $(grep 'Perplexity on wikitext2:' test_logs/fixed_quantile_4bit.log | tail -1)"
echo "PTB:       $(grep 'Perplexity on ptb:' test_logs/fixed_quantile_4bit.log | tail -1)"
echo "C4:        $(grep 'Perplexity on c4:' test_logs/fixed_quantile_4bit.log | tail -1)"
echo ""
echo "对比基准 (原版50%+4bit):"
echo "  C4:        37.23"
echo "  WikiText2: 38.86"
echo "  PTB:       61.13"
echo ""
echo "================================================================================"

