#!/bin/bash
# 对比测试脚本：原版 vs 增强版在 OPT-125M 上的效果

echo "================================================================================"
echo "SparseGPT 版本对比测试"
echo "================================================================================"
echo ""

MODEL="facebook/opt-125m"
DATASET="c4"
SPARSITY="0.5"
WBITS="4"

echo "测试配置:"
echo "  模型: $MODEL"
echo "  数据集: $DATASET"
echo "  剪枝率: $SPARSITY (50%)"
echo "  量化: $WBITS-bit"
echo ""
echo "================================================================================"
echo ""

# 1. 原版结果（已测试）
echo "1. 原版 sparsegpt_toky.py 结果（已完成）:"
echo "   命令: python opt_toky.py $MODEL $DATASET --sparsity $SPARSITY"
echo ""
echo "   结果:"
echo "   - WikiText2 PPL: 36.096"
echo "   - PTB PPL:       55.914"
echo "   - C4 PPL:        35.560"
echo "   - 量化策略:      简单3档 (2/4/8 bit)，固定阈值"
echo ""
echo "================================================================================"
echo ""

# 2. 增强版测试
echo "2. 增强版 sparsegpt_enhanced.py 测试:"
echo ""
echo "测试配置 A: 分位数方法，4-bit平均"
echo "   命令: python opt_enhanced.py $MODEL $DATASET --sparsity $SPARSITY \\"
echo "           --target_avg_bits 4.0 --bit_method quantile"
echo ""
read -p "按回车开始测试配置 A..."

python opt_enhanced.py $MODEL $DATASET \
    --sparsity $SPARSITY \
    --target_avg_bits 4.0 \
    --bit_method quantile \
    2>&1 | tee logs/enhanced_quantile_4bit.log

echo ""
echo "================================================================================"
echo ""
echo "测试配置 B: 分位数方法，3.5-bit平均"
echo "   命令: python opt_enhanced.py $MODEL $DATASET --sparsity $SPARSITY \\"
echo "           --target_avg_bits 3.5 --bit_method quantile"
echo ""
read -p "按回车开始测试配置 B..."

python opt_enhanced.py $MODEL $DATASET \
    --sparsity $SPARSITY \
    --target_avg_bits 3.5 \
    --bit_method quantile \
    2>&1 | tee logs/enhanced_quantile_3.5bit.log

echo ""
echo "================================================================================"
echo ""
echo "测试配置 C: 预算方法，4-bit平均"
echo "   命令: python opt_enhanced.py $MODEL $DATASET --sparsity $SPARSITY \\"
echo "           --target_avg_bits 4.0 --bit_method budget"
echo ""
read -p "按回车开始测试配置 C..."

python opt_enhanced.py $MODEL $DATASET \
    --sparsity $SPARSITY \
    --target_avg_bits 4.0 \
    --bit_method budget \
    2>&1 | tee logs/enhanced_budget_4bit.log

echo ""
echo "================================================================================"
echo "所有测试完成！"
echo "================================================================================"
echo ""
echo "结果保存在 logs/ 目录下"
echo "  - logs/enhanced_quantile_4bit.log"
echo "  - logs/enhanced_quantile_3.5bit.log"
echo "  - logs/enhanced_budget_4bit.log"
echo ""
echo "请查看日志文件中的 Perplexity 和 量化统计 部分进行对比"
echo ""

