#!/bin/bash
# 对比测试脚本（修复版）：原版 vs 增强版在 OPT-125M 上的效果
# 修复：添加 --wbits 4 参数以启用量化

echo "================================================================================"
echo "SparseGPT 版本对比测试（修复版）"
echo "================================================================================"
echo ""

MODEL="facebook/opt-125m"
DATASET="c4"
SPARSITY="0.5"
WBITS="4"  # 关键：启用4-bit量化

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
echo "   命令: python ../opt_toky.py $MODEL $DATASET --sparsity $SPARSITY --wbits $WBITS"
echo ""
echo "   结果:"
echo "   - WikiText2 PPL: 36.096"
echo "   - PTB PPL:       55.914"
echo "   - C4 PPL:        35.560"
echo "   - 量化策略:      简单3档 (2/4/8 bit)，固定阈值"
echo ""
echo "================================================================================"
echo ""

# 创建日志目录
mkdir -p logs

# 2. 增强版测试
echo "2. 增强版 sparsegpt_enhanced.py 测试:"
echo ""
echo "测试配置 A: 分位数方法，4-bit平均"
echo "   命令: python opt_enhanced.py $MODEL $DATASET --sparsity $SPARSITY \\"
echo "           --wbits $WBITS --target_avg_bits 4.0 --bit_method quantile"
echo ""
read -p "按回车开始测试配置 A..."

cd .. && python enhanced_version/opt_enhanced.py $MODEL $DATASET \
    --sparsity $SPARSITY \
    --wbits $WBITS \
    --target_avg_bits 4.0 \
    --bit_method quantile \
    2>&1 | tee enhanced_version/logs/enhanced_quantile_4bit_fixed.log
cd enhanced_version

echo ""
echo "================================================================================"
echo ""
echo "测试配置 B: 分位数方法，3.5-bit平均"
echo "   命令: python opt_enhanced.py $MODEL $DATASET --sparsity $SPARSITY \\"
echo "           --wbits $WBITS --target_avg_bits 3.5 --bit_method quantile"
echo ""
read -p "按回车开始测试配置 B..."

cd .. && python enhanced_version/opt_enhanced.py $MODEL $DATASET \
    --sparsity $SPARSITY \
    --wbits $WBITS \
    --target_avg_bits 3.5 \
    --bit_method quantile \
    2>&1 | tee enhanced_version/logs/enhanced_quantile_3.5bit_fixed.log
cd enhanced_version

echo ""
echo "================================================================================"
echo ""
echo "测试配置 C: 预算方法，4-bit平均"
echo "   命令: python opt_enhanced.py $MODEL $DATASET --sparsity $SPARSITY \\"
echo "           --wbits $WBITS --target_avg_bits 4.0 --bit_method budget"
echo ""
read -p "按回车开始测试配置 C..."

cd .. && python enhanced_version/opt_enhanced.py $MODEL $DATASET \
    --sparsity $SPARSITY \
    --wbits $WBITS \
    --target_avg_bits 4.0 \
    --bit_method budget \
    2>&1 | tee enhanced_version/logs/enhanced_budget_4bit_fixed.log
cd enhanced_version

echo ""
echo "================================================================================"
echo "所有测试完成！"
echo "================================================================================"
echo ""
echo "结果保存在 logs/ 目录下"
echo "  - logs/enhanced_quantile_4bit_fixed.log"
echo "  - logs/enhanced_quantile_3.5bit_fixed.log"
echo "  - logs/enhanced_budget_4bit_fixed.log"
echo ""
echo "重要：现在应该能看到量化统计信息了！"
echo "请查看日志文件中的 '量化统计摘要' 部分"
echo ""

