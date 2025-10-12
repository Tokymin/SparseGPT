#!/bin/bash
# 快速测试脚本（修复版）- 验证增强版是否正常工作

echo "=========================================="
echo "增强版快速验证测试"
echo "=========================================="
echo ""

echo "测试 1: 基础功能测试"
echo "--------------------"
cd .. && python enhanced_version/test_enhanced.py | head -80
cd enhanced_version
echo ""
echo "✅ 如果看到 '测试完成! ✅'，说明基础功能正常"
echo ""

read -p "按回车继续 OPT-125M 测试..."

echo ""
echo "测试 2: OPT-125M 单次测试（修复版）"
echo "------------------------------------"
echo "命令: python enhanced_version/opt_enhanced.py facebook/opt-125m c4 --sparsity 0.5 --wbits 4 --target_avg_bits 4.0 --bit_method quantile"
echo ""

cd .. && python enhanced_version/opt_enhanced.py facebook/opt-125m c4 \
    --sparsity 0.5 \
    --wbits 4 \
    --target_avg_bits 4.0 \
    --bit_method quantile
cd enhanced_version

echo ""
echo "=========================================="
echo "验证检查项："
echo "=========================================="
echo ""
echo "✅ 检查1: 统计信息中 '总通道数' 应该 > 0"
echo "✅ 检查2: 应该看到 5档比特分布 (2/3/4/6/8 bit)"
echo "✅ 检查3: 平均比特数应该接近 4.6 bits"
echo "✅ 检查4: 每层统计应该有数据"
echo ""
echo "如果以上都满足，说明增强版工作正常！🎉"
echo ""

