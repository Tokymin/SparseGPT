#!/bin/bash
# 测试脚本：对比原版和增强版在 OPT-125M 上的效果

echo "=========================================="
echo "测试计划：对比原版 vs 增强版"
echo "=========================================="

# 1. 测试原版（已完成，记录结果）
echo ""
echo "1. 原版 sparsegpt_toky.py 结果（已测试）:"
echo "   WikiText2: 36.096 PPL"
echo "   PTB: 55.914 PPL"
echo "   C4: 35.560 PPL"

# 2. 需要创建增强版的 opt 脚本
echo ""
echo "2. 接下来需要："
echo "   a) 创建 opt_enhanced.py（使用 sparsegpt_enhanced）"
echo "   b) 添加参数 --target_avg_bits 和 --bit_method"
echo "   c) 运行相同测试进行对比"

echo ""
echo "3. 预期改进："
echo "   - 更智能的比特分配 -> PPL 下降 0.5-1.0"
echo "   - 相同模型大小下精度更高"
echo "   - 详细的统计报告"

echo ""
echo "=========================================="
echo "运行命令示例："
echo "=========================================="
echo ""
echo "# 测试增强版（分位数方法，4-bit平均）"
echo "python opt_enhanced.py facebook/opt-125m c4 --sparsity 0.5 --target_avg_bits 4.0 --bit_method quantile"
echo ""
echo "# 测试增强版（预算方法，3.5-bit平均）"
echo "python opt_enhanced.py facebook/opt-125m c4 --sparsity 0.5 --target_avg_bits 3.5 --bit_method budget"
echo ""

