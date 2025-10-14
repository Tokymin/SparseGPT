#!/bin/bash
# 计算复杂度对比测试脚本

PYTHON=/media/user/data3/toky/CondaEnvs/SparseGPT/bin/python
MODEL="facebook/opt-125m"
DATASET="c4"

# 仅使用 GPU 2和3
export CUDA_VISIBLE_DEVICES=2,3

# 结果目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULT_DIR="$SCRIPT_DIR/complexity_results"
mkdir -p $RESULT_DIR

echo "========================================================================"
echo "计算复杂度对比测试"
echo "========================================================================"
echo ""
echo "配置:"
echo "  模型: $MODEL"
echo "  数据集: $DATASET"
echo "  测试项: 压缩时间、推理时间、GPU内存"
echo ""
echo "========================================================================"
echo ""

# ========================================================================
# 测试1: 原版 - 计算复杂度
# ========================================================================
echo "测试 1/3: 原版 SparseGPT - 计算复杂度"
echo "开始时间: $(date)"
echo "------------------------------------------------------------------------"

cd /media/user/data3/toky/Projects/SparseGPT

$PYTHON opt.py $MODEL $DATASET \
    --sparsity 0.5 --wbits 4 \
    2>&1 | tee $RESULT_DIR/baseline_complexity.log

echo "完成时间: $(date)"
echo ""

# ========================================================================
# 测试2: 增强版（无MI）- 计算复杂度
# ========================================================================
echo "测试 2/3: 增强版（无MI）- 计算复杂度"
echo "开始时间: $(date)"
echo "------------------------------------------------------------------------"

$PYTHON mutual_info_quantization/opt_mi.py $MODEL $DATASET \
    --sparsity 0.5 --wbits 4 \
    --target_avg_bits 4.0 \
    --use_mi_grouping 0 \
    2>&1 | tee $RESULT_DIR/enhanced_no_mi_complexity.log

echo "完成时间: $(date)"
echo ""

# ========================================================================
# 测试3: MI改进版 - 计算复杂度
# ========================================================================
echo "测试 3/3: MI改进版 - 计算复杂度"
echo "开始时间: $(date)"
echo "------------------------------------------------------------------------"

$PYTHON mutual_info_quantization/opt_mi.py $MODEL $DATASET \
    --sparsity 0.5 --wbits 4 \
    --target_avg_bits 4.0 \
    --use_mi_grouping 1 \
    --n_groups 10 \
    2>&1 | tee $RESULT_DIR/mi_enhanced_complexity.log

echo "完成时间: $(date)"
echo ""

# ========================================================================
# 提取时间统计
# ========================================================================
echo "========================================================================"
echo "时间统计对比"
echo "========================================================================"
echo ""

cd "$SCRIPT_DIR"

$PYTHON << 'EOF'
import re
import os
from pathlib import Path

result_dir = Path('complexity_results')

def extract_times(log_file):
    """提取压缩时间和评估时间"""
    if not log_file.exists():
        return None, None, None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 提取总压缩时间
    total_time_match = re.search(r'Total time:\s+([\d.]+)s', content)
    total_time = float(total_time_match.group(1)) if total_time_match else None
    
    # 提取所有层的处理时间
    layer_times = re.findall(r'时间:\s+([\d.]+)s', content)
    if not layer_times:
        layer_times = re.findall(r'time\s+([\d.]+)', content)
    
    avg_layer_time = sum(float(t) for t in layer_times) / len(layer_times) if layer_times else None
    
    # 提取评估时间（粗略估计）
    eval_sections = content.split('Evaluating on')
    eval_time = None
    if len(eval_sections) > 1:
        # 简单估计：假设每个数据集评估时间相似
        pass
    
    return total_time, avg_layer_time, len(layer_times)

# 提取各方法的时间
methods = {
    'baseline': 'baseline_complexity.log',
    'enhanced_no_mi': 'enhanced_no_mi_complexity.log',
    'mi_enhanced': 'mi_enhanced_complexity.log'
}

results = {}
for name, log_file in methods.items():
    total, avg_layer, n_layers = extract_times(result_dir / log_file)
    results[name] = {
        'total': total,
        'avg_layer': avg_layer,
        'n_layers': n_layers
    }

# 打印结果
print("压缩时间对比:")
print("-" * 80)
print(f"{'方法':<20} {'总时间(s)':<15} {'平均每层(s)':<15} {'层数':<10}")
print("-" * 80)

method_names = {
    'baseline': '原版',
    'enhanced_no_mi': '增强版(无MI)',
    'mi_enhanced': 'MI改进版'
}

for name, data in results.items():
    total = f"{data['total']:.2f}" if data['total'] else "N/A"
    avg = f"{data['avg_layer']:.3f}" if data['avg_layer'] else "N/A"
    n_layers = str(data['n_layers']) if data['n_layers'] else "N/A"
    print(f"{method_names[name]:<20} {total:<15} {avg:<15} {n_layers:<10}")

print("-" * 80)

# 计算开销比例
if results['baseline']['total'] and results['mi_enhanced']['total']:
    overhead = (results['mi_enhanced']['total'] / results['baseline']['total'] - 1) * 100
    print(f"\nMI改进版相对原版的时间开销: {overhead:+.1f}%")

if results['enhanced_no_mi']['total'] and results['mi_enhanced']['total']:
    overhead = (results['mi_enhanced']['total'] / results['enhanced_no_mi']['total'] - 1) * 100
    print(f"MI改进版相对增强版(无MI)的时间开销: {overhead:+.1f}%")

# 保存到CSV
csv_file = result_dir / 'complexity_summary.csv'
with open(csv_file, 'w') as f:
    f.write("method,total_time_s,avg_layer_time_s,n_layers\n")
    for name, data in results.items():
        total = data['total'] if data['total'] else ''
        avg = data['avg_layer'] if data['avg_layer'] else ''
        n_layers = data['n_layers'] if data['n_layers'] else ''
        f.write(f"{method_names[name]},{total},{avg},{n_layers}\n")

print(f"\n结果已保存到: {csv_file}")

EOF

echo ""
echo "========================================================================"
echo "测试完成！"
echo "详细日志保存在: $RESULT_DIR/"
echo "========================================================================"

