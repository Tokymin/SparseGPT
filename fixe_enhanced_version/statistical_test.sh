#!/bin/bash
# ========================================================================
# 统计显著性测试
# ========================================================================
# 目标: 验证改进不是偶然的
# 方法:
# 1. 固定配置下多次运行 (N=5)
# 2. 计算均值和标准差
# 3. 进行 t-test 显著性检验
# 4. 计算置信区间
# ========================================================================

# 仅使用 2号和3号 GPU
export CUDA_VISIBLE_DEVICES=2,3

PYTHON=/media/user/data3/toky/CondaEnvs/SparseGPT/bin/python
MODEL="facebook/opt-125m"
DATASET="c4"

# 结果目录（使用绝对路径）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULT_DIR="$SCRIPT_DIR/statistical_results"
mkdir -p $RESULT_DIR

echo "========================================================================"
echo "统计显著性测试 - 验证改进的稳定性"
echo "========================================================================"
echo ""

# ========================================================================
# 测试配置: 固定最佳配置，多次运行
# ========================================================================
SPARSITY=0.5
BITS=4.0
NUM_RUNS=5  # 每个方法运行5次

# CSV 文件
STATS_CSV="$RESULT_DIR/statistical_runs.csv"
echo "method,run_id,wikitext2_ppl,ptb_ppl,c4_ppl,time_sec" > $STATS_CSV

# ========================================================================
# 函数: 运行单次测试并记录
# ========================================================================
run_single_test() {
    local method=$1
    local run_id=$2
    
    local log_file="$RESULT_DIR/${method}_run${run_id}.log"
    
    echo "----------------------------------------"
    echo "运行: $method (第 $run_id 次)"
    echo "开始时间: $(date)"
    echo "----------------------------------------"
    
    cd /media/user/data3/toky/Projects/SparseGPT
    
    start_time=$(date +%s.%N)
    
    if [ "$method" == "original" ]; then
        $PYTHON opt.py $MODEL $DATASET \
            --sparsity $SPARSITY --wbits 4 \
            2>&1 | tee $log_file
    else
        # enhanced_quantile
        $PYTHON enhanced_version/opt_enhanced.py $MODEL $DATASET \
            --sparsity $SPARSITY --wbits 4 \
            --target_avg_bits $BITS --bit_method quantile \
            2>&1 | tee $log_file
    fi
    
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)
    
    # 提取结果
    wikitext2=$(grep 'wikitext2:' $log_file | tail -1 | awk '{print $NF}')
    ptb=$(grep 'ptb:' $log_file | tail -1 | awk '{print $NF}')
    c4=$(grep 'c4:' $log_file | tail -1 | awk '{print $NF}')
    
    # 写入CSV
    echo "$method,$run_id,$wikitext2,$ptb,$c4,$elapsed" >> $STATS_CSV
    
    echo "结果: WikiText2=$wikitext2, PTB=$ptb, C4=$c4"
    echo "用时: ${elapsed} 秒"
    echo ""
}

# ========================================================================
# 运行多次测试
# ========================================================================
METHODS=("original" "enhanced_quantile")

for method in "${METHODS[@]}"; do
    echo "========================================================================"
    echo "测试方法: $method (运行 $NUM_RUNS 次)"
    echo "========================================================================"
    echo ""
    
    for run_id in $(seq 1 $NUM_RUNS); do
        run_single_test $method $run_id
    done
    
    echo ""
done

echo "========================================================================"
echo "所有运行完成！开始统计分析..."
echo "========================================================================"
echo ""

# ========================================================================
# Python 统计分析
# ========================================================================
cd "$SCRIPT_DIR"

$PYTHON << 'EOF'
import pandas as pd
import numpy as np
from scipy import stats
import os
from pathlib import Path

# 使用绝对路径
script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
csv_file = script_dir / 'statistical_results' / 'statistical_runs.csv'
# 兼容处理：如果在子shell中运行，使用相对路径
if not csv_file.exists():
    csv_file = Path('statistical_results/statistical_runs.csv')
csv_file = str(csv_file)
if not os.path.exists(csv_file):
    print("结果文件不存在")
    exit(1)

df = pd.read_csv(csv_file)

print("\n" + "="*80)
print("统计显著性分析")
print("="*80)

metrics = ['wikitext2_ppl', 'ptb_ppl', 'c4_ppl', 'time_sec']
metric_names = ['WikiText2 PPL', 'PTB PPL', 'C4 PPL', 'Time (sec)']

for metric, metric_name in zip(metrics, metric_names):
    print(f"\n{metric_name}:")
    print("-" * 80)
    
    # 分组统计
    for method in df['method'].unique():
        data = df[df['method'] == method][metric].dropna()
        
        if len(data) > 0:
            mean = data.mean()
            std = data.std()
            stderr = stats.sem(data)  # 标准误差
            ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=stderr)  # 95% 置信区间
            
            print(f"  {method:20s}: {mean:8.3f} ± {std:6.3f} "
                  f"(95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])")
    
    # T-test (如果有两个方法)
    methods = df['method'].unique()
    if len(methods) == 2:
        data1 = df[df['method'] == methods[0]][metric].dropna()
        data2 = df[df['method'] == methods[1]][metric].dropna()
        
        if len(data1) > 1 and len(data2) > 1:
            t_stat, p_value = stats.ttest_ind(data1, data2)
            
            print(f"\n  T-test: t={t_stat:.3f}, p-value={p_value:.4f}")
            
            if p_value < 0.05:
                if metric.endswith('ppl'):
                    # PPL 越小越好
                    better = methods[0] if data1.mean() < data2.mean() else methods[1]
                    improvement = abs(data1.mean() - data2.mean()) / max(data1.mean(), data2.mean()) * 100
                else:
                    # Time 暂不判断
                    better = "N/A"
                    improvement = 0
                
                if better != "N/A":
                    print(f"  ✓ 显著差异！{better} 显著更优 (改进 {improvement:.2f}%)")
            else:
                print(f"  ✗ 无显著差异 (p > 0.05)")

# 生成摘要
print("\n" + "="*80)
print("结论")
print("="*80)

original_wt2 = df[df['method'] == 'original']['wikitext2_ppl'].mean()
enhanced_wt2 = df[df['method'] == 'enhanced_quantile']['wikitext2_ppl'].mean()

if pd.notna(original_wt2) and pd.notna(enhanced_wt2):
    improvement = (original_wt2 - enhanced_wt2) / original_wt2 * 100
    
    print(f"\nWikiText2 PPL:")
    print(f"  原版:   {original_wt2:.3f}")
    print(f"  增强版: {enhanced_wt2:.3f}")
    print(f"  改进:   {improvement:.2f}%")
    
    if improvement > 0:
        print("\n✓ 增强版性能优于原版")
    elif improvement < -5:
        print("\n✗ 增强版性能劣于原版")
    else:
        print("\n≈ 增强版性能相当")

print("\n" + "="*80)
EOF

echo ""
echo "统计分析完成！"
echo "详细结果保存在: $RESULT_DIR/"

