#!/bin/bash
# ========================================================================
# 快速验证测试 - 用于初步验证改进是否有效
# ========================================================================
# 测试内容:
# 1. 原版 vs 增强版 (各运行 3 次)
# 2. 固定配置: sparsity=0.5, bits=4.0
# 3. 计算时间、PPL 和置信区间
# ========================================================================

# 仅使用 2号和3号 GPU
export CUDA_VISIBLE_DEVICES=2,3

PYTHON=/media/user/data3/toky/CondaEnvs/SparseGPT/bin/python
MODEL="facebook/opt-125m"
DATASET="c4"

# 结果目录（使用绝对路径）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULT_DIR="$SCRIPT_DIR/quick_test_results"
mkdir -p $RESULT_DIR

SPARSITY=0.5
BITS=4.0
NUM_RUNS=3

echo "========================================================================"
echo "快速验证测试"
echo "========================================================================"
echo "配置: sparsity=${SPARSITY}, bits=${BITS}"
echo "运行次数: ${NUM_RUNS}"
echo "结果目录: $RESULT_DIR"
echo ""

# CSV 文件
CSV_FILE="$RESULT_DIR/quick_results.csv"
echo "method,run_id,wikitext2_ppl,ptb_ppl,c4_ppl,time_sec" > $CSV_FILE

# ========================================================================
# 测试原版
# ========================================================================
echo "========================================================================"
echo "测试 1/2: 原版 SparseGPT"
echo "========================================================================"

for run_id in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "运行 $run_id / $NUM_RUNS"
    echo "开始时间: $(date)"
    
    log_file="$RESULT_DIR/original_run${run_id}.log"
    
    cd /media/user/data3/toky/Projects/SparseGPT
    
    start=$(date +%s)
    
    $PYTHON opt.py $MODEL $DATASET \
        --sparsity $SPARSITY --wbits 4 \
        2>&1 | tee $log_file
    
    end=$(date +%s)
    elapsed=$((end - start))
    
    # 提取结果
    wt2=$(grep 'wikitext2:' $log_file | tail -1 | awk '{print $NF}')
    ptb=$(grep 'ptb:' $log_file | tail -1 | awk '{print $NF}')
    c4=$(grep 'c4:' $log_file | tail -1 | awk '{print $NF}')
    
    echo "original,$run_id,$wt2,$ptb,$c4,$elapsed" >> $CSV_FILE
    
    echo "结果: WikiText2=$wt2, PTB=$ptb, C4=$c4 | 时间: ${elapsed}s"
done

# ========================================================================
# 测试增强版
# ========================================================================
echo ""
echo "========================================================================"
echo "测试 2/2: 增强版 SparseGPT (Quantile)"
echo "========================================================================"

for run_id in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "运行 $run_id / $NUM_RUNS"
    echo "开始时间: $(date)"
    
    log_file="$RESULT_DIR/enhanced_run${run_id}.log"
    
    cd /media/user/data3/toky/Projects/SparseGPT
    
    start=$(date +%s)
    
    $PYTHON enhanced_version/opt_enhanced.py $MODEL $DATASET \
        --sparsity $SPARSITY --wbits 4 \
        --target_avg_bits $BITS --bit_method quantile \
        2>&1 | tee $log_file
    
    end=$(date +%s)
    elapsed=$((end - start))
    
    # 提取结果
    wt2=$(grep 'wikitext2:' $log_file | tail -1 | awk '{print $NF}')
    ptb=$(grep 'ptb:' $log_file | tail -1 | awk '{print $NF}')
    c4=$(grep 'c4:' $log_file | tail -1 | awk '{print $NF}')
    
    echo "enhanced_quantile,$run_id,$wt2,$ptb,$c4,$elapsed" >> $CSV_FILE
    
    echo "结果: WikiText2=$wt2, PTB=$ptb, C4=$c4 | 时间: ${elapsed}s"
done

# ========================================================================
# 分析结果
# ========================================================================
echo ""
echo "========================================================================"
echo "结果分析"
echo "========================================================================"
echo ""

cd "$SCRIPT_DIR"

$PYTHON << 'EOF'
import pandas as pd
import numpy as np
from scipy import stats
import os

# 读取数据
csv_file = 'quick_test_results/quick_results.csv'
if not os.path.exists(csv_file):
    print(f"错误: 找不到结果文件 {csv_file}")
    exit(1)

df = pd.read_csv(csv_file)

print("="*80)
print("快速验证测试 - 结果汇总")
print("="*80)
print()

# 统计分析
metrics = ['wikitext2_ppl', 'ptb_ppl', 'c4_ppl', 'time_sec']
metric_names = ['WikiText2 PPL', 'PTB PPL', 'C4 PPL', 'Runtime (sec)']

for metric, name in zip(metrics, metric_names):
    print(f"{name}:")
    print("-"*80)
    
    original = df[df['method'] == 'original'][metric].dropna()
    enhanced = df[df['method'] == 'enhanced_quantile'][metric].dropna()
    
    if len(original) > 0:
        orig_mean = original.mean()
        orig_std = original.std()
        orig_se = stats.sem(original)
        orig_ci = stats.t.interval(0.95, len(original)-1, loc=orig_mean, scale=orig_se)
        
        print(f"  原版:   {orig_mean:.3f} ± {orig_std:.3f} "
              f"(95% CI: [{orig_ci[0]:.3f}, {orig_ci[1]:.3f}])")
    
    if len(enhanced) > 0:
        enh_mean = enhanced.mean()
        enh_std = enhanced.std()
        enh_se = stats.sem(enhanced)
        enh_ci = stats.t.interval(0.95, len(enhanced)-1, loc=enh_mean, scale=enh_se)
        
        print(f"  增强版: {enh_mean:.3f} ± {enh_std:.3f} "
              f"(95% CI: [{enh_ci[0]:.3f}, {enh_ci[1]:.3f}])")
    
    # 改进幅度
    if len(original) > 0 and len(enhanced) > 0:
        if metric.endswith('ppl'):
            improvement = (orig_mean - enh_mean) / orig_mean * 100
            print(f"  改进:   {improvement:+.2f}% ", end="")
        else:
            overhead = (enh_mean - orig_mean) / orig_mean * 100
            print(f"  开销:   {overhead:+.2f}% ", end="")
        
        # T-test
        if len(original) > 1 and len(enhanced) > 1:
            t_stat, p_val = stats.ttest_ind(original, enhanced)
            print(f"(t={t_stat:.3f}, p={p_val:.4f})")
            
            if p_val < 0.05:
                print(f"         ✓ 统计显著 (p < 0.05)")
            else:
                print(f"         - 不显著 (p ≥ 0.05)")
        else:
            print()
    
    print()

# 总结
print("="*80)
print("结论")
print("="*80)
print()

original_wt2 = df[df['method'] == 'original']['wikitext2_ppl'].mean()
enhanced_wt2 = df[df['method'] == 'enhanced_quantile']['wikitext2_ppl'].mean()

if pd.notna(original_wt2) and pd.notna(enhanced_wt2):
    improvement = (original_wt2 - enhanced_wt2) / original_wt2 * 100
    
    print(f"WikiText2 PPL: {original_wt2:.3f} → {enhanced_wt2:.3f} ({improvement:+.2f}%)")
    
    if improvement > 2:
        print("✓ 增强版性能明显优于原版")
        print()
        print("建议: 继续进行全面测试以验证稳定性")
    elif improvement > 0:
        print("≈ 增强版性能略优于原版")
        print()
        print("建议: 进行更多次测试以确认改进的稳定性")
    else:
        print("✗ 增强版性能未能超越原版")
        print()
        print("建议: 检查实现或调整超参数")

print()
print("="*80)
EOF

echo ""
echo "快速测试完成！"
echo "结果保存在: $RESULT_DIR/"
echo ""

