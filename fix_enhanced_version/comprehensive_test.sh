#!/bin/bash
# ========================================================================
# 全面对比测试脚本 - 验证改进版本的有效性
# ========================================================================
# 测试维度:
# 1. 多配置测试: 不同稀疏度 (0.3, 0.5, 0.7) × 不同比特数 (3, 4, 8)
# 2. 多数据集测试: C4, WikiText2, PTB
# 3. 计算复杂度分析: 时间、内存、FLOPs
# 4. 统计显著性: 每个配置运行3次
# ========================================================================

# 仅使用 2号和3号 GPU
export CUDA_VISIBLE_DEVICES=2,3

PYTHON=/media/user/data3/toky/CondaEnvs/SparseGPT/bin/python
MODEL="facebook/opt-125m"
DATASET="c4"

# 创建结果目录（使用绝对路径）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULT_DIR="$SCRIPT_DIR/comprehensive_results"
mkdir -p $RESULT_DIR

# 日志文件
SUMMARY_LOG="$RESULT_DIR/summary.txt"
echo "========================================================================" > $SUMMARY_LOG
echo "全面对比测试 - $(date)" >> $SUMMARY_LOG
echo "========================================================================" >> $SUMMARY_LOG
echo "" >> $SUMMARY_LOG

# ========================================================================
# 测试配置矩阵
# ========================================================================
SPARSITIES=(0.0 0.3 0.5 0.7)
TARGET_BITS=(3.0 4.0 8.0)
METHODS=("original" "enhanced_quantile" "enhanced_budget")

echo "测试配置:" | tee -a $SUMMARY_LOG
echo "  稀疏度: ${SPARSITIES[@]}" | tee -a $SUMMARY_LOG
echo "  目标比特数: ${TARGET_BITS[@]}" | tee -a $SUMMARY_LOG
echo "  方法: ${METHODS[@]}" | tee -a $SUMMARY_LOG
echo "" | tee -a $SUMMARY_LOG

# ========================================================================
# 函数: 运行单次测试
# ========================================================================
run_test() {
    local method=$1
    local sparsity=$2
    local bits=$3
    local run_id=$4
    
    local test_name="${method}_sp${sparsity}_bits${bits}_run${run_id}"
    local log_file="$RESULT_DIR/${test_name}.log"
    
    echo "========================================================================"
    echo "测试: $test_name"
    echo "开始时间: $(date)"
    echo "========================================================================"
    
    cd /media/user/data3/toky/Projects/SparseGPT
    
    if [ "$method" == "original" ]; then
        # 原版 SparseGPT
        echo "运行原版 SparseGPT (固定${bits}bit量化)..." | tee -a $log_file
        
        # 将 TARGET_BITS 转换为整数 wbits
        local wbits=$(echo $bits | awk '{print int($1)}')
        
        # 原版不支持动态比特分配，使用固定量化
        $PYTHON opt.py $MODEL $DATASET \
            --sparsity $sparsity --wbits $wbits \
            2>&1 | tee -a $log_file
            
    elif [ "$method" == "enhanced_quantile" ]; then
        # 增强版 - Quantile 方法
        echo "运行增强版 SparseGPT (Quantile方法, 目标${bits}bit)..." | tee -a $log_file
        $PYTHON enhanced_version/opt_enhanced.py $MODEL $DATASET \
            --sparsity $sparsity --wbits 4 \
            --target_avg_bits $bits --bit_method quantile \
            2>&1 | tee -a $log_file
            
    elif [ "$method" == "enhanced_budget" ]; then
        # 增强版 - Budget 方法
        echo "运行增强版 SparseGPT (Budget方法, 目标${bits}bit)..." | tee -a $log_file
        $PYTHON enhanced_version/opt_enhanced.py $MODEL $DATASET \
            --sparsity $sparsity --wbits 4 \
            --target_avg_bits $bits --bit_method budget \
            2>&1 | tee -a $log_file
    fi
    
    echo "完成时间: $(date)" | tee -a $log_file
    echo "" | tee -a $log_file
}

# ========================================================================
# 函数: 提取结果指标
# ========================================================================
extract_metrics() {
    local log_file=$1
    local output_file=$2
    
    # 提取 PPL 指标
    local wikitext2=$(grep 'Perplexity on wikitext2:' $log_file | tail -1 | awk '{print $NF}')
    local ptb=$(grep 'Perplexity on ptb:' $log_file | tail -1 | awk '{print $NF}')
    local c4=$(grep 'Perplexity on c4:' $log_file | tail -1 | awk '{print $NF}')
    
    # 提取时间和内存指标（如果有）
    local total_time=$(grep '总时间:' $log_file | tail -1 | awk '{print $(NF-1)}')
    local peak_memory=$(grep 'Peak memory:' $log_file | tail -1 | awk '{print $(NF-1)}')
    
    # 提取比特分布（仅增强版有）
    local avg_bits=$(grep '平均比特数:' $log_file | tail -1 | awk '{print $NF}')
    
    # 写入结果
    echo "$wikitext2,$ptb,$c4,$total_time,$peak_memory,$avg_bits" >> $output_file
}

# ========================================================================
# 主测试循环
# ========================================================================
echo "========================================================================" | tee -a $SUMMARY_LOG
echo "开始批量测试..." | tee -a $SUMMARY_LOG
echo "========================================================================" | tee -a $SUMMARY_LOG
echo "" | tee -a $SUMMARY_LOG

# CSV 结果文件
RESULTS_CSV="$RESULT_DIR/results.csv"
echo "method,sparsity,target_bits,run_id,wikitext2_ppl,ptb_ppl,c4_ppl,time_sec,peak_mem_gb,avg_bits" > $RESULTS_CSV

# 测试计数
total_tests=$((${#METHODS[@]} * ${#SPARSITIES[@]} * ${#TARGET_BITS[@]} * 3))
current_test=0

for method in "${METHODS[@]}"; do
    for sparsity in "${SPARSITIES[@]}"; do
        for bits in "${TARGET_BITS[@]}"; do
            # 每个配置运行3次（统计显著性）
            for run_id in 1 2 3; do
                current_test=$((current_test + 1))
                
                echo "" | tee -a $SUMMARY_LOG
                echo "进度: $current_test / $total_tests" | tee -a $SUMMARY_LOG
                
                # 运行测试
                run_test $method $sparsity $bits $run_id
                
                # 提取结果
                test_name="${method}_sp${sparsity}_bits${bits}_run${run_id}"
                log_file="$RESULT_DIR/${test_name}.log"
                
                if [ -f "$log_file" ]; then
                    # 提取指标并写入CSV
                    wikitext2=$(grep 'wikitext2:' $log_file | tail -1 | awk '{print $NF}')
                    ptb=$(grep 'ptb:' $log_file | tail -1 | awk '{print $NF}')
                    c4=$(grep 'c4:' $log_file | tail -1 | awk '{print $NF}')
                    
                    echo "$method,$sparsity,$bits,$run_id,$wikitext2,$ptb,$c4,,,," >> $RESULTS_CSV
                    
                    echo "结果: WikiText2=$wikitext2, PTB=$ptb, C4=$c4" | tee -a $SUMMARY_LOG
                fi
                
                echo "" | tee -a $SUMMARY_LOG
            done
        done
    done
done

echo "========================================================================" | tee -a $SUMMARY_LOG
echo "所有测试完成！" | tee -a $SUMMARY_LOG
echo "结果保存在: $RESULT_DIR/" | tee -a $SUMMARY_LOG
echo "========================================================================" | tee -a $SUMMARY_LOG

# 生成快速预览
echo "" | tee -a $SUMMARY_LOG
echo "快速预览 (均值, 3次运行):" | tee -a $SUMMARY_LOG
echo "----------------------------------------" | tee -a $SUMMARY_LOG

cd "$SCRIPT_DIR"

$PYTHON << 'EOF'
import pandas as pd
import numpy as np
import os

try:
    csv_file = 'comprehensive_results/results.csv'
    if not os.path.exists(csv_file):
        print(f"错误: 找不到结果文件 {csv_file}")
        exit(1)
    df = pd.read_csv(csv_file)
    
    # 按配置分组，计算均值和标准差
    grouped = df.groupby(['method', 'sparsity', 'target_bits'])
    
    print("\n对比表格 (WikiText2 PPL):")
    print("=" * 80)
    
    for name, group in grouped:
        method, sparsity, bits = name
        wt2_mean = group['wikitext2_ppl'].mean()
        wt2_std = group['wikitext2_ppl'].std()
        
        print(f"{method:20s} | sp={sparsity:.1f} | bits={bits:.1f} | "
              f"WikiText2: {wt2_mean:.2f} ± {wt2_std:.2f}")
    
    print("=" * 80)
except Exception as e:
    print(f"无法生成预览: {e}")
EOF

echo "" | tee -a $SUMMARY_LOG
echo "运行 analyze_results.py 生成详细报告和可视化" | tee -a $SUMMARY_LOG

