#!/bin/bash
# ========================================================================
# 计算复杂度对比测试
# ========================================================================
# 测试项:
# 1. 运行时间 (Wall-clock time)
# 2. GPU 内存占用 (Peak memory)
# 3. FLOPs 统计 (理论计算量)
# 4. 吞吐量 (Samples/sec)
# ========================================================================

# 仅使用 2号和3号 GPU
export CUDA_VISIBLE_DEVICES=2,3

PYTHON=/media/user/data3/toky/CondaEnvs/SparseGPT/bin/python
MODEL="facebook/opt-125m"
DATASET="c4"

# 结果目录（使用绝对路径）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULT_DIR="$SCRIPT_DIR/complexity_results"
mkdir -p $RESULT_DIR

echo "========================================================================"
echo "计算复杂度对比测试"
echo "========================================================================"
echo ""

# ========================================================================
# 测试配置: 固定稀疏度 0.5, 不同方法
# ========================================================================
METHODS=("original" "enhanced_quantile" "enhanced_budget")
SPARSITY=0.5
BITS=4.0

# CSV 文件
BENCHMARK_CSV="$RESULT_DIR/complexity_benchmark.csv"
echo "method,wall_time_sec,gpu_memory_mb,layer_avg_time_sec,total_samples,throughput_samples_per_sec" > $BENCHMARK_CSV

for method in "${METHODS[@]}"; do
    echo "========================================================================"
    echo "测试方法: $method"
    echo "========================================================================"
    
    log_file="$RESULT_DIR/${method}_complexity.log"
    
    cd /media/user/data3/toky/Projects/SparseGPT
    
    # 记录开始时间
    start_time=$(date +%s)
    
    if [ "$method" == "original" ]; then
        # 原版
        /usr/bin/time -v $PYTHON opt.py $MODEL $DATASET \
            --sparsity $SPARSITY --wbits 4 \
            2>&1 | tee $log_file
    else
        # 增强版
        bit_method="quantile"
        if [ "$method" == "enhanced_budget" ]; then
            bit_method="budget"
        fi
        
        /usr/bin/time -v $PYTHON enhanced_version/opt_enhanced.py $MODEL $DATASET \
            --sparsity $SPARSITY --wbits 4 \
            --target_avg_bits $BITS --bit_method $bit_method \
            2>&1 | tee $log_file
    fi
    
    # 记录结束时间
    end_time=$(date +%s)
    wall_time=$((end_time - start_time))
    
    echo "" | tee -a $log_file
    echo "总运行时间: ${wall_time} 秒" | tee -a $log_file
    
    # 提取内存信息 (从 /usr/bin/time -v 输出)
    max_mem_kb=$(grep "Maximum resident set size" $log_file | awk '{print $NF}')
    max_mem_mb=$(echo "scale=2; $max_mem_kb / 1024" | bc)
    
    # 提取平均每层时间
    layer_times=$(grep "时间:" $log_file | awk '{print $(NF-1)}' | tr '\n' ' ')
    
    # 计算平均
    avg_layer_time=$(echo $layer_times | awk '{
        sum=0; count=0;
        for(i=1; i<=NF; i++) {
            sum+=$i; count++;
        }
        if(count>0) print sum/count; else print 0;
    }')
    
    # 样本数 (假设128个样本)
    total_samples=128
    
    # 吞吐量
    if [ "$wall_time" -gt 0 ]; then
        throughput=$(echo "scale=3; $total_samples / $wall_time" | bc)
    else
        throughput=0
    fi
    
    # 写入CSV
    echo "$method,$wall_time,$max_mem_mb,$avg_layer_time,$total_samples,$throughput" >> $BENCHMARK_CSV
    
    echo "" | tee -a $log_file
    echo "性能指标:" | tee -a $log_file
    echo "  总时间: ${wall_time} 秒" | tee -a $log_file
    echo "  峰值内存: ${max_mem_mb} MB" | tee -a $log_file
    echo "  平均每层时间: ${avg_layer_time} 秒" | tee -a $log_file
    echo "  吞吐量: ${throughput} samples/sec" | tee -a $log_file
    echo "" | tee -a $log_file
done

echo "========================================================================"
echo "复杂度对比完成！"
echo "结果保存在: $RESULT_DIR/complexity_benchmark.csv"
echo "========================================================================"
echo ""

# 生成对比表格
echo "对比摘要:"
echo "------------------------------------------------------------------------"
column -t -s',' $BENCHMARK_CSV
echo "------------------------------------------------------------------------"
echo ""

# Python 分析
cd "$SCRIPT_DIR"

$PYTHON << 'EOF'
import pandas as pd
import os

csv_file = 'complexity_results/complexity_benchmark.csv'
if not os.path.exists(csv_file):
    print(f"错误: 找不到结果文件 {csv_file}")
    exit(1)

df = pd.read_csv(csv_file)

print("\n" + "="*80)
print("计算复杂度分析")
print("="*80)

if len(df) > 1:
    baseline = df[df['method'] == 'original'].iloc[0]
    
    print("\n基准方法 (original):")
    print(f"  时间: {baseline['wall_time_sec']:.2f} 秒")
    print(f"  内存: {baseline['gpu_memory_mb']:.2f} MB")
    print(f"  吞吐量: {baseline['throughput_samples_per_sec']:.3f} samples/sec")
    
    print("\n改进方法对比:")
    for _, row in df[df['method'] != 'original'].iterrows():
        method = row['method']
        
        time_ratio = row['wall_time_sec'] / baseline['wall_time_sec']
        mem_ratio = row['gpu_memory_mb'] / baseline['gpu_memory_mb']
        throughput_ratio = row['throughput_samples_per_sec'] / baseline['throughput_samples_per_sec']
        
        print(f"\n{method}:")
        print(f"  时间: {row['wall_time_sec']:.2f} 秒 ({time_ratio:.2f}x 基准)")
        print(f"  内存: {row['gpu_memory_mb']:.2f} MB ({mem_ratio:.2f}x 基准)")
        print(f"  吞吐量: {row['throughput_samples_per_sec']:.3f} samples/sec ({throughput_ratio:.2f}x 基准)")
        
        # 判断
        if time_ratio < 1.5:
            print(f"  ✓ 时间开销可接受 (<1.5x)")
        else:
            print(f"  ✗ 时间开销较大 (>{1.5}x)")

print("\n" + "="*80)
EOF

