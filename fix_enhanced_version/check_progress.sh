#!/bin/bash
# 检查测试进度

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "========================================================================"
echo "测试进度检查"
echo "========================================================================"
echo ""

# 检查统计测试
echo "1. 统计显著性测试 (目标: 10次运行)"
echo "------------------------------------------------------------------------"
if [ -d "$SCRIPT_DIR/statistical_results" ]; then
    count=$(ls $SCRIPT_DIR/statistical_results/*.log 2>/dev/null | wc -l)
    echo "  已完成: $count / 10 次运行"
    
    if [ -f "$SCRIPT_DIR/statistical_results/statistical_runs.csv" ]; then
        lines=$(wc -l < $SCRIPT_DIR/statistical_results/statistical_runs.csv)
        echo "  CSV记录: $((lines - 1)) 条"
    fi
    
    # 显示最新日志的最后几行
    latest_log=$(ls -t $SCRIPT_DIR/statistical_results/*.log 2>/dev/null | head -1)
    if [ ! -z "$latest_log" ]; then
        echo "  最新日志: $(basename $latest_log)"
        tail -3 "$latest_log" 2>/dev/null | grep -E "(运行|Perplexity|完成)" | head -2
    fi
else
    echo "  未开始 (目录不存在)"
fi
echo ""

# 检查复杂度测试
echo "2. 计算复杂度测试 (目标: 3个方法)"
echo "------------------------------------------------------------------------"
if [ -d "$SCRIPT_DIR/complexity_results" ]; then
    count=$(ls $SCRIPT_DIR/complexity_results/*.log 2>/dev/null | wc -l)
    echo "  已完成: $count / 3 个方法"
    
    if [ -f "$SCRIPT_DIR/complexity_results/complexity_benchmark.csv" ]; then
        lines=$(wc -l < $SCRIPT_DIR/complexity_results/complexity_benchmark.csv)
        echo "  CSV记录: $((lines - 1)) 条"
    fi
    
    # 显示最新日志
    latest_log=$(ls -t $SCRIPT_DIR/complexity_results/*.log 2>/dev/null | head -1)
    if [ ! -z "$latest_log" ]; then
        echo "  最新日志: $(basename $latest_log)"
    fi
else
    echo "  未开始 (目录不存在)"
fi
echo ""

# 检查进程
echo "3. 运行中的进程"
echo "------------------------------------------------------------------------"
ps aux | grep -E "(statistical_test|benchmark_complexity)" | grep -v grep | head -5
if [ $? -ne 0 ]; then
    echo "  没有测试进程在运行"
fi
echo ""

echo "========================================================================"
echo "使用说明:"
echo "  - 监控统计测试: tail -f statistical_test_output.log"
echo "  - 监控复杂度测试: tail -f complexity_test_output.log"
echo "  - 重新检查进度: ./check_progress.sh"
echo "========================================================================"

