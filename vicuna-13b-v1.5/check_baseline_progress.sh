#!/bin/bash

echo "=== 基准测试进度检查 ==="
echo ""

# 检查进程
echo "运行中的进程:"
ps aux | grep -E "test_baseline" | grep -v grep | head -2
echo ""

# 检查GPU使用
echo "GPU使用情况 (GPU 2,3):"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | grep -E "^2,|^3,"
echo ""

# 检查日志大小
echo "日志文件大小:"
ls -lh baseline_*.log 2>/dev/null || echo "  暂无日志文件"
echo ""

# 估计剩余时间
echo "预计总耗时: ~5-10分钟 (3个数据集的评估)"
echo ""

echo "提示: 使用 'tail -f baseline_test.log' 查看实时输出"
echo "      或等待完成后运行 'python compare_results.py' 查看对比结果"

