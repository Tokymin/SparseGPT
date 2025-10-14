# SparseGPT 改进版本 - 全面测试套件

验证改进版本的有效性、稳定性和计算开销。

> **⚠️ 重要**: 所有测试脚本已配置为仅使用 **GPU 2号和3号** (`CUDA_VISIBLE_DEVICES=2,3`)  
> 如需使用其他 GPU，请修改脚本开头的 `export CUDA_VISIBLE_DEVICES=2,3`

---

## 🚀 快速开始

### 1️⃣ 快速验证（15分钟）- 推荐首先运行

```bash
cd /media/user/data3/toky/Projects/SparseGPT/fixe_enhanced_version
./quick_test.sh
```

这会运行原版和改进版各3次，输出统计分析结果。

### 2️⃣ 完整验证（1-5小时）

```bash
./run_full_validation.sh
```

自动运行统计测试 + 复杂度测试 + 生成完整报告。

### 3️⃣ 查看结果

```bash
# 查看快速测试结果
cat quick_test_results/quick_results.csv

# 查看完整报告（运行完整验证后）
cat analysis_output/ANALYSIS_REPORT.md

# 查看图表
ls analysis_output/*.png
```

---

## 📊 测试脚本说明

| 脚本 | 时间 | 说明 |
|------|------|------|
| `quick_test.sh` | 15分钟 | 快速验证，原版vs改进版各3次 |
| `statistical_test.sh` | 1小时 | 统计显著性测试，各5次，含T-test |
| `benchmark_complexity.sh` | 20分钟 | 计算复杂度对比（时间/内存） |
| `comprehensive_test.sh` | 4小时 | 全面配置测试（多稀疏度/比特数） |
| `run_full_validation.sh` | 1-5小时 | 一键运行上述所有测试 |

**分析工具**: `analyze_results.py` - 自动生成报告和可视化图表

---

## 📈 预期结果

改进版本应达到的目标：

| 指标 | 原版 | 目标 | 判断标准 |
|------|------|------|----------|
| WikiText2 PPL | ~38.86 | < 38.0 | 改进 > 2% |
| 统计显著性 | - | p < 0.05 | T-test |
| 时间开销 | 1.0x | < 1.5x | 可接受 |

---

## 🔍 核心改进

### 多维度重要性评估
```
综合评分 = 0.25×激活 + 0.25×Hessian + 0.15×权重 + 0.25×输出敏感度 + 0.10×稳定性
```

### 自适应比特分配
- **5档**: 2, 3, 4, 6, 8 bits
- **Quantile方法**: 基于分位数均匀分配
- **Budget方法**: 基于预算优化分配

---

## 📂 结果目录

运行测试后自动创建：

```
fixe_enhanced_version/
├── quick_test_results/       # 快速测试
├── statistical_results/      # 统计测试
├── complexity_results/       # 复杂度测试
├── comprehensive_results/    # 全面测试（可选）
└── analysis_output/          # 报告和图表
    ├── ANALYSIS_REPORT.md    # ← 最终报告
    └── *.png                 # ← 可视化图表
```

---

## 🛠️ 使用示例

### 示例1: 快速验证

```bash
./quick_test.sh

# 输出示例：
# WikiText2 PPL:
#   原版:   38.86 ± 0.05 (95% CI: [38.78, 38.94])
#   增强版: 37.45 ± 0.07 (95% CI: [37.35, 37.55])
#   改进:   +3.63% (t=-18.234, p=0.0001)
#          ✓ 统计显著 (p < 0.05)
```

### 示例2: 完整验证

```bash
./run_full_validation.sh
# 选择快速模式 [Y/n]: Y
# ... 运行约1小时 ...

cat analysis_output/ANALYSIS_REPORT.md
```

### 示例3: 单独运行某个测试

```bash
# 只运行统计测试
./statistical_test.sh

# 只运行复杂度测试
./benchmark_complexity.sh

# 手动分析结果
python3 analyze_results.py
```

---

## 🔧 故障排查

### 1. 测试失败

```bash
# 检查环境
/media/user/data3/toky/CondaEnvs/SparseGPT/bin/python --version
nvidia-smi

# 检查依赖
pip list | grep -E "torch|transformers|scipy|pandas|matplotlib"
```

### 2. 结果异常

查看日志文件：
```bash
tail -f quick_test_results/*.log
grep -i error statistical_results/*.log
```

### 3. 内存不足

使用更小的模型或减少运行次数（修改脚本中的 `NUM_RUNS`）

---

## 📋 快速命令参考

```bash
# 快速验证
./quick_test.sh                           # 15分钟
cat quick_test_results/quick_results.csv  # 查看结果

# 完整验证
./run_full_validation.sh                  # 1-5小时
cat analysis_output/ANALYSIS_REPORT.md   # 查看报告

# 单项测试
./statistical_test.sh                     # 统计测试
./benchmark_complexity.sh                 # 复杂度测试
./comprehensive_test.sh                   # 全面测试

# 分析工具
python3 analyze_results.py                # 生成报告
ls analysis_output/*.png                  # 查看图表
```

---

## ✅ 判断标准

### 改进有效 ✓
- WikiText2 PPL 改进 > 2%
- T-test p-value < 0.05
- 时间开销 < 1.5x

### 改进有限 ⚠
- WikiText2 PPL 改进 0-2%
- T-test p-value 0.05-0.10
- 需要优化参数

### 改进无效 ✗
- WikiText2 PPL 未改进
- T-test p-value > 0.10
- 需要重新设计

---

## 💡 使用建议

1. **首次使用**: 先运行 `quick_test.sh`，确认改进有效
2. **论文实验**: 运行 `run_full_validation.sh` 获取完整数据
3. **调试优化**: 单独运行各个测试脚本，逐步验证

---

**作者**: Toky  
**日期**: 2025-10-12
