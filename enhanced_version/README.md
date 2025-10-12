# SparseGPT 增强版 - 改进2：激活感知的量化精度分配

## 📁 文件夹说明

这个文件夹包含了**改进2**的所有相关文件，与原始项目分离，便于管理和使用。

---

## 🐛 重要修复说明

### ❌ 之前的测试问题

在第一次运行 `run_comparison_test.sh` 时，**忘记添加 `--wbits 4` 参数**，导致：
- 量化功能没有启用
- 统计信息为空（总通道数: 0）
- PPL 结果与原版完全相同
- **增强版的混合精度量化根本没有生效！**

### ✅ 修复方法

使用 `run_comparison_test_fixed.sh`，它正确添加了 `--wbits 4` 参数：

```bash
python opt_enhanced.py facebook/opt-125m c4 \
    --sparsity 0.5 \
    --wbits 4 \              # ← 关键！启用4-bit量化
    --target_avg_bits 4.0 \
    --bit_method quantile
```

---

## 📂 文件清单

### 🔥 核心文件

| 文件 | 说明 | 行数 |
|------|------|-----|
| **sparsegpt_enhanced.py** | 增强版核心实现 | 395 |
| **opt_enhanced.py** | OPT模型测试脚本 | 450+ |

### 🧪 测试文件

| 文件 | 说明 |
|------|------|
| `test_enhanced.py` | 基础功能测试 |
| `example_usage.py` | 5个使用示例 |
| `compare_versions.py` | 版本对比工具 |
| ~~`run_comparison_test.sh`~~ | ❌ 旧版（缺少--wbits） |
| **`run_comparison_test_fixed.sh`** | ✅ **修复版（推荐）** |

### 📚 文档文件

| 文件 | 说明 |
|------|------|
| `README.md` | 本文档 |
| `README_enhanced.md` | 详细使用文档 |
| `IMPROVEMENTS_SUMMARY.md` | 改进详解 |
| `QUICKSTART_ENHANCED.md` | 快速入门 |
| `EVALUATION_GUIDE.md` | 评估指南 |

---

## 🚀 快速开始（正确方法）

### 1. 基础功能测试
```bash
cd /media/user/data3/toky/Projects/SparseGPT/enhanced_version
python test_enhanced.py
```

### 2. 在 OPT-125M 上测试（单次）
```bash
python opt_enhanced.py facebook/opt-125m c4 \
    --sparsity 0.5 \
    --wbits 4 \              # 必须！
    --target_avg_bits 4.0 \
    --bit_method quantile
```

### 3. 完整对比测试（修复版）
```bash
chmod +x run_comparison_test_fixed.sh
bash run_comparison_test_fixed.sh
```

---

## 📊 预期输出

### ✅ 正确输出（加了 --wbits 4）

```
============================================================
量化统计摘要 (Quantization Statistics Summary)
============================================================

总通道数: 9216

比特分布:
  2-bit:   1843 通道 (20.00%)
  3-bit:   1843 通道 (20.00%)
  4-bit:   1843 通道 (20.00%)
  6-bit:   1843 通道 (20.00%)
  8-bit:   1844 通道 (20.01%)

平均比特数: 4.600 bits

每层统计:
  layer0.self_attn.k_proj: avg=4.60 bits, importance_range=(0.979, 1.021)
  layer0.self_attn.v_proj: avg=4.60 bits, importance_range=(0.984, 1.015)
  ...
============================================================
```

### ❌ 错误输出（没加 --wbits）

```
============================================================
量化统计摘要 (Quantization Statistics Summary)
============================================================

总通道数: 0        ← 问题！

比特分布:          ← 空的！

每层统计:
============================================================
```

---

## 🎯 核心改进点

### 相比 `sparsegpt_toky.py`

| 方面 | sparsegpt_toky.py | **sparsegpt_enhanced.py** |
|------|-------------------|---------------------------|
| 评估维度 | 2个 | **5个** |
| 量化档数 | 3档 (2/4/8) | **5档 (2/3/4/6/8)** |
| 阈值方式 | 固定 | **自适应分位数** |
| 比特控制 | 无 | **精确预算** |
| 统计报告 | 无 | **详细可视化** |

---

## 📈 如何验证改进效果

### 1. 检查统计信息
运行后查看日志，确保：
- ✅ 总通道数 > 0
- ✅ 比特分布有5档
- ✅ 平均比特数接近目标值

### 2. 对比 PPL
```bash
# 原版
python ../opt_toky.py facebook/opt-125m c4 --sparsity 0.5 --wbits 4
# WikiText2: ~36.096

# 增强版
python opt_enhanced.py facebook/opt-125m c4 --sparsity 0.5 --wbits 4 \
    --target_avg_bits 4.0 --bit_method quantile
# 预期: WikiText2 下降 0.5-1.5 点
```

### 3. 分析比特分布
- 查看哪些层使用了更高比特
- 验证重要层是否被保护
- 确认分布是否合理

---

## 🔧 常见问题

### Q1: 为什么统计信息为空？
**A**: 忘记加 `--wbits 4` 参数！量化器没有被初始化。

### Q2: 如何调整平均比特数？
**A**: 使用 `--target_avg_bits` 参数：
```bash
--target_avg_bits 3.5  # 更小的模型
--target_avg_bits 4.5  # 更高的精度
```

### Q3: quantile 和 budget 方法有什么区别？
**A**:
- `quantile`: 快速，均衡分布（推荐）
- `budget`: 精确控制平均比特数

### Q4: PPL 没有改善怎么办？
**A**: 
1. 检查统计信息是否正常
2. 尝试调整重要性权重
3. 尝试不同的 target_avg_bits
4. 查看每层的比特分配是否合理

---

## 📚 相关文档

- **快速入门**: `QUICKSTART_ENHANCED.md`
- **详细文档**: `README_enhanced.md`
- **改进说明**: `IMPROVEMENTS_SUMMARY.md`
- **评估指南**: `EVALUATION_GUIDE.md`

---

## 🎓 下一步

### 如果效果好 ✅
1. 在更大模型上验证（OPT-1.3B, 6.7B）
2. 实现改进3（动态通道缩放）
3. 实现改进1（互信息分组）
4. 发表论文

### 如果效果一般 ⚠️
1. 调整重要性权重
2. 尝试不同比特选项
3. 分析哪些层改善/变差
4. 消融实验

### 测试命令
```bash
# 修复版完整测试
bash run_comparison_test_fixed.sh

# 快速验证
python opt_enhanced.py facebook/opt-125m c4 \
    --sparsity 0.5 --wbits 4 \
    --target_avg_bits 4.0 --bit_method quantile
```

---

**重要**: 请使用 `run_comparison_test_fixed.sh` 而不是旧版的 `run_comparison_test.sh`！

**作者**: Toky  
**版本**: Enhanced v1.0 (Fixed)  
**日期**: 2025-10-12

