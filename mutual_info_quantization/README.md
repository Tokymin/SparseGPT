# 基于互信息的量化分组改进方案

## 📋 改进思路

### 现有方案的局限性

**当前增强版（enhanced_fix_acc_version）**：
- 按**通道独立**计算重要性
- 每个通道独立分配比特数
- 未考虑通道间的相关性和冗余

### 互信息改进方案

**核心思想**：利用互信息（Mutual Information）量化通道间的依赖关系，将相似/冗余的通道分组量化。

#### 理论基础

1. **互信息定义**：
   ```
   I(X; Y) = H(X) - H(X|Y)
   ```
   - 衡量两个随机变量的相关性
   - 越高表示越相关/冗余

2. **通道分组策略**：
   - 高互信息通道 → 分为一组（冗余，可共享低比特）
   - 低互信息通道 → 独立量化（重要，需要高比特）

3. **优势**：
   - 减少冗余：相关通道共享量化参数
   - 提高效率：在相同比特预算下提升性能
   - 保持结构：考虑通道间的依赖关系

---

## 🎯 技术方案

### 1. 互信息计算

基于激活的互信息估计：

```python
def compute_mutual_information(activations_i, activations_j):
    """
    计算两个通道激活的互信息
    使用 KDE 或直方图方法估计
    """
    # 方法1: 基于熵的估计
    H_i = entropy(activations_i)
    H_j = entropy(activations_j)
    H_ij = joint_entropy(activations_i, activations_j)
    MI = H_i + H_j - H_ij
    
    # 方法2: 归一化互信息 (NMI)
    NMI = 2 * MI / (H_i + H_j)
    
    return MI, NMI
```

### 2. 通道分组算法

```python
# 步骤1: 计算互信息矩阵
MI_matrix = compute_pairwise_MI(all_activations)  # [N_channels, N_channels]

# 步骤2: 聚类分组
from sklearn.cluster import SpectralClustering
clustering = SpectralClustering(
    n_clusters=K,  # 分组数
    affinity='precomputed'
)
groups = clustering.fit_predict(MI_matrix)

# 步骤3: 为每组分配比特数
for group_id in range(K):
    group_channels = np.where(groups == group_id)[0]
    group_importance = compute_group_importance(group_channels)
    group_bits = allocate_bits(group_importance)
```

### 3. 分组量化策略

#### 策略A：组内共享量化参数
```python
for group_id, channels in enumerate(channel_groups):
    # 所有通道共享 scale 和 zero_point
    group_activations = activations[:, channels]
    scale, zero = compute_quantization_params(group_activations)
    
    for ch in channels:
        quantized[ch] = quantize(weights[ch], scale, zero, bits[group_id])
```

#### 策略B：组内自适应量化
```python
for group_id, channels in enumerate(channel_groups):
    # 组内通道共享比特数，但独立 scale/zero
    group_bits = bits[group_id]
    
    for ch in channels:
        scale, zero = compute_quantization_params(weights[ch])
        quantized[ch] = quantize(weights[ch], scale, zero, group_bits)
```

---

## 📊 预期效果

### 优势分析

1. **减少冗余**：
   - 当前版本：1000通道 × 独立量化 = 高冗余
   - 互信息版：50组 × 20通道/组 = 低冗余

2. **提升性能**：
   - 相关通道共享低比特 → 节省比特预算
   - 独立通道使用高比特 → 保持关键信息

3. **理论支撑**：
   - 信息论基础：最小化互信息损失
   - 压缩理论：去除冗余，保留熵

### 性能预测

基于当前结果（sp=0.5, 4bit）：
- 当前增强版：PPL = 36.186
- 互信息改进预期：PPL < 35.0 (**再提升 3-5%**)

---

## 🛠️ 实施计划

### Phase 1: 基础实现
- [ ] 实现互信息计算模块
- [ ] 实现通道分组算法（谱聚类/层次聚类）
- [ ] 集成到 SparseGPT 流程

### Phase 2: 优化策略
- [ ] 测试不同分组数 K
- [ ] 对比共享参数 vs 独立参数
- [ ] 调优比特分配策略

### Phase 3: 全面测试
- [ ] 多数据集验证
- [ ] 不同稀疏度测试
- [ ] 计算开销评估

---

## 📁 文件结构

```
mutual_info_quantization/
├── README.md                    # 本文档
├── sparsegpt_mi.py             # 互信息量化实现
├── mutual_info.py              # 互信息计算工具
├── channel_grouping.py         # 通道分组算法
├── opt_mi.py                   # OPT 模型测试脚本
├── test_mi.sh                  # 快速测试脚本
├── compare_methods.sh          # 对比测试脚本
└── analysis/                   # 分析结果
    ├── mi_matrix.png          # 互信息矩阵可视化
    ├── grouping_result.png    # 分组结果
    └── performance_report.md  # 性能报告
```

---

## 🔬 研究价值

### 学术贡献

1. **创新性**：
   - 首次将互信息引入 SparseGPT 量化
   - 结合剪枝和信息论的新范式

2. **实用性**：
   - 在保持计算开销的前提下提升性能
   - 适用于资源受限的边缘设备

3. **可扩展性**：
   - 可推广到其他压缩方法
   - 与知识蒸馏等技术结合

---

**创建日期**: 2025-10-13  
**版本**: 1.0  
**基于**: enhanced_fix_acc_version

