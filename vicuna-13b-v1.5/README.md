# Vicuna/LLaMA 模型 MI 量化脚本

这个脚本专门用于对 LLaMA/Vicuna 架构的模型进行基于互信息的量化和剪枝。

## 支持的模型架构

- LLaMA (所有版本)
- Vicuna (所有版本)
- Mistral
- Yi
- Qwen

## 与 OPT 架构的主要区别

| 组件 | OPT | LLaMA/Vicuna |
|------|-----|--------------|
| 层结构 | `model.decoder.layers` | `model.model.layers` |
| 嵌入层 | `model.decoder.embed_tokens` + `embed_positions` | `model.model.embed_tokens` (无位置嵌入) |
| 归一化层 | `model.decoder.final_layer_norm` | `model.model.norm` |
| Q/K/V投影 | `q_proj`, `k_proj`, `v_proj` | `q_proj`, `k_proj`, `v_proj` |
| 注意力输出 | `out_proj` | **`o_proj`** ⚠️ |
| FFN层 | `fc1`, `fc2` | **`gate_proj`, `up_proj`, `down_proj`** ⚠️ |

## 快速开始

### 1. 基本用法

```bash
# 使用 GPU 2 和 3
export CUDA_VISIBLE_DEVICES=2,3

# 运行量化
python vicuna_mi.py \
    lmsys/vicuna-13b-v1.5 \
    c4 \
    --sparsity 0.5 \
    --wbits 4 \
    --target_avg_bits 4.0 \
    --use_mi_grouping 1 \
    --n_groups 10
```

### 2. 使用测试脚本

```bash
# 编辑 test_vicuna.sh 中的模型路径
vim test_vicuna.sh

# 运行测试
./test_vicuna.sh
```

### 3. 本地模型

如果你已经下载了 Vicuna 模型到本地：

```bash
python vicuna_mi.py \
    /path/to/your/vicuna-13b-v1.5 \
    c4 \
    --sparsity 0.5 \
    --wbits 4
```

## 参数说明

### 必需参数
- `model`: 模型名称或路径
- `dataset`: 校准数据集 (wikitext2/ptb/c4)

### 可选参数
- `--nsamples`: 校准样本数 (默认: 128)
- `--sparsity`: 稀疏度 (默认: 0)
- `--wbits`: 基础量化位宽 (默认: 16)
- `--target_avg_bits`: 目标平均位宽 (默认: 4.0)
- `--use_mi_grouping`: 是否使用MI分组 (0/1, 默认: 1)
- `--n_groups`: MI聚类的分组数 (默认: 10)
- `--save`: 保存压缩后的模型路径

## 使用示例

### 示例 1: Vicuna-13b-v1.5 (50%稀疏 + 4-bit量化)

```bash
python vicuna_mi.py \
    lmsys/vicuna-13b-v1.5 \
    c4 \
    --sparsity 0.5 \
    --wbits 4 \
    --nsamples 128 \
    --save ./compressed_vicuna
```

### 示例 2: LLaMA-2-7b (仅量化)

```bash
python vicuna_mi.py \
    meta-llama/Llama-2-7b-hf \
    wikitext2 \
    --wbits 4 \
    --target_avg_bits 3.5 \
    --use_mi_grouping 1 \
    --n_groups 15
```

### 示例 3: Mistral-7B (混合精度量化)

```bash
python vicuna_mi.py \
    mistralai/Mistral-7B-v0.1 \
    c4 \
    --wbits 4 \
    --target_avg_bits 4.5 \
    --use_mi_grouping 1
```

## GPU 设置

脚本默认使用 GPU 2 和 3，你可以通过以下方式修改：

```bash
# 方法1: 环境变量
export CUDA_VISIBLE_DEVICES=0,1
python vicuna_mi.py ...

# 方法2: 修改脚本（第12-13行）
# 将 '2,3' 改为你想要的GPU编号
```

## 依赖项

确保已安装：
- `torch`
- `transformers`
- `sentencepiece` ⚠️ **LLaMA/Vicuna 模型必需**
- `numpy`
- `scikit-learn`

安装 sentencepiece:
```bash
pip install sentencepiece
```

## 常见问题

### Q1: 出现 "LlamaTokenizer requires the SentencePiece library"

**解决方法:**
```bash
pip install sentencepiece
```

### Q2: 出现 "不支持的模型架构" 错误

**原因:** 该脚本仅支持 LLaMA/Vicuna 类架构。如果要使用 OPT、BLOOM 等模型，请使用 `mutual_info_quantization/opt_mi.py`。

### Q3: CUDA out of memory

**解决方法:**
- 减少样本数: `--nsamples 64`
- 使用更少的分组: `--n_groups 5`
- 使用更大的GPU或减少batch size

### Q4: 如何查看量化统计信息？

脚本会自动打印每层的量化统计信息，包括：
- 每个分组的平均位宽
- MI值分布
- 分组效果

## 输出说明

运行完成后，你会看到：
1. **压缩过程日志**: 每层的处理进度和MI统计
2. **性能评估**: 在 wikitext2/ptb/c4 上的困惑度
3. **压缩模型**: 如果指定了 `--save`

## 适配到其他 LLaMA 类模型

如果你有其他基于 LLaMA 架构的模型（如自定义微调模型），只需：

1. 确保模型使用 `model.model.layers` 结构
2. 确保线性层命名符合 LLaMA 规范
3. 直接使用此脚本，无需修改

```bash
python vicuna_mi.py \
    /path/to/your/custom-llama-model \
    c4 \
    --sparsity 0.5 \
    --wbits 4
```

## 技术细节

该脚本实现了：
- **互信息计算**: 分析通道间的统计依赖性
- **谱聚类**: 基于MI相似度对通道分组
- **自适应量化**: 不同分组使用不同的量化位宽
- **误差补偿**: 使用Hessian信息传播量化误差

详见核心模块：
- `mutual_info.py`: MI计算
- `channel_grouping.py`: 通道分组
- `sparsegpt_mi.py`: 核心剪枝/量化逻辑

