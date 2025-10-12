"""
简单使用示例：如何将增强版 SparseGPT 集成到您的工作流

场景：对 OPT 或 BLOOM 模型的某一层进行剪枝和混合精度量化
"""

import torch
import torch.nn as nn
from sparsegpt_enhanced import SparseGPT, QuantizationStats
from quant import Quantizer


def example_1_basic_usage():
    """示例 1: 基本使用"""
    print("\n" + "="*60)
    print("示例 1: 基本使用")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 假设你已经加载了一个模型
    layer = nn.Linear(4096, 4096).to(device)
    
    # 创建 SparseGPT 实例
    sparsegpt = SparseGPT(layer, layer_name="transformer.layer.0")
    
    # 配置量化器（4-bit 基础精度）
    sparsegpt.quantizer = Quantizer()
    sparsegpt.quantizer.configure(bits=4, perchannel=True, sym=True)
    
    # 收集激活统计（通过前向传播）
    # 这里用随机数据模拟，实际应该用真实数据
    for _ in range(10):  # 多个批次
        inp = torch.randn(8, 512, 4096, device=device)  # [batch, seq, hidden]
        out = layer(inp.reshape(-1, 4096)).reshape(8, 512, -1)
        sparsegpt.add_batch(inp, out)
    
    # 执行剪枝和混合精度量化
    sparsegpt.fasterprune(
        sparsity=0.5,              # 50% 权重剪为 0
        target_avg_bits=4.0,       # 平均 4-bit（范围: 2-8 bit）
        bit_allocation_method='quantile'
    )
    
    # 现在 layer 的权重已经被剪枝和量化了
    print("✅ 完成！")
    sparsegpt.free()


def example_2_with_statistics():
    """示例 2: 使用统计收集器"""
    print("\n" + "="*60)
    print("示例 2: 收集详细统计")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建统计收集器
    stats = QuantizationStats()
    
    # 处理多个层
    layers = [
        nn.Linear(4096, 4096).to(device),
        nn.Linear(4096, 16384).to(device),
        nn.Linear(16384, 4096).to(device)
    ]
    
    for i, layer in enumerate(layers):
        print(f"\n处理层 {i}...")
        
        sparsegpt = SparseGPT(
            layer, 
            layer_name=f"layer_{i}",
            stats_collector=stats  # 传入统计收集器
        )
        
        # 配置量化
        sparsegpt.quantizer = Quantizer()
        sparsegpt.quantizer.configure(bits=4, perchannel=True, sym=True)
        
        # 收集激活
        for _ in range(5):
            in_dim = layer.in_features
            inp = torch.randn(8, 256, in_dim, device=device)
            out = layer(inp.reshape(-1, in_dim)).reshape(8, 256, -1)
            sparsegpt.add_batch(inp, out)
        
        # 剪枝和量化
        sparsegpt.fasterprune(
            sparsity=0.5,
            target_avg_bits=4.0,
            bit_allocation_method='quantile'
        )
        
        sparsegpt.free()
    
    # 打印所有层的统计
    stats.print_summary()
    
    print("\n✅ 完成！")


def example_3_different_configs():
    """示例 3: 不同层使用不同配置"""
    print("\n" + "="*60)
    print("示例 3: 自适应配置")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟 Transformer 的不同层
    layers_config = [
        {'name': 'embedding', 'sparsity': 0.3, 'avg_bits': 5.0},
        {'name': 'attention_qkv', 'sparsity': 0.4, 'avg_bits': 4.5},
        {'name': 'attention_out', 'sparsity': 0.5, 'avg_bits': 4.0},
        {'name': 'ffn_in', 'sparsity': 0.6, 'avg_bits': 3.5},
        {'name': 'ffn_out', 'sparsity': 0.5, 'avg_bits': 4.0},
    ]
    
    for config in layers_config:
        print(f"\n处理 {config['name']}...")
        print(f"  剪枝率: {config['sparsity']*100}%")
        print(f"  目标比特: {config['avg_bits']}")
        
        layer = nn.Linear(2048, 2048).to(device)
        sparsegpt = SparseGPT(layer, layer_name=config['name'])
        
        sparsegpt.quantizer = Quantizer()
        sparsegpt.quantizer.configure(bits=4, perchannel=True, sym=True)
        
        # 收集激活
        for _ in range(3):
            inp = torch.randn(4, 128, 2048, device=device)
            out = layer(inp.reshape(-1, 2048)).reshape(4, 128, -1)
            sparsegpt.add_batch(inp, out)
        
        # 应用不同的配置
        sparsegpt.fasterprune(
            sparsity=config['sparsity'],
            target_avg_bits=config['avg_bits'],
            bit_allocation_method='quantile'
        )
        
        sparsegpt.free()
    
    print("\n✅ 完成！")


def example_4_integration_with_model():
    """示例 4: 集成到完整模型"""
    print("\n" + "="*60)
    print("示例 4: 完整模型集成（伪代码）")
    print("="*60)
    
    print("""
# 伪代码示例：如何集成到 OPT 模型

from transformers import AutoModelForCausalLM
from sparsegpt_enhanced import SparseGPT, QuantizationStats

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
model.eval()

# 2. 加载校准数据
dataloader = get_calibration_data(dataset="c4", nsamples=128)

# 3. 创建统计收集器
stats = QuantizationStats()

# 4. 遍历所有线性层
for name, module in model.named_modules():
    if not isinstance(module, nn.Linear):
        continue
    
    print(f"Processing {name}...")
    
    # 创建 SparseGPT
    sparsegpt = SparseGPT(module, layer_name=name, stats_collector=stats)
    sparsegpt.quantizer = Quantizer()
    sparsegpt.quantizer.configure(bits=4, perchannel=True, sym=True)
    
    # 收集激活（通过 hook）
    def hook_fn(module, inp, out):
        sparsegpt.add_batch(inp[0], out)
    
    handle = module.register_forward_hook(hook_fn)
    
    # 前向传播校准数据
    with torch.no_grad():
        for batch in dataloader:
            model(batch['input_ids'])
    
    handle.remove()
    
    # 剪枝和量化
    sparsegpt.fasterprune(
        sparsity=0.5,
        target_avg_bits=4.0,
        bit_allocation_method='quantile'
    )
    
    sparsegpt.free()

# 5. 打印统计
stats.print_summary()

# 6. 保存压缩后的模型
model.save_pretrained("opt-125m-sparse50-mixed4bit")
""")
    
    print("\n✅ 示例说明完成！")


def example_5_compare_methods():
    """示例 5: 对比不同的比特分配方法"""
    print("\n" + "="*60)
    print("示例 5: 对比分位数 vs 预算方法")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layer_orig = nn.Linear(1024, 1024).to(device)
    
    methods = ['quantile', 'budget']
    
    for method in methods:
        print(f"\n--- 方法: {method} ---")
        
        # 复制层
        import copy
        layer = copy.deepcopy(layer_orig)
        
        stats = QuantizationStats()
        sparsegpt = SparseGPT(layer, layer_name=f"test_{method}", stats_collector=stats)
        
        sparsegpt.quantizer = Quantizer()
        sparsegpt.quantizer.configure(bits=4, perchannel=True, sym=True)
        
        # 收集激活
        for _ in range(5):
            inp = torch.randn(8, 128, 1024, device=device)
            out = layer(inp.reshape(-1, 1024)).reshape(8, 128, -1)
            sparsegpt.add_batch(inp, out)
        
        # 剪枝和量化
        sparsegpt.fasterprune(
            sparsity=0.5,
            target_avg_bits=4.0,
            bit_allocation_method=method
        )
        
        stats.print_summary()
        sparsegpt.free()
    
    print("\n✅ 对比完成！")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SparseGPT Enhanced - 使用示例")
    print("="*80)
    
    # 运行所有示例
    examples = [
        ("基本使用", example_1_basic_usage),
        ("统计收集", example_2_with_statistics),
        ("自适应配置", example_3_different_configs),
        ("完整模型集成", example_4_integration_with_model),
        ("方法对比", example_5_compare_methods),
    ]
    
    import sys
    
    if len(sys.argv) > 1:
        # 运行指定示例
        idx = int(sys.argv[1]) - 1
        if 0 <= idx < len(examples):
            print(f"\n运行示例 {idx+1}...")
            examples[idx][1]()
        else:
            print(f"错误：示例编号应在 1-{len(examples)} 之间")
    else:
        # 显示菜单
        print("\n可用示例：")
        for i, (name, _) in enumerate(examples):
            print(f"  {i+1}. {name}")
        print("\n用法: python example_usage.py <示例编号>")
        print("例如: python example_usage.py 1")
        
        # 默认运行示例 1
        print("\n运行示例 1（基本使用）...")
        example_1_basic_usage()

