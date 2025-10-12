"""
测试增强版 SparseGPT 的示例脚本

演示:
1. 如何使用增强版的激活感知量化
2. 如何收集和显示统计信息
3. 对比不同比特分配方法的效果
"""

import torch
import torch.nn as nn
from sparsegpt_enhanced import SparseGPT, QuantizationStats


def create_test_model():
    """创建一个简单的测试模型"""
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512)
    )
    return model


def test_enhanced_quantization():
    """测试增强版量化"""
    print("="*80)
    print("测试增强版 SparseGPT - 激活感知量化精度分配")
    print("="*80)
    
    # 设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建模型
    model = create_test_model().to(device)
    print(f"模型结构: {len([m for m in model.modules() if isinstance(m, nn.Linear)])} 个线性层")
    
    # 创建统计收集器
    stats_collector = QuantizationStats()
    
    # 模拟数据
    batch_size = 32
    seq_len = 128
    hidden_dim = 512
    num_batches = 5
    
    print(f"\n生成模拟数据: batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")
    
    # 测试第一个线性层
    layer = model[0]  # 第一个线性层
    layer_name = "model.0.linear"
    
    print(f"\n处理层: {layer_name}")
    print(f"权重形状: {layer.weight.shape}")
    
    # 创建 SparseGPT 实例
    sparsegpt = SparseGPT(layer, layer_name=layer_name, stats_collector=stats_collector)
    
    # 模拟量化器
    from quant import Quantizer
    sparsegpt.quantizer = Quantizer()
    sparsegpt.quantizer.configure(bits=4, perchannel=True, sym=True)
    
    print("\n阶段1: 收集激活统计信息...")
    # 收集多个批次的统计信息
    for batch_idx in range(num_batches):
        # 生成随机输入（模拟真实数据分布）
        inp = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        out = layer(inp.reshape(-1, hidden_dim)).reshape(batch_size, seq_len, -1)
        
        sparsegpt.add_batch(inp, out)
        print(f"  批次 {batch_idx+1}/{num_batches} 完成")
    
    print("\n阶段2: 执行剪枝 + 增强量化...")
    
    # 测试不同的配置
    configs = [
        {
            'name': '分位数方法 (4-bit平均)',
            'sparsity': 0.5,
            'target_avg_bits': 4.0,
            'method': 'quantile'
        },
        {
            'name': '分位数方法 (3-bit平均)',
            'sparsity': 0.5,
            'target_avg_bits': 3.0,
            'method': 'quantile'
        },
        {
            'name': '预算方法 (4-bit平均)',
            'sparsity': 0.5,
            'target_avg_bits': 4.0,
            'method': 'budget'
        }
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"配置: {config['name']}")
        print(f"{'='*60}")
        
        # 重新创建实例（避免状态污染）
        sparsegpt_test = SparseGPT(layer, layer_name=config['name'], stats_collector=stats_collector)
        sparsegpt_test.quantizer = Quantizer()
        sparsegpt_test.quantizer.configure(bits=4, perchannel=True, sym=True)
        
        # 收集统计
        for batch_idx in range(num_batches):
            inp = torch.randn(batch_size, seq_len, hidden_dim, device=device)
            out = layer(inp.reshape(-1, hidden_dim)).reshape(batch_size, seq_len, -1)
            sparsegpt_test.add_batch(inp, out)
        
        # 执行剪枝 + 量化
        sparsegpt_test.fasterprune(
            sparsity=config['sparsity'],
            target_avg_bits=config['target_avg_bits'],
            bit_allocation_method=config['method']
        )
        
        sparsegpt_test.free()
    
    # 打印整体统计
    stats_collector.print_summary()
    
    print("\n测试完成! ✅")


def compare_with_original():
    """对比原版和增强版的效果"""
    print("\n" + "="*80)
    print("对比测试: 原版 vs 增强版")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建两个相同的层
    layer_original = nn.Linear(512, 1024).to(device)
    layer_enhanced = nn.Linear(512, 1024).to(device)
    
    # 复制权重
    layer_enhanced.load_state_dict(layer_original.state_dict())
    
    print("\n1. 测试原版 (固定4-bit)...")
    # 这里需要原版的 sparsegpt.py
    # from sparsegpt import SparseGPT as SparseGPT_Original
    # ... (简化示例，实际需要完整实现)
    
    print("\n2. 测试增强版 (混合精度2/3/4/6/8-bit)...")
    stats = QuantizationStats()
    sparsegpt = SparseGPT(layer_enhanced, layer_name="enhanced", stats_collector=stats)
    
    # 模拟数据收集
    for _ in range(3):
        inp = torch.randn(32, 128, 512, device=device)
        out = layer_enhanced(inp.reshape(-1, 512)).reshape(32, 128, -1)
        sparsegpt.add_batch(inp, out)
    
    # 配置量化器
    from quant import Quantizer
    sparsegpt.quantizer = Quantizer()
    sparsegpt.quantizer.configure(bits=4, perchannel=True, sym=True)
    
    # 执行增强量化
    sparsegpt.fasterprune(
        sparsity=0.5,
        target_avg_bits=4.0,
        bit_allocation_method='quantile'
    )
    
    stats.print_summary()
    
    print("\n对比完成! ✅")


def analyze_importance_scores():
    """分析重要性分数的各个组成部分"""
    print("\n" + "="*80)
    print("重要性分数分析")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layer = nn.Linear(512, 1024).to(device)
    
    sparsegpt = SparseGPT(layer, layer_name="analysis")
    
    # 收集激活
    print("\n收集激活统计...")
    for _ in range(5):
        inp = torch.randn(32, 128, 512, device=device)
        out = layer(inp.reshape(-1, 512)).reshape(32, 128, -1)
        sparsegpt.add_batch(inp, out)
    
    # 计算 Hinv
    H = sparsegpt.H
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(sparsegpt.columns, device=device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H
    
    # 计算重要性
    W = layer.weight.data.float()
    importance_scores, component_scores = sparsegpt.compute_importance_scores(W, Hinv)
    
    print("\n重要性分数统计:")
    print(f"  总体范围: [{importance_scores.min():.3f}, {importance_scores.max():.3f}]")
    print(f"  平均值: {importance_scores.mean():.3f}")
    print(f"  标准差: {importance_scores.std():.3f}")
    
    print("\n各组成部分统计:")
    for name, scores in component_scores.items():
        print(f"  {name:12s}: 范围=[{scores.min():.3f}, {scores.max():.3f}], "
              f"均值={scores.mean():.3f}, 标准差={scores.std():.3f}")
    
    print("\n分析完成! ✅")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print("\n🚀 SparseGPT Enhanced - 测试套件\n")
    
    # 运行测试
    try:
        # 测试1: 基本功能
        test_enhanced_quantization()
        
        # 测试2: 重要性分析
        analyze_importance_scores()
        
        # 测试3: 对比测试 (可选)
        # compare_with_original()
        
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("所有测试完成!")
    print("="*80)

