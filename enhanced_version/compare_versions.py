"""
对比脚本：原版 vs toky版 vs 增强版

对比维度：
1. 量化比特分布
2. 模型精度 (Perplexity)
3. 执行时间
4. 内存占用
"""

import sys
import os
# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time
import copy
from tabulate import tabulate

# 导入三个版本
try:
    from sparsegpt import SparseGPT as SparseGPT_Original
    ORIGINAL_AVAILABLE = True
except:
    ORIGINAL_AVAILABLE = False
    print("⚠️ 原版 sparsegpt.py 不可用")

try:
    from sparsegpt_toky import SparseGPT as SparseGPT_Toky
    TOKY_AVAILABLE = True
except:
    TOKY_AVAILABLE = False
    print("⚠️ sparsegpt_toky.py 不可用")

from enhanced_version.sparsegpt_enhanced import SparseGPT as SparseGPT_Enhanced, QuantizationStats
from quant import Quantizer


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def create_test_layer(self, in_features=512, out_features=1024):
        """创建测试层"""
        layer = nn.Linear(in_features, out_features).to(self.device)
        return layer
    
    def generate_test_data(self, batch_size=32, seq_len=128, hidden_dim=512, num_batches=5):
        """生成测试数据"""
        data = []
        for _ in range(num_batches):
            inp = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
            data.append(inp)
        return data
    
    def compute_output_error(self, layer_original, layer_compressed, test_data):
        """计算输出误差"""
        total_error = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for inp in test_data:
                # 原始输出
                out_original = layer_original(inp.reshape(-1, inp.shape[-1]))
                
                # 压缩后输出
                out_compressed = layer_compressed(inp.reshape(-1, inp.shape[-1]))
                
                # MSE
                error = torch.mean((out_original - out_compressed) ** 2).item()
                total_error += error * inp.shape[0] * inp.shape[1]
                total_samples += inp.shape[0] * inp.shape[1]
        
        return total_error / total_samples
    
    def measure_memory(self, layer):
        """测量模型内存占用"""
        # 简化计算：只计算权重大小
        total_params = sum(p.numel() for p in layer.parameters())
        # 假设每个参数 4 bytes (float32)
        memory_mb = total_params * 4 / (1024 ** 2)
        return memory_mb


def test_version(version_name, SparseGPT_Class, layer, test_data, config):
    """测试单个版本"""
    print(f"\n{'='*60}")
    print(f"测试: {version_name}")
    print(f"{'='*60}")
    
    # 复制层（避免修改原始层）
    layer_copy = copy.deepcopy(layer)
    
    # 创建实例
    if version_name == "增强版":
        stats = QuantizationStats()
        sparsegpt = SparseGPT_Class(layer_copy, layer_name=version_name, stats_collector=stats)
    else:
        sparsegpt = SparseGPT_Class(layer_copy)
    
    # 配置量化器
    sparsegpt.quantizer = Quantizer()
    sparsegpt.quantizer.configure(bits=4, perchannel=True, sym=True)
    
    # 收集激活
    print("收集激活统计...")
    start_time = time.time()
    for inp in test_data:
        out = layer_copy(inp.reshape(-1, inp.shape[-1])).reshape(inp.shape[0], inp.shape[1], -1)
        sparsegpt.add_batch(inp, out)
    collect_time = time.time() - start_time
    
    # 执行剪枝 + 量化
    print("执行剪枝和量化...")
    start_time = time.time()
    
    if version_name == "增强版":
        sparsegpt.fasterprune(
            sparsity=config['sparsity'],
            target_avg_bits=config.get('target_avg_bits', 4.0),
            bit_allocation_method=config.get('bit_allocation_method', 'quantile')
        )
    else:
        sparsegpt.fasterprune(
            sparsity=config['sparsity']
        )
    
    prune_time = time.time() - start_time
    
    # 收集统计
    results = {
        'version': version_name,
        'collect_time': collect_time,
        'prune_time': prune_time,
        'total_time': collect_time + prune_time,
        'layer': layer_copy
    }
    
    if version_name == "增强版":
        results['stats'] = stats
        # 打印详细统计
        stats.print_summary()
    
    sparsegpt.free()
    
    return results


def compare_all_versions():
    """对比所有版本"""
    print("\n" + "="*80)
    print("SparseGPT 版本对比测试")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    evaluator = ModelEvaluator(device=device)
    
    # 创建测试层和数据
    print("\n准备测试数据...")
    original_layer = evaluator.create_test_layer(in_features=512, out_features=1024)
    test_data = evaluator.generate_test_data(batch_size=32, seq_len=128, hidden_dim=512, num_batches=5)
    
    # 配置
    config = {
        'sparsity': 0.5,
        'target_avg_bits': 4.0,
        'bit_allocation_method': 'quantile'
    }
    
    # 测试结果
    results = []
    
    # 测试原版
    if ORIGINAL_AVAILABLE:
        try:
            result = test_version("原版", SparseGPT_Original, original_layer, test_data, config)
            results.append(result)
        except Exception as e:
            print(f"❌ 原版测试失败: {e}")
    
    # 测试 toky 版
    if TOKY_AVAILABLE:
        try:
            result = test_version("Toky版", SparseGPT_Toky, original_layer, test_data, config)
            results.append(result)
        except Exception as e:
            print(f"❌ Toky版测试失败: {e}")
    
    # 测试增强版
    try:
        result = test_version("增强版", SparseGPT_Enhanced, original_layer, test_data, config)
        results.append(result)
    except Exception as e:
        print(f"❌ 增强版测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 计算输出误差
    print("\n" + "="*80)
    print("计算输出误差 (相对于原始层)")
    print("="*80)
    
    for result in results:
        error = evaluator.compute_output_error(original_layer, result['layer'], test_data)
        result['output_error'] = error
        print(f"{result['version']:12s}: MSE = {error:.6f}")
    
    # 生成对比表格
    print("\n" + "="*80)
    print("综合对比表")
    print("="*80)
    
    table_data = []
    for result in results:
        row = [
            result['version'],
            f"{result['collect_time']:.2f}s",
            f"{result['prune_time']:.2f}s",
            f"{result['total_time']:.2f}s",
            f"{result['output_error']:.6f}"
        ]
        table_data.append(row)
    
    headers = ["版本", "收集时间", "剪枝时间", "总时间", "输出误差(MSE)"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 比特分布对比（仅增强版有详细统计）
    print("\n" + "="*80)
    print("比特分布对比")
    print("="*80)
    
    for result in results:
        if 'stats' in result:
            print(f"\n{result['version']}:")
            bit_dist = result['stats'].bit_distribution
            total = sum(bit_dist.values())
            for bit in sorted(bit_dist.keys()):
                count = bit_dist[bit]
                percentage = count / total * 100
                print(f"  {bit}-bit: {count:5d} 通道 ({percentage:5.2f}%)")
        else:
            print(f"\n{result['version']}: 固定 4-bit 量化")
    
    print("\n" + "="*80)
    print("测试完成! ✅")
    print("="*80)


def quick_demo():
    """快速演示"""
    print("\n" + "="*80)
    print("快速演示：增强版的优势")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建一个简单的层
    layer = nn.Linear(256, 512).to(device)
    
    print("\n权重形状:", layer.weight.shape)
    
    # 创建 SparseGPT
    stats = QuantizationStats()
    sparsegpt = SparseGPT_Enhanced(layer, layer_name="demo_layer", stats_collector=stats)
    
    # 配置量化
    sparsegpt.quantizer = Quantizer()
    sparsegpt.quantizer.configure(bits=4, perchannel=True, sym=True)
    
    # 模拟数据
    print("\n收集激活统计...")
    for i in range(3):
        inp = torch.randn(16, 64, 256, device=device)
        out = layer(inp.reshape(-1, 256)).reshape(16, 64, -1)
        sparsegpt.add_batch(inp, out)
        print(f"  批次 {i+1}/3 完成")
    
    # 执行剪枝和量化
    print("\n执行剪枝和量化...")
    sparsegpt.fasterprune(
        sparsity=0.5,
        target_avg_bits=4.0,
        bit_allocation_method='quantile'
    )
    
    # 显示统计
    stats.print_summary()
    
    print("\n演示完成! ✅")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    import sys
    
    # 检查 tabulate 是否安装
    try:
        import tabulate
    except ImportError:
        print("⚠️ 请安装 tabulate: pip install tabulate")
        print("运行快速演示模式...")
        quick_demo()
        sys.exit(0)
    
    # 运行完整对比
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_demo()
    else:
        try:
            compare_all_versions()
        except Exception as e:
            print(f"\n❌ 对比测试出错: {e}")
            import traceback
            traceback.print_exc()
            print("\n尝试运行快速演示...")
            quick_demo()

