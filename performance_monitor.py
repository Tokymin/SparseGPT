#!/usr/bin/env python3
"""
SparseGPT 性能监控和优化工具
用于实时监控模型压缩效果和性能指标
"""

import torch
import time
import psutil
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import json
from pathlib import Path


class PerformanceMonitor:
    """性能监控器：跟踪压缩过程中的关键指标"""
    
    def __init__(self, model_name: str, save_dir: str = "./monitoring_results"):
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 监控数据存储
        self.metrics = {
            'layer_metrics': [],
            'memory_usage': [],
            'compression_ratios': [],
            'quantization_errors': [],
            'timing': []
        }
        
        # 基准性能
        self.baseline_ppl = None
        self.baseline_memory = None
        
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        self.baseline_memory = psutil.virtual_memory().used / 1024**3  # GB
        print(f"🔍 开始监控 {self.model_name}")
        print(f"📊 基准内存使用: {self.baseline_memory:.2f} GB")
        
    def log_layer_processing(self, layer_idx: int, layer_name: str, 
                           sparsity: float, quantization_bits: int,
                           compression_ratio: float, error: float):
        """记录单层处理指标"""
        layer_metric = {
            'layer_idx': layer_idx,
            'layer_name': layer_name,
            'sparsity': sparsity,
            'quantization_bits': quantization_bits,
            'compression_ratio': compression_ratio,
            'quantization_error': error,
            'timestamp': time.time() - self.start_time
        }
        self.metrics['layer_metrics'].append(layer_metric)
        
        print(f"📈 Layer {layer_idx} ({layer_name}): "
              f"Sparsity={sparsity:.3f}, Bits={quantization_bits}, "
              f"Compression={compression_ratio:.2f}x, Error={error:.6f}")
    
    def log_memory_usage(self):
        """记录内存使用情况"""
        current_memory = psutil.virtual_memory().used / 1024**3
        gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        memory_info = {
            'cpu_memory_gb': current_memory,
            'gpu_memory_gb': gpu_memory,
            'timestamp': time.time() - self.start_time
        }
        self.metrics['memory_usage'].append(memory_info)
        
    def log_perplexity(self, dataset: str, perplexity: float, is_baseline: bool = False):
        """记录困惑度结果"""
        if is_baseline:
            self.baseline_ppl = perplexity
            print(f"📊 基准困惑度 ({dataset}): {perplexity:.3f}")
        else:
            improvement = ((self.baseline_ppl - perplexity) / self.baseline_ppl * 100) if self.baseline_ppl else 0
            print(f"📊 压缩后困惑度 ({dataset}): {perplexity:.3f} "
                  f"(变化: {improvement:+.2f}%)")
    
    def calculate_compression_stats(self, model) -> Dict:
        """计算整体压缩统计"""
        total_params = sum(p.numel() for p in model.parameters())
        sparse_params = sum(torch.sum(p == 0).item() for p in model.parameters() if 'weight' in str(p))
        
        # 计算量化节省
        quantized_layers = 0
        total_quantization_bits = 0
        for name, param in model.named_parameters():
            if 'weight' in name and hasattr(param, 'quantization_bits'):
                quantized_layers += 1
                total_quantization_bits += param.quantization_bits
        
        avg_bits = total_quantization_bits / max(quantized_layers, 1)
        
        stats = {
            'total_parameters': total_params,
            'sparse_parameters': sparse_params,
            'sparsity_ratio': sparse_params / total_params,
            'quantized_layers': quantized_layers,
            'average_bits': avg_bits,
            'estimated_compression': 32 / avg_bits if avg_bits > 0 else 1.0
        }
        
        return stats
    
    def generate_report(self, model, final_ppl: Dict[str, float]):
        """生成性能报告"""
        stats = self.calculate_compression_stats(model)
        
        report = {
            'model_name': self.model_name,
            'processing_time': time.time() - self.start_time,
            'compression_stats': stats,
            'final_perplexity': final_ppl,
            'layer_metrics': self.metrics['layer_metrics'],
            'memory_usage': self.metrics['memory_usage']
        }
        
        # 保存报告
        report_path = self.save_dir / f"{self.model_name.replace('/', '_')}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 打印摘要
        print("\n" + "="*60)
        print("📊 压缩性能报告")
        print("="*60)
        print(f"🏷️  模型: {self.model_name}")
        print(f"⏱️  处理时间: {report['processing_time']:.2f}秒")
        print(f"🗜️  稀疏度: {stats['sparsity_ratio']:.1%}")
        print(f"🔢 量化层数: {stats['quantized_layers']}")
        print(f"📏 平均比特: {stats['average_bits']:.1f}")
        print(f"📦 估计压缩比: {stats['estimated_compression']:.1f}x")
        
        for dataset, ppl in final_ppl.items():
            print(f"📈 {dataset} 困惑度: {ppl:.3f}")
        
        print(f"💾 报告已保存至: {report_path}")
        print("="*60)
        
        return report
    
    def plot_metrics(self):
        """绘制性能指标图表"""
        if not self.metrics['layer_metrics']:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Performance Metrics - {self.model_name}', fontsize=16)
        
        # 稀疏度变化
        layers = [m['layer_idx'] for m in self.metrics['layer_metrics']]
        sparsities = [m['sparsity'] for m in self.metrics['layer_metrics']]
        axes[0, 0].plot(layers, sparsities, 'b-o')
        axes[0, 0].set_title('Sparsity by Layer')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Sparsity Ratio')
        axes[0, 0].grid(True)
        
        # 压缩比
        compression_ratios = [m['compression_ratio'] for m in self.metrics['layer_metrics']]
        axes[0, 1].plot(layers, compression_ratios, 'g-o')
        axes[0, 1].set_title('Compression Ratio by Layer')
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('Compression Ratio')
        axes[0, 1].grid(True)
        
        # 量化误差
        errors = [m['quantization_error'] for m in self.metrics['layer_metrics']]
        axes[1, 0].plot(layers, errors, 'r-o')
        axes[1, 0].set_title('Quantization Error by Layer')
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].grid(True)
        
        # 内存使用
        if self.metrics['memory_usage']:
            timestamps = [m['timestamp'] for m in self.metrics['memory_usage']]
            cpu_memory = [m['cpu_memory_gb'] for m in self.metrics['memory_usage']]
            axes[1, 1].plot(timestamps, cpu_memory, 'purple', label='CPU Memory')
            if torch.cuda.is_available():
                gpu_memory = [m['gpu_memory_gb'] for m in self.metrics['memory_usage']]
                axes[1, 1].plot(timestamps, gpu_memory, 'orange', label='GPU Memory')
            axes[1, 1].set_title('Memory Usage Over Time')
            axes[1, 1].set_xlabel('Time (seconds)')
            axes[1, 1].set_ylabel('Memory (GB)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = self.save_dir / f"{self.model_name.replace('/', '_')}_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 性能图表已保存至: {plot_path}")


def optimize_quantization_parameters(model, monitor: PerformanceMonitor):
    """动态优化量化参数"""
    print("🔧 开始动态优化量化参数...")
    
    # 分析每层的激活分布
    layer_activations = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 这里可以添加激活分布分析
            pass
    
    # 根据激活分布调整量化参数
    optimization_suggestions = []
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            # 分析权重分布
            weight_std = torch.std(param).item()
            weight_mean = torch.mean(torch.abs(param)).item()
            
            # 建议量化比特数
            if weight_std > 0.1:  # 高方差，需要更多比特
                suggested_bits = 8
            elif weight_std > 0.05:  # 中等方差
                suggested_bits = 4
            else:  # 低方差，可以用更少比特
                suggested_bits = 2
                
            optimization_suggestions.append({
                'layer': name,
                'current_std': weight_std,
                'suggested_bits': suggested_bits
            })
    
    print("💡 量化优化建议:")
    for suggestion in optimization_suggestions[:5]:  # 显示前5个
        print(f"  {suggestion['layer']}: 建议 {suggestion['suggested_bits']}bit "
              f"(当前std: {suggestion['current_std']:.4f})")
    
    return optimization_suggestions


if __name__ == "__main__":
    # 示例用法
    monitor = PerformanceMonitor("facebook/opt-125m")
    monitor.start_monitoring()
    
    # 模拟一些数据
    for i in range(12):
        monitor.log_layer_processing(
            layer_idx=i,
            layer_name=f"layer_{i}",
            sparsity=0.1 * i,
            quantization_bits=4,
            compression_ratio=2.0 + 0.1 * i,
            error=0.001 * i
        )
        monitor.log_memory_usage()
    
    # 生成报告
    final_ppl = {"wikitext2": 25.953, "ptb": 38.985}
    report = monitor.generate_report(None, final_ppl)
    monitor.plot_metrics()
