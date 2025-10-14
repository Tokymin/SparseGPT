"""
推理性能测试脚本

测试压缩后模型的推理时间、吞吐量和GPU内存使用
"""

import time
import torch
import torch.nn as nn
import argparse
import sys
import os
import numpy as np

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datautils import get_loaders


def get_opt(model_name):
    """加载OPT模型"""
    def skip_init(*args, **kwargs):
        pass
    
    torch.nn.init.kaiming_uniform_ = skip_init
    torch.nn.init.uniform_ = skip_init
    torch.nn.init.normal_ = skip_init
    
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.seqlen = model.config.max_position_embeddings
    return model


def benchmark_inference(model, testloader, dev, n_samples=100):
    """
    测试推理性能
    
    Returns:
        avg_time: 平均推理时间 (秒/样本)
        throughput: 吞吐量 (样本/秒)
        peak_memory: 峰值GPU内存 (GB)
    """
    model.eval()
    model = model.to(dev)
    
    # 预热
    print("预热中...")
    with torch.no_grad():
        for i in range(5):
            batch = testloader.input_ids[:, :model.seqlen].to(dev)
            _ = model(batch)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # 开始测试
    print(f"开始推理测试 ({n_samples} 样本)...")
    times = []
    
    with torch.no_grad():
        for i in range(n_samples):
            # 准备输入
            start_idx = (i * model.seqlen) % (testloader.input_ids.shape[1] - model.seqlen)
            batch = testloader.input_ids[:, start_idx:(start_idx + model.seqlen)].to(dev)
            
            # 计时
            torch.cuda.synchronize()
            start_time = time.time()
            
            _ = model(batch)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{n_samples}")
    
    # 统计
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = 1.0 / avg_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    return avg_time, std_time, throughput, peak_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='OPT model to load')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'], 
                        help='Test dataset')
    parser.add_argument('--n_samples', type=int, default=100, 
                        help='Number of samples for benchmarking')
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='Device to use')
    
    args = parser.parse_args()
    
    # 设置设备
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("推理性能测试")
    print("="*80)
    print(f"模型: {args.model}")
    print(f"数据集: {args.dataset}")
    print(f"测试样本数: {args.n_samples}")
    print(f"设备: {dev}")
    print("="*80)
    print()
    
    # 加载模型
    print("加载模型...")
    model = get_opt(args.model)
    
    # 加载数据
    print("加载数据...")
    _, testloader = get_loaders(
        args.dataset, 
        seed=0, 
        model=args.model, 
        seqlen=model.seqlen
    )
    
    # 测试推理性能
    avg_time, std_time, throughput, peak_memory = benchmark_inference(
        model, testloader, dev, args.n_samples
    )
    
    # 打印结果
    print()
    print("="*80)
    print("测试结果")
    print("="*80)
    print(f"平均推理时间: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms/样本")
    print(f"吞吐量: {throughput:.2f} 样本/秒")
    print(f"峰值GPU内存: {peak_memory:.2f} GB")
    print("="*80)
    
    # 保存结果
    result_file = f"inference_benchmark_{args.dataset}.txt"
    with open(result_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Samples: {args.n_samples}\n")
        f.write(f"Avg Time (ms): {avg_time*1000:.2f}\n")
        f.write(f"Std Time (ms): {std_time*1000:.2f}\n")
        f.write(f"Throughput (samples/s): {throughput:.2f}\n")
        f.write(f"Peak Memory (GB): {peak_memory:.2f}\n")
    
    print(f"\n结果已保存到: {result_file}")


if __name__ == '__main__':
    main()

