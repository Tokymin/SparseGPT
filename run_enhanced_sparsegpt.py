#!/usr/bin/env python3
"""
增强版 SparseGPT 运行脚本
集成性能监控、动态优化和结果分析
"""

import argparse
import time
import torch
import json
from pathlib import Path
from performance_monitor import PerformanceMonitor, optimize_quantization_parameters
from opt_toky import get_opt, opt_sequential, opt_eval
from datautils import get_loaders


def run_enhanced_sparsegpt():
    """运行增强版 SparseGPT"""
    parser = argparse.ArgumentParser(description='Enhanced SparseGPT with Performance Monitoring')
    
    # 基础参数
    parser.add_argument('model', type=str, help='Model to load (e.g., facebook/opt-125m)')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'], 
                       help='Dataset for calibration/evaluation')
    
    # 压缩参数
    parser.add_argument('--sparsity', type=float, default=0.5, help='Target sparsity ratio')
    parser.add_argument('--wbits', type=int, default=4, help='Quantization bits')
    parser.add_argument('--nsamples', type=int, default=128, help='Calibration samples')
    
    # 高级参数
    parser.add_argument('--use-mutual-info', action='store_true', help='Enable mutual information grouping')
    parser.add_argument('--dynamic-scaling', action='store_true', help='Enable dynamic scaling')
    parser.add_argument('--num-clusters', type=int, default=4, help='Number of clusters for grouping')
    
    # 监控参数
    parser.add_argument('--monitor', action='store_true', help='Enable performance monitoring')
    parser.add_argument('--save-results', type=str, default='./results', help='Save results directory')
    parser.add_argument('--optimize-params', action='store_true', help='Enable parameter optimization')
    
    args = parser.parse_args()
    
    # 初始化性能监控器
    if args.monitor:
        monitor = PerformanceMonitor(args.model, args.save_results)
        monitor.start_monitoring()
    else:
        monitor = None
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")
    print(f"📊 模型: {args.model}")
    print(f"📈 数据集: {args.dataset}")
    print(f"🗜️  目标稀疏度: {args.sparsity}")
    print(f"🔢 量化比特: {args.wbits}")
    
    # 加载模型
    print("📥 加载模型...")
    model = get_opt(args.model)
    model.eval()
    
    # 记录基准性能
    if monitor:
        baseline_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        print(f"📊 基准内存使用: {baseline_memory:.2f} GB")
    
    # 加载数据
    print("📊 加载校准数据...")
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=0,
        model=args.model, seqlen=model.seqlen
    )
    
    # 执行压缩
    if args.sparsity > 0:
        print("🔧 开始稀疏化和量化...")
        start_time = time.time()
        
        # 创建参数对象
        class Args:
            def __init__(self):
                self.sparsity = args.sparsity
                self.wbits = args.wbits
                self.nsamples = args.nsamples
                self.use_mutual_info = args.use_mutual_info
                self.dynamic_scaling = args.dynamic_scaling
                self.num_clusters = args.num_clusters
                self.prunen = 0
                self.prunem = 0
                self.percdamp = 0.01
                self.blocksize = 128
                self.gmp = False
                self.minlayer = -1
                self.maxlayer = 1000
                self.prune_only = ''
                self.invert = False
                self.log_wandb = False
        
        args_obj = Args()
        
        # 执行压缩
        opt_sequential(model, dataloader, device, args_obj)
        
        compression_time = time.time() - start_time
        print(f"⏱️  压缩完成，耗时: {compression_time:.2f}秒")
        
        # 记录压缩统计
        if monitor:
            total_params = sum(p.numel() for p in model.parameters())
            sparse_params = sum(torch.sum(p == 0).item() for p in model.parameters() if 'weight' in str(p))
            actual_sparsity = sparse_params / total_params
            
            print(f"📊 实际稀疏度: {actual_sparsity:.1%}")
            print(f"🗜️  压缩比: {1/(1-actual_sparsity):.1f}x")
    
    # 多数据集评估
    print("📈 开始评估...")
    results = {}
    
    for dataset in ['wikitext2', 'ptb']:
        print(f"🔍 评估 {dataset}...")
        
        # 重新加载数据
        _, testloader = get_loaders(
            dataset, seed=0, model=args.model, seqlen=model.seqlen
        )
        
        # 评估
        ppl = opt_eval(model, testloader, device, dataset, args_obj, False)
        results[dataset] = ppl
        
        if monitor:
            monitor.log_perplexity(dataset, ppl, is_baseline=False)
    
    # 参数优化建议
    if args.optimize_params and monitor:
        print("🔧 分析量化参数优化机会...")
        suggestions = optimize_quantization_parameters(model, monitor)
    
    # 生成最终报告
    if monitor:
        print("📊 生成性能报告...")
        report = monitor.generate_report(model, results)
        monitor.plot_metrics()
        
        # 保存模型
        if args.save_results:
            save_path = Path(args.save_results) / f"{args.model.replace('/', '_')}_compressed"
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 保存模型状态
            torch.save(model.state_dict(), save_path / "model_state.pt")
            
            # 保存压缩配置
            config = {
                'model': args.model,
                'sparsity': args.sparsity,
                'wbits': args.wbits,
                'use_mutual_info': args.use_mutual_info,
                'dynamic_scaling': args.dynamic_scaling,
                'results': results
            }
            with open(save_path / "config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"💾 模型已保存至: {save_path}")
    
    # 打印最终结果
    print("\n" + "="*60)
    print("🎉 压缩完成！最终结果:")
    print("="*60)
    for dataset, ppl in results.items():
        print(f"📊 {dataset}: {ppl:.3f}")
    print("="*60)


if __name__ == "__main__":
    run_enhanced_sparsegpt()
