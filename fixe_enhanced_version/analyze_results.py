#!/usr/bin/env python3
"""
全面结果分析和可视化工具

功能:
1. 读取所有测试结果
2. 生成对比表格
3. 绘制性能对比图
4. 计算统计指标
5. 生成 Markdown 报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_palette("husl")


class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, base_dir=None):
        # 获取脚本所在目录
        if base_dir is None:
            base_dir = Path(__file__).parent
        else:
            base_dir = Path(base_dir)
        
        self.result_dir = base_dir / 'comprehensive_results'
        self.stats_dir = base_dir / 'statistical_results'
        self.complexity_dir = base_dir / 'complexity_results'
        self.output_dir = base_dir / 'analysis_output'
        self.output_dir.mkdir(exist_ok=True)
        
        self.df_comprehensive = None
        self.df_stats = None
        self.df_complexity = None
        
    def load_data(self):
        """加载所有数据"""
        print("="*80)
        print("加载测试数据...")
        print("="*80)
        
        # 加载全面测试结果
        csv_file = self.result_dir / 'results.csv'
        if csv_file.exists():
            self.df_comprehensive = pd.read_csv(csv_file)
            print(f"✓ 加载全面测试数据: {len(self.df_comprehensive)} 条记录")
        else:
            print(f"✗ 未找到全面测试数据: {csv_file}")
        
        # 加载统计测试结果
        csv_file = self.stats_dir / 'statistical_runs.csv'
        if csv_file.exists():
            self.df_stats = pd.read_csv(csv_file)
            print(f"✓ 加载统计测试数据: {len(self.df_stats)} 条记录")
        else:
            print(f"⚠ 未找到统计测试数据: {csv_file}")
        
        # 加载复杂度测试结果
        csv_file = self.complexity_dir / 'complexity_benchmark.csv'
        if csv_file.exists():
            self.df_complexity = pd.read_csv(csv_file)
            print(f"✓ 加载复杂度测试数据: {len(self.df_complexity)} 条记录")
        else:
            print(f"⚠ 未找到复杂度测试数据: {csv_file}")
        
        print()
    
    def analyze_comprehensive_results(self):
        """分析全面测试结果"""
        if self.df_comprehensive is None or len(self.df_comprehensive) == 0:
            print("无全面测试数据，跳过分析")
            return
        
        print("="*80)
        print("全面测试结果分析")
        print("="*80)
        
        df = self.df_comprehensive
        
        # 按配置分组计算统计量
        grouped = df.groupby(['method', 'sparsity', 'target_bits'])
        
        results = []
        for name, group in grouped:
            method, sparsity, bits = name
            
            row = {
                'method': method,
                'sparsity': sparsity,
                'target_bits': bits,
                'wikitext2_mean': group['wikitext2_ppl'].mean(),
                'wikitext2_std': group['wikitext2_ppl'].std(),
                'ptb_mean': group['ptb_ppl'].mean(),
                'ptb_std': group['ptb_ppl'].std(),
                'c4_mean': group['c4_ppl'].mean(),
                'c4_std': group['c4_ppl'].std(),
            }
            results.append(row)
        
        df_summary = pd.DataFrame(results)
        
        # 保存摘要
        summary_file = self.output_dir / 'comprehensive_summary.csv'
        df_summary.to_csv(summary_file, index=False)
        print(f"✓ 摘要已保存: {summary_file}")
        
        # 打印表格
        print("\nWikiText2 PPL 对比:")
        print("-"*80)
        
        pivot = df_summary.pivot_table(
            values='wikitext2_mean',
            index=['method'],
            columns=['sparsity', 'target_bits']
        )
        print(pivot)
        
        print()
        
    def plot_performance_comparison(self):
        """绘制性能对比图"""
        if self.df_comprehensive is None or len(self.df_comprehensive) == 0:
            print("无数据，跳过绘图")
            return
        
        print("="*80)
        print("生成性能对比图...")
        print("="*80)
        
        df = self.df_comprehensive
        
        # 图1: 不同稀疏度下的性能对比
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics = ['wikitext2_ppl', 'ptb_ppl', 'c4_ppl']
        titles = ['WikiText2 PPL', 'PTB PPL', 'C4 PPL']
        
        for ax, metric, title in zip(axes, metrics, titles):
            # 按方法和稀疏度分组
            grouped = df.groupby(['method', 'sparsity'])[metric].mean().reset_index()
            
            for method in grouped['method'].unique():
                method_data = grouped[grouped['method'] == method]
                ax.plot(method_data['sparsity'], method_data[metric], 
                       marker='o', label=method, linewidth=2)
            
            ax.set_xlabel('Sparsity', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(f'{title} vs Sparsity', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.output_dir / 'performance_vs_sparsity.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ 图表已保存: {plot_file}")
        plt.close()
        
        # 图2: 不同比特数下的性能对比
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for ax, metric, title in zip(axes, metrics, titles):
            grouped = df.groupby(['method', 'target_bits'])[metric].mean().reset_index()
            
            for method in grouped['method'].unique():
                method_data = grouped[grouped['method'] == method]
                ax.plot(method_data['target_bits'], method_data[metric], 
                       marker='s', label=method, linewidth=2)
            
            ax.set_xlabel('Target Bits', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(f'{title} vs Target Bits', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.output_dir / 'performance_vs_bits.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ 图表已保存: {plot_file}")
        plt.close()
        
        # 图3: 热力图 - WikiText2 PPL
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 只看 enhanced_quantile 方法
        df_enhanced = df[df['method'] == 'enhanced_quantile']
        if len(df_enhanced) > 0:
            pivot = df_enhanced.pivot_table(
                values='wikitext2_ppl',
                index='sparsity',
                columns='target_bits',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax)
            ax.set_title('Enhanced Method: WikiText2 PPL Heatmap', fontsize=14)
            ax.set_xlabel('Target Bits', fontsize=12)
            ax.set_ylabel('Sparsity', fontsize=12)
            
            plt.tight_layout()
            plot_file = self.output_dir / 'heatmap_enhanced.png'
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"✓ 热力图已保存: {plot_file}")
            plt.close()
        
        print()
    
    def plot_complexity_comparison(self):
        """绘制复杂度对比图"""
        if self.df_complexity is None or len(self.df_complexity) == 0:
            print("无复杂度数据，跳过绘图")
            return
        
        print("="*80)
        print("生成复杂度对比图...")
        print("="*80)
        
        df = self.df_complexity
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # 时间对比
        ax = axes[0]
        ax.bar(df['method'], df['wall_time_sec'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Wall Time (sec)', fontsize=12)
        ax.set_title('Runtime Comparison', fontsize=14)
        ax.tick_params(axis='x', rotation=15)
        
        # 内存对比
        ax = axes[1]
        ax.bar(df['method'], df['gpu_memory_mb'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Peak Memory (MB)', fontsize=12)
        ax.set_title('Memory Comparison', fontsize=14)
        ax.tick_params(axis='x', rotation=15)
        
        # 吞吐量对比
        ax = axes[2]
        ax.bar(df['method'], df['throughput_samples_per_sec'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Throughput (samples/sec)', fontsize=12)
        ax.set_title('Throughput Comparison', fontsize=14)
        ax.tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plot_file = self.output_dir / 'complexity_comparison.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ 复杂度对比图已保存: {plot_file}")
        plt.close()
        
        print()
    
    def plot_statistical_results(self):
        """绘制统计测试结果"""
        if self.df_stats is None or len(self.df_stats) == 0:
            print("无统计数据，跳过绘图")
            return
        
        print("="*80)
        print("生成统计测试图...")
        print("="*80)
        
        df = self.df_stats
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['wikitext2_ppl', 'ptb_ppl', 'c4_ppl', 'time_sec']
        titles = ['WikiText2 PPL', 'PTB PPL', 'C4 PPL', 'Runtime (sec)']
        
        for ax, metric, title in zip(axes.flat, metrics, titles):
            # 箱线图
            data_to_plot = []
            labels = []
            
            for method in df['method'].unique():
                method_data = df[df['method'] == method][metric].dropna()
                if len(method_data) > 0:
                    data_to_plot.append(method_data)
                    labels.append(method)
            
            if len(data_to_plot) > 0:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                # 美化
                colors = ['#1f77b4', '#ff7f0e']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                
                ax.set_ylabel(title, fontsize=12)
                ax.set_title(f'{title} Distribution', fontsize=14)
                ax.grid(True, alpha=0.3, axis='y')
                ax.tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plot_file = self.output_dir / 'statistical_distribution.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ 统计分布图已保存: {plot_file}")
        plt.close()
        
        print()
    
    def generate_markdown_report(self):
        """生成 Markdown 报告"""
        print("="*80)
        print("生成 Markdown 报告...")
        print("="*80)
        
        report_file = self.output_dir / 'ANALYSIS_REPORT.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# SparseGPT 改进版本 - 全面测试报告\n\n")
            f.write("---\n\n")
            
            # 1. 概述
            f.write("## 1. 测试概述\n\n")
            f.write("本报告对比了 SparseGPT 的原版和改进版本，验证改进的有效性。\n\n")
            
            f.write("### 测试方法\n\n")
            f.write("- **原版 (original)**: 固定比特量化\n")
            f.write("- **改进版 (enhanced_quantile)**: 多维度重要性评估 + 自适应比特分配 (Quantile方法)\n")
            f.write("- **改进版 (enhanced_budget)**: 多维度重要性评估 + 自适应比特分配 (Budget方法)\n\n")
            
            # 2. 性能对比
            f.write("## 2. 性能对比\n\n")
            
            if self.df_comprehensive is not None and len(self.df_comprehensive) > 0:
                f.write("### 2.1 WikiText2 PPL 对比\n\n")
                
                # 计算摘要统计
                grouped = self.df_comprehensive.groupby(['method', 'sparsity', 'target_bits'])
                summary = grouped['wikitext2_ppl'].agg(['mean', 'std']).reset_index()
                
                f.write("| Method | Sparsity | Target Bits | WikiText2 PPL (mean ± std) |\n")
                f.write("|--------|----------|-------------|---------------------------|\n")
                
                for _, row in summary.iterrows():
                    method_str = str(row['method'])
                    f.write(f"| {method_str:20s} | {row['sparsity']:.1f} | "
                           f"{row['target_bits']:.1f} | {row['mean']:.3f} ± {row['std']:.3f} |\n")
                
                f.write("\n")
                f.write("![Performance vs Sparsity](performance_vs_sparsity.png)\n\n")
                f.write("![Performance vs Bits](performance_vs_bits.png)\n\n")
            
            # 3. 计算复杂度对比
            f.write("## 3. 计算复杂度对比\n\n")
            
            if self.df_complexity is not None and len(self.df_complexity) > 0:
                f.write("| Method | Runtime (sec) | Peak Memory (MB) | Throughput (samples/sec) |\n")
                f.write("|--------|---------------|------------------|---------------------------|\n")
                
                for _, row in self.df_complexity.iterrows():
                    f.write(f"| {row['method']:20s} | {row['wall_time_sec']:.2f} | "
                           f"{row['gpu_memory_mb']:.2f} | {row['throughput_samples_per_sec']:.3f} |\n")
                
                f.write("\n")
                f.write("![Complexity Comparison](complexity_comparison.png)\n\n")
            
            # 4. 统计显著性分析
            f.write("## 4. 统计显著性分析\n\n")
            
            if self.df_stats is not None and len(self.df_stats) > 0:
                f.write("### 4.1 多次运行结果\n\n")
                
                for metric in ['wikitext2_ppl', 'ptb_ppl', 'c4_ppl']:
                    metric_name = metric.replace('_ppl', '').upper()
                    f.write(f"#### {metric_name}\n\n")
                    
                    f.write("| Method | Mean | Std | 95% CI |\n")
                    f.write("|--------|------|-----|--------|\n")
                    
                    for method in self.df_stats['method'].unique():
                        data = self.df_stats[self.df_stats['method'] == method][metric].dropna()
                        if len(data) > 0:
                            mean = data.mean()
                            std = data.std()
                            stderr = stats.sem(data)
                            ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=stderr)
                            
                            f.write(f"| {method:20s} | {mean:.3f} | {std:.3f} | "
                                   f"[{ci[0]:.3f}, {ci[1]:.3f}] |\n")
                    
                    f.write("\n")
                
                f.write("![Statistical Distribution](statistical_distribution.png)\n\n")
            
            # 5. 结论
            f.write("## 5. 结论\n\n")
            
            if self.df_stats is not None and len(self.df_stats) > 0:
                # 计算改进幅度
                original_wt2 = self.df_stats[self.df_stats['method'] == 'original']['wikitext2_ppl'].mean()
                enhanced_wt2 = self.df_stats[self.df_stats['method'] == 'enhanced_quantile']['wikitext2_ppl'].mean()
                
                if pd.notna(original_wt2) and pd.notna(enhanced_wt2):
                    improvement = (original_wt2 - enhanced_wt2) / original_wt2 * 100
                    
                    f.write(f"- **性能改进**: WikiText2 PPL 从 {original_wt2:.3f} 降至 {enhanced_wt2:.3f} "
                           f"(改进 {improvement:.2f}%)\n")
                    
                    # T-test
                    data1 = self.df_stats[self.df_stats['method'] == 'original']['wikitext2_ppl'].dropna()
                    data2 = self.df_stats[self.df_stats['method'] == 'enhanced_quantile']['wikitext2_ppl'].dropna()
                    
                    if len(data1) > 1 and len(data2) > 1:
                        t_stat, p_value = stats.ttest_ind(data1, data2)
                        
                        f.write(f"- **统计显著性**: t={t_stat:.3f}, p-value={p_value:.4f}\n")
                        
                        if p_value < 0.05:
                            f.write("  - ✓ **改进具有统计显著性** (p < 0.05)\n")
                        else:
                            f.write("  - ✗ 改进不具有统计显著性 (p ≥ 0.05)\n")
            
            if self.df_complexity is not None and len(self.df_complexity) > 0:
                baseline = self.df_complexity[self.df_complexity['method'] == 'original'].iloc[0]
                enhanced = self.df_complexity[self.df_complexity['method'] == 'enhanced_quantile'].iloc[0]
                
                time_ratio = enhanced['wall_time_sec'] / baseline['wall_time_sec']
                mem_ratio = enhanced['gpu_memory_mb'] / baseline['gpu_memory_mb']
                
                f.write(f"- **计算开销**: 运行时间 {time_ratio:.2f}x, 内存占用 {mem_ratio:.2f}x\n")
                
                if time_ratio < 1.5:
                    f.write("  - ✓ **计算开销可接受** (<1.5x)\n")
                else:
                    f.write("  - ⚠ 计算开销较大 (>1.5x)\n")
            
            f.write("\n")
            f.write("---\n")
            f.write(f"\n*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        print(f"✓ Markdown 报告已保存: {report_file}")
        print()
    
    def run_all(self):
        """运行所有分析"""
        self.load_data()
        self.analyze_comprehensive_results()
        self.plot_performance_comparison()
        self.plot_complexity_comparison()
        self.plot_statistical_results()
        self.generate_markdown_report()
        
        print("="*80)
        print("分析完成！所有结果已保存到:", self.output_dir)
        print("="*80)


if __name__ == '__main__':
    analyzer = ResultAnalyzer()
    analyzer.run_all()

