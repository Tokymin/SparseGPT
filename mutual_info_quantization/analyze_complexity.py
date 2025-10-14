"""
计算复杂度综合分析脚本

分析压缩时间、推理时间、内存使用等指标
"""

import re
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def extract_compression_stats(log_file):
    """从日志中提取压缩统计信息"""
    if not log_file.exists():
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    stats = {}
    
    # 提取总压缩时间
    total_time_match = re.search(r'Total time:\s+([\d.]+)s', content)
    stats['total_time'] = float(total_time_match.group(1)) if total_time_match else None
    
    # 提取层级时间
    layer_times = re.findall(r'时间:\s+([\d.]+)s', content)
    if layer_times:
        layer_times = [float(t) for t in layer_times]
        stats['avg_layer_time'] = np.mean(layer_times)
        stats['max_layer_time'] = np.max(layer_times)
        stats['min_layer_time'] = np.min(layer_times)
        stats['n_layers'] = len(layer_times)
    
    # 提取MI计算时间（如果有）
    mi_times = re.findall(r'计算互信息矩阵.*?(\d+)\s*通道', content)
    stats['mi_computed'] = len(mi_times) > 0
    
    # 提取PPL结果
    ppl_matches = {
        'wikitext2': re.search(r'Perplexity on wikitext2:\s+([\d.]+)', content),
        'ptb': re.search(r'Perplexity on ptb:\s+([\d.]+)', content),
        'c4': re.search(r'Perplexity on c4:\s+([\d.]+)', content)
    }
    
    for dataset, match in ppl_matches.items():
        if match:
            stats[f'ppl_{dataset}'] = float(match.group(1))
    
    return stats


def print_comparison_table(results):
    """打印对比表格"""
    print("\n" + "="*100)
    print("计算复杂度对比分析")
    print("="*100)
    
    # 压缩时间对比
    print("\n【压缩时间对比】")
    print("-"*100)
    print(f"{'方法':<20} {'总时间(s)':<12} {'平均每层(s)':<15} {'最大层(s)':<12} {'层数':<8}")
    print("-"*100)
    
    for name, stats in results.items():
        if stats and stats.get('total_time'):
            total = f"{stats['total_time']:.2f}"
            avg = f"{stats.get('avg_layer_time', 0):.3f}"
            max_t = f"{stats.get('max_layer_time', 0):.3f}"
            n_layers = str(stats.get('n_layers', 'N/A'))
            print(f"{name:<20} {total:<12} {avg:<15} {max_t:<12} {n_layers:<8}")
    
    print("-"*100)
    
    # 计算开销
    if 'MI改进版' in results and '原版' in results:
        if results['MI改进版'].get('total_time') and results['原版'].get('total_time'):
            overhead = (results['MI改进版']['total_time'] / results['原版']['total_time'] - 1) * 100
            print(f"\nMI改进版 vs 原版: 时间开销 {overhead:+.1f}%")
    
    if 'MI改进版' in results and '增强版(无MI)' in results:
        if results['MI改进版'].get('total_time') and results['增强版(无MI)'].get('total_time'):
            overhead = (results['MI改进版']['total_time'] / results['增强版(无MI)']['total_time'] - 1) * 100
            print(f"MI改进版 vs 增强版(无MI): 时间开销 {overhead:+.1f}%")
    
    # PPL对比
    print("\n【性能对比 (PPL)】")
    print("-"*100)
    print(f"{'方法':<20} {'WikiText2':<12} {'PTB':<12} {'C4':<12}")
    print("-"*100)
    
    for name, stats in results.items():
        if stats:
            wt2 = f"{stats.get('ppl_wikitext2', 0):.3f}" if stats.get('ppl_wikitext2') else 'N/A'
            ptb = f"{stats.get('ppl_ptb', 0):.3f}" if stats.get('ppl_ptb') else 'N/A'
            c4 = f"{stats.get('ppl_c4', 0):.3f}" if stats.get('ppl_c4') else 'N/A'
            print(f"{name:<20} {wt2:<12} {ptb:<12} {c4:<12}")
    
    print("-"*100)
    
    # 性能提升
    if 'MI改进版' in results and '增强版(无MI)' in results:
        print("\n【MI改进版的性能提升】")
        for dataset in ['wikitext2', 'ptb', 'c4']:
            key = f'ppl_{dataset}'
            if results['MI改进版'].get(key) and results['增强版(无MI)'].get(key):
                baseline = results['增强版(无MI)'][key]
                improved = results['MI改进版'][key]
                improvement = (baseline - improved) / baseline * 100
                print(f"  {dataset}: {baseline:.3f} → {improved:.3f} ({improvement:+.2f}%)")
    
    # 效率分析
    print("\n【效率分析】")
    print("-"*100)
    if 'MI改进版' in results and '增强版(无MI)' in results:
        mi_stats = results['MI改进版']
        base_stats = results['增强版(无MI)']
        
        if mi_stats.get('total_time') and base_stats.get('total_time'):
            time_overhead = (mi_stats['total_time'] / base_stats['total_time'] - 1) * 100
            
            # 计算平均PPL改进
            ppl_improvements = []
            for dataset in ['wikitext2', 'ptb', 'c4']:
                key = f'ppl_{dataset}'
                if mi_stats.get(key) and base_stats.get(key):
                    improvement = (base_stats[key] - mi_stats[key]) / base_stats[key] * 100
                    ppl_improvements.append(improvement)
            
            if ppl_improvements:
                avg_ppl_improvement = np.mean(ppl_improvements)
                efficiency = avg_ppl_improvement / time_overhead if time_overhead > 0 else float('inf')
                
                print(f"时间开销: {time_overhead:+.1f}%")
                print(f"平均性能提升: {avg_ppl_improvement:+.2f}%")
                print(f"效率比 (性能提升/时间开销): {efficiency:.2f}")
                
                if efficiency > 1:
                    print("✅ 结论: MI改进版的性能提升大于时间开销，非常值得！")
                else:
                    print("⚠️  结论: MI改进版的时间开销较大，需要权衡。")
    
    print("="*100)


def save_results_csv(results, output_file):
    """保存结果到CSV"""
    with open(output_file, 'w') as f:
        # 表头
        f.write("method,total_time_s,avg_layer_time_s,max_layer_time_s,n_layers,")
        f.write("ppl_wikitext2,ppl_ptb,ppl_c4,mi_computed\n")
        
        # 数据
        for name, stats in results.items():
            if stats:
                f.write(f"{name},")
                f.write(f"{stats.get('total_time', '')},")
                f.write(f"{stats.get('avg_layer_time', '')},")
                f.write(f"{stats.get('max_layer_time', '')},")
                f.write(f"{stats.get('n_layers', '')},")
                f.write(f"{stats.get('ppl_wikitext2', '')},")
                f.write(f"{stats.get('ppl_ptb', '')},")
                f.write(f"{stats.get('ppl_c4', '')},")
                f.write(f"{stats.get('mi_computed', False)}\n")


def main():
    script_dir = Path(__file__).parent
    result_dir = script_dir / 'complexity_results'
    
    if not result_dir.exists():
        print(f"错误: 找不到结果目录 {result_dir}")
        print("请先运行 benchmark_complexity.sh")
        return
    
    # 读取日志
    log_files = {
        '原版': result_dir / 'baseline_complexity.log',
        '增强版(无MI)': result_dir / 'enhanced_no_mi_complexity.log',
        'MI改进版': result_dir / 'mi_enhanced_complexity.log'
    }
    
    results = {}
    for name, log_file in log_files.items():
        print(f"分析 {name}...")
        results[name] = extract_compression_stats(log_file)
    
    # 打印对比表格
    print_comparison_table(results)
    
    # 保存结果
    output_csv = result_dir / 'complexity_analysis.csv'
    save_results_csv(results, output_csv)
    print(f"\n详细结果已保存到: {output_csv}")


if __name__ == '__main__':
    main()

