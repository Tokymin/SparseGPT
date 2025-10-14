"""
快速分析已有的测试结果
"""

import re
from pathlib import Path

def extract_stats(log_file):
    """提取统计信息"""
    if not log_file.exists():
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    stats = {}
    
    # 提取总时间
    total_match = re.search(r'Total time:\s+([\d.]+)s', content)
    stats['total_time'] = float(total_match.group(1)) if total_match else None
    
    # 提取层级时间
    layer_times = re.findall(r'时间:\s+([\d.]+)s', content)
    if layer_times:
        layer_times = [float(t) for t in layer_times]
        stats['avg_layer_time'] = sum(layer_times) / len(layer_times)
        stats['n_layers'] = len(layer_times)
    
    # 提取PPL
    wt2 = re.search(r'Perplexity on wikitext2:\s+([\d.]+)', content)
    ptb = re.search(r'Perplexity on ptb:\s+([\d.]+)', content)
    c4 = re.search(r'Perplexity on c4:\s+([\d.]+)', content)
    
    if wt2: stats['ppl_wikitext2'] = float(wt2.group(1))
    if ptb: stats['ppl_ptb'] = float(ptb.group(1))
    if c4: stats['ppl_c4'] = float(c4.group(1))
    
    return stats

# 分析test_results目录
result_dir = Path('test_results')

logs = {
    '原版': result_dir / 'baseline_original.log',
    '增强版(无MI)': result_dir / 'enhanced_no_mi.log',
    'MI改进版': result_dir / 'mi_enhanced.log'
}

results = {}
for name, log_file in logs.items():
    results[name] = extract_stats(log_file)

# 打印结果
print("="*90)
print("测试结果分析")
print("="*90)

# 时间对比
print("\n【压缩时间】")
print("-"*90)
print(f"{'方法':<20} {'总时间(s)':<15} {'平均每层(s)':<15} {'层数':<10}")
print("-"*90)
for name, stats in results.items():
    if stats and stats.get('total_time'):
        total = f"{stats['total_time']:.2f}"
        avg = f"{stats.get('avg_layer_time', 0):.3f}"
        n = str(stats.get('n_layers', 'N/A'))
        print(f"{name:<20} {total:<15} {avg:<15} {n:<10}")
print("-"*90)

# PPL对比
print("\n【性能对比 (PPL - 越低越好)】")
print("-"*90)
print(f"{'方法':<20} {'WikiText2':<15} {'PTB':<15} {'C4':<15}")
print("-"*90)
for name, stats in results.items():
    if stats:
        wt2 = f"{stats.get('ppl_wikitext2', 0):.3f}" if stats.get('ppl_wikitext2') else 'N/A'
        ptb = f"{stats.get('ppl_ptb', 0):.3f}" if stats.get('ppl_ptb') else 'N/A'
        c4 = f"{stats.get('ppl_c4', 0):.3f}" if stats.get('ppl_c4') else 'N/A'
        print(f"{name:<20} {wt2:<15} {ptb:<15} {c4:<15}")
print("-"*90)

# 改进分析
print("\n【MI改进版 vs 增强版(无MI)】")
print("-"*90)

mi = results.get('MI改进版')
base = results.get('增强版(无MI)')

if mi and base:
    # 时间开销
    if mi.get('total_time') and base.get('total_time'):
        time_overhead = (mi['total_time'] / base['total_time'] - 1) * 100
        print(f"时间开销: {time_overhead:+.1f}%")
    
    # PPL改进
    print("\nPPL改进:")
    improvements = []
    for dataset in ['wikitext2', 'ptb', 'c4']:
        key = f'ppl_{dataset}'
        if mi.get(key) and base.get(key):
            baseline_ppl = base[key]
            improved_ppl = mi[key]
            improvement = (baseline_ppl - improved_ppl) / baseline_ppl * 100
            improvements.append(improvement)
            print(f"  {dataset:10s}: {baseline_ppl:.3f} → {improved_ppl:.3f} ({improvement:+.2f}%)")
    
    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        print(f"\n平均PPL改进: {avg_improvement:+.2f}%")
        
        if mi.get('total_time') and base.get('total_time'):
            efficiency = avg_improvement / time_overhead if time_overhead > 0 else float('inf')
            print(f"效率比 (性能提升/时间开销): {efficiency:.2f}")
            
            print("\n结论:")
            if efficiency > 1:
                print("  ✅ MI改进版的性能提升远大于时间开销，非常值得！")
            else:
                print("  ⚠️  MI改进版的时间开销较大，需要权衡。")

print("="*90)

