"""
对比基准模型和MI量化模型的性能
"""

import json
import os
import re

def extract_compressed_results(log_file):
    """从test_fix.log中提取压缩模型的结果"""
    results = {}
    
    if not os.path.exists(log_file):
        print(f"警告: 找不到文件 {log_file}")
        return results
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取总时间
    time_match = re.search(r'Total time:\s+([\d.]+)s', content)
    if time_match:
        results['total_time'] = float(time_match.group(1))
    
    # 提取平均比特数
    bits_match = re.search(r'平均比特数:\s+([\d.]+)\s+bits', content)
    if bits_match:
        results['avg_bits'] = float(bits_match.group(1))
    
    # 提取各数据集的困惑度
    datasets = {
        'wikitext2': r'Perplexity on wikitext2:\s+([\d.]+)',
        'ptb': r'Perplexity on ptb:\s+([\d.]+)',
        'c4': r'Perplexity on c4:\s+([\d.]+)'
    }
    
    for dataset, pattern in datasets.items():
        match = re.search(pattern, content)
        if match:
            results[dataset] = float(match.group(1))
    
    # 提取比特分配统计
    bit_dist_match = re.search(
        r'2-bit:\s+(\d+)\s+通道.*?\n.*?'
        r'3-bit:\s+(\d+)\s+通道.*?\n.*?'
        r'4-bit:\s+(\d+)\s+通道.*?\n.*?'
        r'6-bit:\s+(\d+)\s+通道.*?\n.*?'
        r'8-bit:\s+(\d+)\s+通道',
        content, re.DOTALL
    )
    if bit_dist_match:
        results['bit_distribution'] = {
            '2bit': int(bit_dist_match.group(1)),
            '3bit': int(bit_dist_match.group(2)),
            '4bit': int(bit_dist_match.group(3)),
            '6bit': int(bit_dist_match.group(4)),
            '8bit': int(bit_dist_match.group(5))
        }
    
    return results


def load_baseline_results(json_file):
    """加载基准测试结果"""
    if not os.path.exists(json_file):
        print(f"警告: 找不到基准测试结果文件 {json_file}")
        return None
    
    with open(json_file, 'r') as f:
        return json.load(f)


def print_comparison(baseline, compressed):
    """打印对比结果"""
    
    print("="*100)
    print(" "*35 + "Vicuna-13B 性能对比分析")
    print("="*100)
    print()
    
    # 1. 模型配置对比
    print("📊 模型配置对比")
    print("-"*100)
    print(f"{'配置项':<30} {'基准模型 (FP16)':<30} {'MI量化模型':<30}")
    print("-"*100)
    print(f"{'精度':<30} {'16-bit (FP16)':<30} {f'{compressed.get(\"avg_bits\", \"N/A\"):.2f}-bit (平均)':<30}")
    print(f"{'稀疏度':<30} {'0% (dense)':<30} {'50% (sparse)':<30}")
    print(f"{'压缩方法':<30} {'无':<30} {'MI分组量化 + 剪枝':<30}")
    print("-"*100)
    print()
    
    # 2. 困惑度对比
    print("📈 困惑度 (Perplexity) 对比")
    print("-"*100)
    print(f"{'数据集':<20} {'基准 (FP16)':<20} {'压缩后':<20} {'变化':<20} {'性能保持率':<20}")
    print("-"*100)
    
    for dataset in ['wikitext2', 'ptb', 'c4']:
        if baseline and dataset in baseline:
            base_ppl = baseline[dataset]['ppl']
            comp_ppl = compressed.get(dataset, 0)
            
            if comp_ppl > 0:
                change = comp_ppl - base_ppl
                change_pct = (change / base_ppl) * 100
                retention = 100 - abs(change_pct)
                
                # 判断性能等级
                if retention >= 95:
                    rating = "⭐⭐⭐⭐⭐ 优秀"
                elif retention >= 90:
                    rating = "⭐⭐⭐⭐ 良好"
                elif retention >= 85:
                    rating = "⭐⭐⭐ 一般"
                else:
                    rating = "⭐⭐ 较差"
                
                print(f"{dataset.upper():<20} {base_ppl:<20.3f} {comp_ppl:<20.3f} "
                      f"{change:+.3f} ({change_pct:+.2f}%){'':<5} {retention:.1f}% {rating}")
            else:
                print(f"{dataset.upper():<20} {base_ppl:<20.3f} {'N/A':<20} {'N/A':<20} {'N/A':<20}")
    
    print("-"*100)
    print()
    
    # 3. 压缩效果分析
    if 'avg_bits' in compressed and 'bit_distribution' in compressed:
        print("🗜️  压缩效果分析")
        print("-"*100)
        
        avg_bits = compressed['avg_bits']
        bit_dist = compressed['bit_distribution']
        total_channels = sum(bit_dist.values())
        
        # 理论压缩率 (相对于FP16)
        compression_ratio_bits = 16.0 / avg_bits
        
        # 考虑稀疏度的实际压缩率
        sparsity = 0.5  # 50%稀疏度
        effective_bits = avg_bits * (1 - sparsity)  # 稀疏权重不存储
        compression_ratio_total = 16.0 / effective_bits
        
        print(f"平均量化位宽: {avg_bits:.2f} bits")
        print(f"目标位宽: 4.0 bits")
        print(f"稀疏度: 50%")
        print()
        print(f"压缩率:")
        print(f"  - 仅量化: {compression_ratio_bits:.2f}x (16bit → {avg_bits:.2f}bit)")
        print(f"  - 量化+剪枝: {compression_ratio_total:.2f}x (考虑50%稀疏度)")
        print()
        print(f"比特分配分布:")
        for bit_level in ['2bit', '3bit', '4bit', '6bit', '8bit']:
            count = bit_dist.get(bit_level, 0)
            pct = (count / total_channels * 100) if total_channels > 0 else 0
            bar_len = int(pct / 2)  # 缩放到50字符宽度
            bar = '█' * bar_len
            print(f"  {bit_level}: {bar:<50} {count:>10} ({pct:>5.2f}%)")
        
        print("-"*100)
        print()
    
    # 4. 时间开销
    if baseline and compressed.get('total_time'):
        print("⏱️  计算开销对比")
        print("-"*100)
        
        # 基准模型评估时间（3个数据集）
        base_eval_time = sum(baseline[ds]['time'] for ds in ['wikitext2', 'ptb', 'c4'] if ds in baseline)
        
        # 压缩模型总时间（包含压缩+评估）
        comp_total_time = compressed['total_time']
        
        print(f"基准模型评估时间: {base_eval_time:.2f}秒 ({base_eval_time/60:.1f}分钟)")
        print(f"压缩+评估总时间: {comp_total_time:.2f}秒 ({comp_total_time/60:.1f}分钟)")
        print()
        print(f"注: 压缩是一次性开销，压缩后的模型推理速度会更快")
        print("-"*100)
        print()
    
    # 5. 总体评价
    print("🎯 总体评价")
    print("-"*100)
    
    # 计算平均性能保持率
    if baseline:
        retentions = []
        for dataset in ['wikitext2', 'ptb', 'c4']:
            if dataset in baseline and dataset in compressed:
                base_ppl = baseline[dataset]['ppl']
                comp_ppl = compressed[dataset]
                if comp_ppl > 0:
                    change_pct = abs((comp_ppl - base_ppl) / base_ppl * 100)
                    retention = 100 - change_pct
                    retentions.append(retention)
        
        if retentions:
            avg_retention = sum(retentions) / len(retentions)
            
            print(f"平均性能保持率: {avg_retention:.1f}%")
            
            if compressed.get('avg_bits'):
                avg_bits = compressed['avg_bits']
                compression_ratio = 16.0 / avg_bits * 2  # 考虑50%稀疏度
                
                print(f"总体压缩率: {compression_ratio:.2f}x")
                print()
                
                # 评价
                if avg_retention >= 95 and compression_ratio >= 3:
                    rating = "⭐⭐⭐⭐⭐ 优秀"
                    comment = "在保持高性能的同时实现了显著的模型压缩"
                elif avg_retention >= 90 and compression_ratio >= 2:
                    rating = "⭐⭐⭐⭐ 良好"
                    comment = "性能和压缩率取得了较好平衡"
                elif avg_retention >= 85:
                    rating = "⭐⭐⭐ 一般"
                    comment = "有一定压缩效果，但性能下降明显"
                else:
                    rating = "⭐⭐ 需要改进"
                    comment = "性能下降较大，需要优化压缩策略"
                
                print(f"综合评级: {rating}")
                print(f"评价: {comment}")
    
    print("-"*100)
    print()
    
    # 6. 改进建议
    if compressed.get('bit_distribution'):
        bit_dist = compressed['bit_distribution']
        total = sum(bit_dist.values())
        pct_8bit = (bit_dist.get('8bit', 0) / total * 100) if total > 0 else 0
        pct_4bit = (bit_dist.get('4bit', 0) / total * 100) if total > 0 else 0
        
        if pct_8bit > 80:
            print("💡 改进建议")
            print("-"*100)
            print(f"⚠️  当前 {pct_8bit:.1f}% 的通道使用 8-bit，压缩率不足")
            print()
            print("建议:")
            print("  1. 调整比特分配策略，使用更激进的阈值")
            print("  2. 增加校准样本数 (16 → 128)")
            print("  3. 增加分组数 (5 → 10-15)")
            print("  4. 修改 channel_grouping.py 中的比特分配逻辑")
            print()
            print("详细优化建议请查看: 测试结果分析.md")
            print("-"*100)
    
    print()
    print("="*100)


def main():
    # 文件路径
    baseline_file = "baseline_results.json"
    compressed_log = "test_fix.log"
    
    print("\n正在加载测试结果...")
    
    # 加载结果
    baseline = load_baseline_results(baseline_file)
    compressed = extract_compressed_results(compressed_log)
    
    if not baseline:
        print("\n⚠️  基准测试结果未找到!")
        print("请先运行: ./test_baseline.sh")
        print()
        return
    
    if not compressed:
        print("\n⚠️  压缩模型测试结果未找到!")
        print("请检查 test_fix.log 文件")
        print()
        return
    
    # 打印对比
    print_comparison(baseline, compressed)
    
    # 保存对比结果
    comparison = {
        'baseline': baseline,
        'compressed': compressed,
        'timestamp': __import__('datetime').datetime.now().isoformat()
    }
    
    with open('comparison_results.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print("对比结果已保存到: comparison_results.json")
    print()


if __name__ == '__main__':
    main()

