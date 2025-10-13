#!/usr/bin/env python3
"""
从日志文件中提取测试结果
"""

import re
import os
from pathlib import Path

def extract_ppl_from_log(log_file):
    """从日志文件中提取 PPL 结果"""
    wikitext2 = None
    ptb = None
    c4 = None
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # 尝试格式1: "Perplexity on wikitext2: 36.186" (增强版)
            match = re.search(r'Perplexity on wikitext2:\s+([\d.]+)', content)
            if match:
                wikitext2 = match.group(1)
            
            match = re.search(r'Perplexity on ptb:\s+([\d.]+)', content)
            if match:
                ptb = match.group(1)
            
            match = re.search(r'Perplexity on c4:\s+([\d.]+)', content)
            if match:
                c4 = match.group(1)
            
            # 如果没找到，尝试格式2: "Perplexity: 39.108929" (原版)
            if not wikitext2:
                # 查找 wikitext2 部分后的 Perplexity
                match = re.search(r'wikitext2\s+Evaluating.*?Perplexity:\s+([\d.]+)', content, re.DOTALL)
                if match:
                    wikitext2 = match.group(1)
            
            if not ptb:
                match = re.search(r'ptb\s+Evaluating.*?Perplexity:\s+([\d.]+)', content, re.DOTALL)
                if match:
                    ptb = match.group(1)
            
            if not c4:
                match = re.search(r'c4\s+Evaluating.*?Perplexity:\s+([\d.]+)', content, re.DOTALL)
                if match:
                    c4 = match.group(1)
    
    except Exception as e:
        print(f"错误读取 {log_file}: {e}")
    
    return wikitext2, ptb, c4


def main():
    script_dir = Path(__file__).parent
    result_dir = script_dir / 'comprehensive_results'
    csv_file = result_dir / 'results.csv'
    
    print("重新提取测试结果...")
    
    # 写入 CSV 头
    with open(csv_file, 'w') as f:
        f.write("method,sparsity,target_bits,run_id,wikitext2_ppl,ptb_ppl,c4_ppl,time_sec,peak_mem_gb,avg_bits\n")
    
    count = 0
    success_count = 0
    
    # 遍历所有日志文件
    for log_file in sorted(result_dir.glob('*.log')):
        filename = log_file.stem
        
        # 解析文件名: method_sp{sparsity}_bits{bits}_run{run_id}
        match = re.match(r'^(.+)_sp([\d.]+)_bits([\d.]+)_run(\d+)$', filename)
        if not match:
            continue
        
        method, sparsity, bits, run_id = match.groups()
        
        # 提取 PPL
        wikitext2, ptb, c4 = extract_ppl_from_log(log_file)
        
        # 写入 CSV
        with open(csv_file, 'a') as f:
            f.write(f"{method},{sparsity},{bits},{run_id},{wikitext2 or ''},{ptb or ''},{c4 or ''},,,\n")
        
        count += 1
        if wikitext2:
            success_count += 1
            print(f"[{count}] {filename}: WikiText2={wikitext2}, PTB={ptb}, C4={c4}")
        else:
            print(f"[{count}] {filename}: 未找到结果")
    
    print(f"\n完成！共处理 {count} 个日志文件，成功提取 {success_count} 个")
    print(f"结果已保存到: {csv_file}")
    print(f"\n查看结果:")
    print(f"  head -20 {csv_file}")


if __name__ == '__main__':
    main()

