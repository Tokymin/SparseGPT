#!/bin/bash
# 修复脚本：重新从日志文件中提取结果

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULT_DIR="$SCRIPT_DIR/comprehensive_results"

CSV_FILE="$RESULT_DIR/results.csv"

echo "重新提取测试结果..."
echo "method,sparsity,target_bits,run_id,wikitext2_ppl,ptb_ppl,c4_ppl,time_sec,peak_mem_gb,avg_bits" > $CSV_FILE

count=0
for log_file in $RESULT_DIR/*.log; do
    if [ -f "$log_file" ]; then
        # 从文件名提取信息
        filename=$(basename "$log_file" .log)
        
        # 解析文件名: method_sp{sparsity}_bits{bits}_run{run_id}
        if [[ $filename =~ ^(.+)_sp([0-9.]+)_bits([0-9.]+)_run([0-9]+)$ ]]; then
            method="${BASH_REMATCH[1]}"
            sparsity="${BASH_REMATCH[2]}"
            bits="${BASH_REMATCH[3]}"
            run_id="${BASH_REMATCH[4]}"
            
            # 提取 PPL 结果
            wikitext2=""
            ptb=""
            c4=""
            
            # 尝试两种格式
            # 格式1: "Perplexity on wikitext2: 36.186" (增强版)
            wt2_line=$(grep "Perplexity on wikitext2:" "$log_file" | tail -1)
            if [ ! -z "$wt2_line" ]; then
                wikitext2=$(echo "$wt2_line" | awk '{print $NF}')
            else
                # 格式2: "Perplexity: 39.108929" (原版)
                wt2_line=$(grep -A 20 "^wikitext2$" "$log_file" | grep "^Perplexity:" | head -1)
                if [ ! -z "$wt2_line" ]; then
                    wikitext2=$(echo "$wt2_line" | awk '{print $2}')
                fi
            fi
            
            # PTB
            ptb_line=$(grep "Perplexity on ptb:" "$log_file" | tail -1)
            if [ ! -z "$ptb_line" ]; then
                ptb=$(echo "$ptb_line" | awk '{print $NF}')
            else
                ptb_line=$(grep -A 20 "^ptb$" "$log_file" | grep "^Perplexity:" | head -1)
                if [ ! -z "$ptb_line" ]; then
                    ptb=$(echo "$ptb_line" | awk '{print $2}')
                fi
            fi
            
            # C4
            c4_line=$(grep "Perplexity on c4:" "$log_file" | tail -1)
            if [ ! -z "$c4_line" ]; then
                c4=$(echo "$c4_line" | awk '{print $NF}')
            else
                c4_line=$(grep -A 20 "^c4$" "$log_file" | grep "^Perplexity:" | head -1)
                if [ ! -z "$c4_line" ]; then
                    c4=$(echo "$c4_line" | awk '{print $2}')
                fi
            fi
            
            # 写入 CSV
            echo "$method,$sparsity,$bits,$run_id,$wikitext2,$ptb,$c4,,,," >> $CSV_FILE
            
            count=$((count + 1))
            if [ ! -z "$wikitext2" ]; then
                echo "[$count] $filename: WikiText2=$wikitext2, PTB=$ptb, C4=$c4"
            else
                echo "[$count] $filename: 未找到结果"
            fi
        fi
    fi
done

echo ""
echo "完成！共处理 $count 个日志文件"
echo "结果已保存到: $CSV_FILE"
echo ""
echo "查看结果:"
echo "  head -20 $CSV_FILE"

