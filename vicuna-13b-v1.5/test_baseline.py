"""
测试未压缩的Vicuna-13B基准性能
"""

import time
import sys
import os

# 确保使用指定的GPU
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

# 添加路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from modelutils import DEV

def get_vicuna(model_name):
    """加载Vicuna模型"""
    print(f"Loading Vicuna model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.eval()
    
    # 自动检测序列长度
    if hasattr(model.config, 'max_position_embeddings'):
        model.seqlen = model.config.max_position_embeddings
    else:
        model.seqlen = 2048
    
    return model


@torch.no_grad()
def vicuna_eval(model, testenc, dev):
    """评估Vicuna模型（基准版本 - 未压缩）"""
    print('Evaluating ...')
    
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)
    
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    
    for i in range(len(layers)):
        print(f'Evaluating layer {i}/{len(layers)}')
        layer = layers[i].to(dev)
        
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    
    # LLaMA/Vicuna使用 model.model.norm
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)
    
    testenc = testenc.to(dev)
    nlls = []
    
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
        
        # 清理
        del hidden_states, lm_logits, shift_logits, shift_labels
        torch.cuda.empty_cache()
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f'Perplexity: {ppl.item():.2f}')
    
    model.config.use_cache = use_cache
    
    return ppl.item()


if __name__ == '__main__':
    import argparse
    from datautils import get_loaders
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Vicuna model to load')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    args = parser.parse_args()
    
    # 加载模型
    model = get_vicuna(args.model)
    
    print("\n" + "="*80)
    print("测试未压缩的Vicuna-13B基准性能")
    print("="*80)
    print(f"模型: {args.model}")
    print(f"参数量: ~13B")
    print(f"精度: FP16")
    print(f"序列长度: {model.seqlen}")
    print("="*80 + "\n")
    
    # 评估
    datasets = ['wikitext2', 'ptb', 'c4']
    results = {}
    
    for dataset in datasets:
        print(f'\n{"="*80}')
        print(f'评估数据集: {dataset}')
        print("="*80)
        
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        
        tick = time.time()
        ppl = vicuna_eval(model, testloader, DEV)
        eval_time = time.time() - tick
        
        results[dataset] = {
            'ppl': ppl,
            'time': eval_time
        }
        
        print(f'\n结果: Perplexity on {dataset}: {ppl:.3f}')
        print(f'评估时间: {eval_time:.2f}秒')
        
        # 清理
        del dataloader, testloader
        torch.cuda.empty_cache()
    
    # 打印总结
    print("\n" + "="*80)
    print("基准测试结果总结")
    print("="*80)
    print(f"{'数据集':<15} {'困惑度 (PPL)':<20} {'评估时间':<15}")
    print("-"*80)
    for dataset in datasets:
        ppl = results[dataset]['ppl']
        eval_time = results[dataset]['time']
        print(f"{dataset:<15} {ppl:<20.3f} {eval_time:.2f}秒")
    print("="*80)
    
    # 保存结果
    import json
    result_file = "baseline_results.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {result_file}")

