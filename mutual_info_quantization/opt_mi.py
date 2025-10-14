"""
基于互信息的 OPT 模型测试脚本

使用 SparseGPT_MI 进行剪枝和互信息分组量化
"""

import time
import sys
import os

# 确保使用指定的GPU（如果环境变量未设置）
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from quant import Quantizer
from sparsegpt_mi import SparseGPT_MI, MIQuantizationStats
from modelutils import find_layers, DEV

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


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


@torch.no_grad()
def opt_sequential_mi(model, dataloader, dev, args):
    """使用互信息分组的逐层剪枝+量化"""
    print('Starting MI-based sequential pruning & quantization ...')
    print(f'  使用MI分组: {args.use_mi_grouping}')
    if args.use_mi_grouping:
        print(f'  分组数: {args.n_groups}')
    
    # 创建统计收集器
    stats = MIQuantizationStats()
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers
    
    # 移动嵌入层
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in is not None:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)
    
    # 捕获输入
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    
    # 移回CPU
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in is not None:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    
    print('Ready to process layers.')
    
    # 逐层处理
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        
        # 查找需要量化的子层
        subset = find_layers(layer)
        
        # 为每个子层创建 SparseGPT_MI 实例
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                continue
            
            layer_name = f"layer_{i}.{name}"
            gpts[name] = SparseGPT_MI(subset[name], layer_name=layer_name, stats_collector=stats)
            
            # 配置量化器
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )
        
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        
        # 注册hook
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        # 前向传播收集激活
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        
        # 移除hook
        for h in handles:
            h.remove()
        
        # 执行剪枝+量化
        for name in gpts:
            print(f'\nProcessing {layer_name} ({i+1}/{len(layers)}) ...')
            gpts[name].fasterprune(
                sparsity=args.sparsity,
                prunen=args.prunen,
                prunem=args.prunem,
                percdamp=args.percdamp,
                target_avg_bits=args.target_avg_bits,
                use_mi_grouping=bool(args.use_mi_grouping),
                n_groups=args.n_groups
            )
            gpts[name].free()
        
        # 更新输入
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        
        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()
        
        inps, outs = outs, inps
    
    model.config.use_cache = use_cache
    
    # 打印统计摘要
    stats.print_summary()
    
    return model


def opt_eval(model, testenc, dev):
    """评估模型"""
    print('Evaluating ...')
    
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers
    
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in is not None:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    
    layers[0] = layers[0].to(dev)
    
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
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
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in is not None:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    
    for i in range(len(layers)):
        print(f'Evaluating layer {i}')
        layer = layers[i].to(dev)
        
        # 分批处理以节省内存
        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    
    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)
    
    testenc = testenc.to(dev)
    nlls = []
    
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f'Perplexity: {ppl.item():.2f}')
    
    model.config.use_cache = use_cache
    
    return ppl.item()


if __name__ == '__main__':
    import argparse
    from datautils import *
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, help='OPT model to load')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'], help='Calibration dataset')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent dampening')
    parser.add_argument('--sparsity', type=float, default=0, help='Target sparsity')
    parser.add_argument('--prunen', type=int, default=0, help='N for N:M pruning')
    parser.add_argument('--prunem', type=int, default=0, help='M for N:M pruning')
    parser.add_argument('--wbits', type=int, default=16, help='Base weight bits')
    parser.add_argument('--minlayer', type=int, default=-1, help='Minimum layer')
    parser.add_argument('--maxlayer', type=int, default=1000, help='Maximum layer')
    parser.add_argument('--prune_only', type=str, default='', help='Prune only layers containing this string')
    parser.add_argument('--invert', action='store_true', help='Invert prune_only')
    parser.add_argument('--save', type=str, default='', help='Save compressed model')
    parser.add_argument('--log_wandb', action='store_true', help='Log to W&B')
    
    # MI相关参数
    parser.add_argument('--target_avg_bits', type=float, default=4.0, help='Target average bits')
    parser.add_argument('--use_mi_grouping', type=int, default=1, help='Use MI-based grouping (0/1)')
    parser.add_argument('--n_groups', type=int, default=10, help='Number of groups for MI clustering')
    
    args = parser.parse_args()
    
    # 加载模型
    model = get_opt(args.model)
    model.eval()
    
    # 加载数据
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    
    # 执行压缩
    if args.sparsity or args.wbits < 16:
        tick = time.time()
        opt_sequential_mi(model, dataloader, DEV, args)
        print(f'Total time: {time.time() - tick:.2f}s')
    
    # 评估
    datasets = ['wikitext2', 'ptb', 'c4']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(f'\nEvaluating on {dataset} ...')
        ppl = opt_eval(model, testloader, DEV)
        print(f'Perplexity on {dataset}: {ppl:.3f}')
    
    # 保存模型
    if args.save:
        model.save_pretrained(args.save)
        print(f'Model saved to {args.save}')

