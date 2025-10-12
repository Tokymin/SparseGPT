import time
import sys
import os

# 添加父目录到路径，以便导入其他模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from quant_toky import Quantizer
from enhanced_version.sparsegpt_enhanced import SparseGPT, QuantizationStats
from modelutils import find_layers

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False


def get_opt(model_name):
    """加载OPT模型并跳过冗余初始化"""
    import torch
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
def opt_sequential(model, dataloader, dev, args):
    """逐层执行SparseGPT剪枝+增强型混合精度量化"""
    print('Starting sequential pruning & quantization ...')
    
    # 创建统计收集器
    stats = QuantizationStats()

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    # 移动嵌入层到设备以捕获输入：修复project_out/project_in的None判断
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    # 关键修复1：增加 "is not None" 判断，避免对None调用.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in is not None:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    # 初始化输入缓存
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    # 自定义Catcher类：捕获第一层输入
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError("Trigger input capture")

    layers[0] = Catcher(layers[0])

    # 执行前向传播以捕获输入
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module  # 恢复原始层

    # 移回CPU释放内存：同样修复None判断
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    # 关键修复2：增加 "is not None" 判断
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in is not None:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready for layer-wise processing.')

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        # 筛选目标层（线性层）
        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                continue
            layer_name = f"layer{i}.{name}"
            gpts[name] = SparseGPT(subset[name], layer_name=layer_name, stats_collector=stats)
            # 量化器配置：整合三大改进
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits,
                    perchannel=True,
                    sym=False,
                    mse=False,
                    use_mutual_info=args.use_mutual_info,  # 互信息分组
                    num_clusters=args.num_clusters,  # 聚类数量
                    dynamic_scaling=args.dynamic_scaling  # 动态通道缩放
                )

        # 注册前向钩子，收集激活统计
        def add_batch_hook(name):
            def hook(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return hook

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch_hook(name)))

        # 前向传播，触发钩子收集统计
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        # 执行剪枝+量化
        for name in gpts:
            print(f"Processing layer {i}, component {name}")
            print('Pruning with SparseGPT ...')
            gpts[name].fasterprune(
                args.sparsity,
                prunen=args.prunen,
                prunem=args.prunem,
                percdamp=args.percdamp,
                blocksize=args.blocksize,
                target_avg_bits=args.target_avg_bits,
                bit_allocation_method=args.bit_method
            )
            gpts[name].free()

        # 验证剪枝后输出
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        # 移回CPU并释放内存
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps  # 交换输入输出，供下一层使用

    # 打印量化统计摘要
    print("\n" + "="*80)
    stats.print_summary()
    print("="*80 + "\n")
    
    model.config.use_cache = use_cache


@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, args, log_wandb: bool = False):
    """评估压缩后模型的困惑度（Perplexity）"""
    print(f'Evaluating on {dataset} ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    # 移动嵌入层到设备以捕获输入：修复None判断
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    # 关键修复3：增加 "is not None" 判断
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in is not None:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    # 初始化输入缓存
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    # 自定义Catcher类：捕获评估输入
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError("Trigger eval input capture")

    layers[0] = Catcher(layers[0])

    # 执行前向传播，捕获评估输入
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module  # 恢复原始层

    # 移回CPU释放内存：修复None判断
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    # 关键修复4：增加 "is not None" 判断
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in is not None:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    # 逐层前向传播，生成输出
    for i in range(len(layers)):
        print(f"Evaluating layer {i}")
        layer = layers[i].to(dev)

        if args.gmp:
            # GMP基线剪枝（对比实验用）
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        # 前向传播，生成当前层输出
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps  # 交换输入输出，供下一层使用

    # 计算困惑度（Perplexity）：修复project_out的None判断
    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    # 关键修复5：增加 "is not None" 判断
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)
    testenc = testenc.to(dev)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        # 关键修复6：增加 "is not None" 判断
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity on {dataset}: {ppl.item():.3f}")
    if log_wandb:
        wandb.log({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache
    return ppl.item()


if __name__ == '__main__':
    import argparse
    from datautils import get_loaders  # 需确保datautils提供get_loaders函数

    parser = argparse.ArgumentParser()

    # 原有核心参数
    parser.add_argument(
        'model', type=str,
        help='OPT model to load (e.g., `facebook/opt-125m`).'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Dataset for calibration/evaluation.'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed for data sampling.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=0.01,
        help='Hessian dampening percentage.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0.0,
        help='Target sparsity ratio (0.0-1.0).'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M structured pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M structured pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Block size for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Use GMP (Global Magnitude Pruning) baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Bit width for quantization (set <16 to enable).'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Start pruning from this layer index.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Stop pruning before this layer index.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers containing this string.'
    )
    parser.add_argument(
        '--invert', action='store_true',
        help='Invert layer selection for pruning.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Path to save the compressed model.'
    )
    parser.add_argument(
        '--log_wandb', action='store_true',
        help='Log results to Weights & Biases.'
    )

    # 新增：改进型量化参数
    parser.add_argument(
        '--use-mutual-info', action='store_true',
        help='Enable mutual information-based quantization grouping.'
    )
    parser.add_argument(
        '--num-clusters', type=int, default=4,
        help='Number of clusters for mutual information grouping.'
    )
    parser.add_argument(
        '--dynamic-scaling', action='store_true',
        help='Enable dynamic channel scaling for quantization.'
    )
    
    # 增强版参数
    parser.add_argument(
        '--target_avg_bits', type=float, default=4.0,
        help='Target average bits for mixed-precision quantization (default: 4.0).'
    )
    parser.add_argument(
        '--bit_method', type=str, default='quantile', choices=['quantile', 'budget'],
        help='Bit allocation method: quantile (fast, balanced) or budget (precise control).'
    )

    args = parser.parse_args()

    # 初始化W&B（若启用）
    if args.log_wandb:
        assert has_wandb, "wandb not installed. Run `pip install wandb`."
        wandb.init(config=args, project="improved-quant-pruning")

    # 设备配置
    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEV}")

    # 加载模型
    model = get_opt(args.model)
    model.eval()

    # 加载校准数据
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        model=args.model, seqlen=model.seqlen
    )

    # 执行剪枝+量化
    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        opt_sequential(model, dataloader, DEV, args)
        # 统计稀疏度
        sparse_stats = {}
        for n, p in model.named_parameters():
            if 'weight' in n:
                sparse_ratio = torch.mean((p == 0).float()).item()
                sparse_stats[n] = sparse_ratio
                print(f"Layer {n}: {sparse_ratio * 100:.2f}% sparse")
                if 'fc2' in n:
                    break
        print(f"Total pruning time: {time.time() - tick:.2f}s")
        if args.log_wandb:
            wandb.log({"pruning_time": time.time() - tick, "sparsity": sparse_stats})

    # 多数据集评估
    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(f"Evaluating on {dataset}")
        ppl = opt_eval(model, testloader, DEV, dataset, args, args.log_wandb)
        if args.log_wandb:
            wandb.log({f"{dataset}_perplexity": ppl})

    # 保存压缩模型
    if args.save:
        print(f"Saving model to {args.save}")
        model.save_pretrained(args.save)