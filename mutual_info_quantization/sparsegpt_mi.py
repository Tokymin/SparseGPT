"""
SparseGPT with Mutual Information Based Quantization Grouping

基于互信息的量化分组改进版

主要创新:
1. 保留原有的多维度重要性评估
2. 新增: 基于互信息的通道分组
3. 新增: 分组内共享量化策略
4. 新增: 跨组自适应比特分配

作者: Toky
基于: enhanced_fix_acc_version
"""

import math
import time
import torch
import torch.nn as nn
import transformers
import sys
import os

# 添加路径以导入基础模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from quant import *
from collections import defaultdict
import numpy as np

# 导入互信息模块
from mutual_info import compute_mi_matrix_fast
from channel_grouping import ChannelGrouping

DEBUG = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class MIQuantizationStats:
    """互信息量化统计信息收集器"""
    
    def __init__(self):
        self.bit_distribution = defaultdict(int)
        self.group_stats = []  # 分组统计
        self.layer_stats = []
    
    def update(self, bit_allocation, group_info, layer_name=""):
        """更新统计信息（包含分组信息）"""
        # 统计比特分布
        unique_bits, counts = torch.unique(bit_allocation, return_counts=True)
        for bit, count in zip(unique_bits.cpu().numpy(), counts.cpu().numpy()):
            self.bit_distribution[int(bit)] += int(count)
        
        # 计算平均比特数
        avg_bits = torch.mean(bit_allocation.float()).item()
        
        # 保存层级统计
        self.layer_stats.append({
            'layer_name': layer_name,
            'avg_bits': avg_bits,
            'n_groups': len(group_info) if group_info else 0,
            'bit_distribution': {int(k): int(v) for k, v in zip(unique_bits.cpu().numpy(), counts.cpu().numpy())}
        })
        
        # 保存分组统计
        if group_info:
            self.group_stats.append({
                'layer_name': layer_name,
                'groups': group_info
            })
    
    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "="*70)
        print("互信息量化统计摘要 (MI-based Quantization Statistics)")
        print("="*70)
        
        # 总体比特分布
        total_channels = sum(self.bit_distribution.values())
        print(f"\n总通道数: {total_channels}")
        print("\n比特分布:")
        for bit in sorted(self.bit_distribution.keys()):
            count = self.bit_distribution[bit]
            percentage = count / total_channels * 100
            print(f"  {bit}-bit: {count:6d} 通道 ({percentage:5.2f}%)")
        
        # 平均比特数
        if self.layer_stats:
            avg_bits_overall = sum(s['avg_bits'] for s in self.layer_stats) / len(self.layer_stats)
            avg_groups = sum(s['n_groups'] for s in self.layer_stats) / len(self.layer_stats)
            print(f"\n平均比特数: {avg_bits_overall:.3f} bits")
            print(f"平均分组数: {avg_groups:.1f} groups/layer")
        
        # 分组统计
        if self.group_stats:
            print(f"\n分组详情 (前3层):")
            for i, group_stat in enumerate(self.group_stats[:3]):
                layer_name = group_stat['layer_name']
                groups = group_stat['groups']
                print(f"\n  {layer_name}: {len(groups)} 组")
                for j, group in enumerate(groups[:5]):  # 只显示前5组
                    print(f"    组{j}: {group['size']}通道, {group.get('bits', 'N/A')}bit")
        
        print("="*70 + "\n")


class SparseGPT_MI:
    """基于互信息的 SparseGPT"""
    
    def __init__(self, layer, layer_name="", stats_collector=None):
        self.layer = layer
        self.layer_name = layer_name
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        
        # 激活统计
        self.activation_amps = None
        self.activation_variance = None
        self.activation_history = []  # 新增: 保存激活历史用于MI计算
        self.activation_count = 0
        
        # 互信息分组
        self.channel_groups = None
        self.group_bits = None
        
        # 统计收集器
        self.stats_collector = stats_collector
    
    def add_batch(self, inp, out, blocksize=1024):
        """收集激活统计 + 保存激活历史"""
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        
        # 保存激活用于MI计算（采样以节省内存）
        if len(self.activation_history) < 20:  # 最多保存20个batch（减少内存占用）
            inp_for_mi = inp.clone().detach()
            if len(inp_for_mi.shape) == 3:
                # [batch, seq, feat] -> 采样部分
                inp_for_mi = inp_for_mi[:, ::8, :]  # 每8个token采样1个（进一步减少）
            self.activation_history.append(inp_for_mi.cpu())
        
        # 激活统计
        inp_for_stats = inp.clone()
        if len(inp_for_stats.shape) == 3:
            current_amp = torch.mean(torch.abs(inp_for_stats), dim=(0, 1))
            current_var = torch.var(inp_for_stats, dim=(0, 1))
        else:
            current_amp = torch.mean(torch.abs(inp_for_stats), dim=0)
            current_var = torch.var(inp_for_stats, dim=0)
        
        if self.activation_amps is None:
            self.activation_amps = current_amp
            self.activation_variance = current_var
        else:
            alpha = 0.5
            self.activation_amps = alpha * self.activation_amps + (1 - alpha) * current_amp
            self.activation_variance = alpha * self.activation_variance + (1 - alpha) * current_var
        
        self.activation_count += tmp
        
        # Hessian 更新
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
    
    def compute_mi_grouping(self, n_groups=10, method='spectral'):
        """
        基于激活历史计算互信息并分组
        
        Args:
            n_groups: 分组数量
            method: 聚类方法
        
        Returns:
            channel_groups: 分组结果
            MI_matrix: 互信息矩阵
        """
        if len(self.activation_history) == 0:
            print(f"[{self.layer_name}] 警告: 没有激活历史，跳过MI分组")
            return None, None
        
        print(f"[{self.layer_name}] 计算互信息矩阵...")
        
        # 合并激活历史
        activations = torch.cat(self.activation_history, dim=0)  # [samples, seq_len, channels] or [samples, channels]
        
        # 计算互信息矩阵（使用快速相关系数方法）
        MI_matrix, NMI_matrix = compute_mi_matrix_fast(activations, method='correlation')
        
        # 立即清理激活历史以释放内存
        del activations
        self.activation_history = []
        torch.cuda.empty_cache()
        
        # 基于MI矩阵分组
        print(f"[{self.layer_name}] 进行通道分组 ({n_groups} 组)...")
        grouping = ChannelGrouping(n_groups=n_groups, method=method)
        
        # 计算重要性分数用于分组
        importance_scores = self.activation_amps.cpu().numpy()
        
        groups = grouping.fit(NMI_matrix, importance_scores)
        
        # 打印分组摘要
        #grouping.print_summary()
        
        self.channel_groups = grouping
        
        return grouping, MI_matrix
    
    def allocate_group_bits(self, target_avg_bits=4.0):
        """为分组分配比特数"""
        if self.channel_groups is None:
            return None
        
        print(f"[{self.layer_name}] 为分组分配比特数...")
        group_bits_dict = self.channel_groups.allocate_bits(target_avg_bits=target_avg_bits)
        
        # 转换为通道级比特分配
        bit_allocation = torch.zeros(self.columns, device=self.dev)
        
        for group_info in self.channel_groups.group_info:
            gid = group_info['group_id']
            channels = group_info['channels']
            bits = group_bits_dict[gid]
            
            for ch in channels:
                bit_allocation[ch] = bits
            
            # 更新group_info
            group_info['bits'] = bits
        
        self.group_bits = bit_allocation
        
        return bit_allocation
    
    def fasterprune(
        self,
        sparsity,
        prunen=0,
        prunem=0,
        blocksize=128,
        percdamp=.01,
        target_avg_bits=4.0,
        use_mi_grouping=True,
        n_groups=10
    ):
        """
        互信息增强的剪枝+量化
        
        新增参数:
            use_mi_grouping: 是否使用互信息分组
            n_groups: 分组数量
        """
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        
        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)
        
        tick = time.time()
        
        # Hessian 处理
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        
        Losses = torch.zeros(self.rows, device=self.dev)
        
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        
        # === 核心改进: 互信息分组量化 ===
        use_mi_quantization = (
            hasattr(self, 'quantizer') and
            use_mi_grouping and
            len(self.activation_history) > 0
        )
        
        bit_allocation = None
        
        if use_mi_quantization:
            # 计算互信息并分组
            grouping, MI_matrix = self.compute_mi_grouping(n_groups=n_groups)
            
            if grouping is not None:
                # 分配比特数
                bit_allocation = self.allocate_group_bits(target_avg_bits=target_avg_bits)
                
                # 统计比特分布
                unique_bits, counts = torch.unique(bit_allocation, return_counts=True)
                print(f"[{self.layer_name}] MI分组比特分布: ", end="")
                for bit, count in zip(unique_bits, counts):
                    print(f"{int(bit)}bit({int(count)}) ", end="")
                avg_bits = torch.mean(bit_allocation.float()).item()
                print(f"| 平均: {avg_bits:.2f} bits")
                
                # 更新统计
                if self.stats_collector is not None:
                    self.stats_collector.update(bit_allocation, grouping.group_info, self.layer_name)
        
        mask = None
        
        # 逐块剪枝+量化
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            
            # 剪枝掩码
            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1
            
            # 逐列处理
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                global_idx = i1 + i
                
                # N:M 剪枝
                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)
                
                # 剪枝
                q = w.clone()
                q[mask1[:, i]] = 0
                
                # === MI分组量化 ===
                if use_mi_quantization and bit_allocation is not None:
                    target_bits = int(bit_allocation[global_idx].item())
                    self.quantizer.maxq = torch.tensor(2 ** target_bits - 1, device=self.dev)
                    self.quantizer.find_params(q.unsqueeze(1), weight=True)
                    q = quantize(
                        q.unsqueeze(1),
                        self.quantizer.scale,
                        self.quantizer.zero,
                        self.quantizer.maxq
                    ).flatten()
                elif hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1),
                        self.quantizer.scale,
                        self.quantizer.zero,
                        self.quantizer.maxq
                    ).flatten()
                
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                
                # 误差传播
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            
            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        
        torch.cuda.synchronize()
        elapsed_time = time.time() - tick
        total_error = torch.sum(Losses).item()
        
        print(f'[{self.layer_name}] 时间: {elapsed_time:.2f}s | 误差: {total_error:.4f}')
        
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
    
    def free(self):
        """释放资源"""
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.activation_amps = None
        self.activation_variance = None
        self.activation_history = []
        torch.cuda.empty_cache()

