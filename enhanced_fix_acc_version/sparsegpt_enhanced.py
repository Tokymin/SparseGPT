"""
SparseGPT Enhanced Version - 改进2: 激活感知的量化精度分配增强版

主要改进:
1. 多维度重要性评估（激活、Hessian、权重幅值、输出敏感度）
2. 精细化比特分配（2/3/4/6/8 bit，5档）
3. 自适应分位数阈值（替代固定阈值）
4. 动态比特预算约束
5. 详细的性能统计和可视化

作者: Toky
基于: SparseGPT (Frantar et al., 2023)
"""

import math
import time
import torch
import torch.nn as nn
import transformers
from quant import *
from collections import defaultdict

DEBUG = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class QuantizationStats:
    """量化统计信息收集器"""
    
    def __init__(self):
        self.bit_distribution = defaultdict(int)  # 每个比特数的通道数
        self.importance_scores = []  # 重要性分数历史
        self.layer_stats = []  # 每层统计
    
    def update(self, bit_allocation, importance_scores, layer_name=""):
        """更新统计信息"""
        # 统计比特分布
        unique_bits, counts = torch.unique(bit_allocation, return_counts=True)
        for bit, count in zip(unique_bits.cpu().numpy(), counts.cpu().numpy()):
            self.bit_distribution[int(bit)] += int(count)
        
        # 保存重要性分数
        self.importance_scores.append(importance_scores.cpu())
        
        # 计算平均比特数
        avg_bits = torch.mean(bit_allocation.float()).item()
        
        # 保存层级统计
        self.layer_stats.append({
            'layer_name': layer_name,
            'avg_bits': avg_bits,
            'bit_distribution': {int(k): int(v) for k, v in zip(unique_bits.cpu().numpy(), counts.cpu().numpy())},
            'importance_range': (importance_scores.min().item(), importance_scores.max().item())
        })
    
    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "="*60)
        print("量化统计摘要 (Quantization Statistics Summary)")
        print("="*60)
        
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
            print(f"\n平均比特数: {avg_bits_overall:.3f} bits")
        
        # 每层详细信息
        if len(self.layer_stats) <= 10:  # 只显示前10层
            print("\n每层统计:")
            for i, stats in enumerate(self.layer_stats):
                print(f"  Layer {i}: avg={stats['avg_bits']:.2f} bits, "
                      f"importance_range=({stats['importance_range'][0]:.3f}, {stats['importance_range'][1]:.3f})")
        
        print("="*60 + "\n")


class SparseGPT:
    """增强版 SparseGPT - 改进的激活感知量化"""
    
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
        
        # 激活感知相关
        self.activation_amps = None  # 激活幅值
        self.activation_variance = None  # 激活方差（新增）
        self.activation_count = 0
        
        # 统计收集器
        self.stats_collector = stats_collector

    def add_batch(self, inp, out, blocksize=1024):
        """
        增强版批次添加: 收集多维度激活统计信息
        
        新增统计:
        - 激活幅值 (平均绝对值)
        - 激活方差 (动态范围)
        """
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        
        # 保存原始输入用于统计
        inp_for_stats = inp.clone()
        
        # === 新增: 多维度激活统计 ===
        if len(inp_for_stats.shape) == 3:
            # [batch, seq, feat] -> [feat]
            current_amp = torch.mean(torch.abs(inp_for_stats), dim=(0, 1))
            current_var = torch.var(inp_for_stats, dim=(0, 1))
        else:
            # [batch, feat] -> [feat]
            current_amp = torch.mean(torch.abs(inp_for_stats), dim=0)
            current_var = torch.var(inp_for_stats, dim=0)
        
        # 激活幅值的指数移动平均
        if self.activation_amps is None:
            self.activation_amps = current_amp
            self.activation_variance = current_var
        else:
            alpha = 0.5  # EMA 系数
            self.activation_amps = alpha * self.activation_amps + (1 - alpha) * current_amp
            self.activation_variance = alpha * self.activation_variance + (1 - alpha) * current_var
        
        self.activation_count += tmp
        
        # === 原有 Hessian 更新逻辑 ===
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def compute_importance_scores(self, W, Hinv):
        """
        多维度重要性评估 (核心改进)
        
        评估维度:
        1. 激活重要性: 基于激活幅值
        2. Hessian 重要性: 基于二阶信息
        3. 权重重要性: 基于权重幅值
        4. 输出敏感度: 综合权重和激活
        5. 激活稳定性: 基于激活方差
        
        Returns:
            importance_scores: [columns] 综合重要性分数
            component_scores: dict, 各维度分数（用于分析）
        """
        # 1. 激活重要性
        act_importance = self.activation_amps / (torch.mean(self.activation_amps) + 1e-8)
        
        # 2. Hessian 重要性 (对角元素反映参数重要性)
        hessian_importance = torch.diag(Hinv)
        hessian_importance = torch.abs(hessian_importance)
        hessian_importance = hessian_importance / (torch.mean(hessian_importance) + 1e-8)
        
        # 3. 权重重要性 (平均绝对值)
        weight_importance = torch.mean(torch.abs(W), dim=0)
        weight_importance = weight_importance / (torch.mean(weight_importance) + 1e-8)
        
        # 4. 输出敏感度 (权重×激活的RMS)
        output_sensitivity = torch.sqrt(
            torch.mean((W ** 2) * (self.activation_amps.unsqueeze(0) ** 2), dim=0)
        )
        output_sensitivity = output_sensitivity / (torch.mean(output_sensitivity) + 1e-8)
        
        # 5. 激活稳定性 (方差越大越不稳定，需要更高精度)
        activation_stability = self.activation_variance / (torch.mean(self.activation_variance) + 1e-8)
        
        # === 加权融合 (可调参数) ===
        weights = {
            'activation': 0.25,      # 激活重要性
            'hessian': 0.25,         # Hessian 重要性
            'weight': 0.15,          # 权重重要性
            'output': 0.25,          # 输出敏感度
            'stability': 0.10        # 激活稳定性
        }
        
        importance_scores = (
            weights['activation'] * act_importance +
            weights['hessian'] * hessian_importance +
            weights['weight'] * weight_importance +
            weights['output'] * output_sensitivity +
            weights['stability'] * activation_stability
        )
        
        # 保存各维度分数（用于调试和分析）
        component_scores = {
            'activation': act_importance,
            'hessian': hessian_importance,
            'weight': weight_importance,
            'output': output_sensitivity,
            'stability': activation_stability
        }
        
        return importance_scores, component_scores

    def allocate_bits(self, importance_scores, target_avg_bits=4.0, method='quantile'):
        """
        精细化比特分配 (核心改进)
        
        Args:
            importance_scores: [columns] 重要性分数
            target_avg_bits: 目标平均比特数
            method: 分配方法 ('quantile' 或 'budget')
        
        Returns:
            bit_allocation: [columns] 每个通道的比特数
        """
        bit_allocation = torch.zeros_like(importance_scores)
        
        if method == 'quantile':
            # 方法1: 基于分位数的分配 (简单高效)
            # 将通道分为5档: 2bit (20%), 3bit (20%), 4bit (20%), 6bit (20%), 8bit (20%)
            quantiles = torch.quantile(
                importance_scores,
                torch.tensor([0.2, 0.4, 0.6, 0.8], device=self.dev)
            )
            
            bit_allocation[importance_scores < quantiles[0]] = 2
            bit_allocation[(importance_scores >= quantiles[0]) & 
                          (importance_scores < quantiles[1])] = 3
            bit_allocation[(importance_scores >= quantiles[1]) & 
                          (importance_scores < quantiles[2])] = 4
            bit_allocation[(importance_scores >= quantiles[2]) & 
                          (importance_scores < quantiles[3])] = 6
            bit_allocation[importance_scores >= quantiles[3]] = 8
            
        elif method == 'budget':
            # 方法2: 基于比特预算的优化分配 (更精确但稍慢)
            bit_options = torch.tensor([2, 3, 4, 6, 8], device=self.dev)
            n_channels = len(importance_scores)
            
            # 贪心算法: 优先给重要通道分配高比特
            sorted_indices = torch.argsort(importance_scores, descending=True)
            
            # 初始化为最低比特
            bit_allocation[:] = 2
            current_avg = 2.0
            
            # 逐步提升重要通道的比特数
            for idx in sorted_indices:
                for bit in [3, 4, 6, 8]:
                    # 尝试提升到更高比特
                    new_avg = current_avg + (bit - bit_allocation[idx]) / n_channels
                    if new_avg <= target_avg_bits:
                        bit_allocation[idx] = bit
                        current_avg = new_avg
                    else:
                        break
        
        return bit_allocation

    def fasterprune(
        self, 
        sparsity, 
        prunen=0, 
        prunem=0, 
        blocksize=128, 
        percdamp=.01,
        target_avg_bits=4.0,
        bit_allocation_method='quantile'
    ):
        """
        增强版剪枝 + 量化
        
        新增参数:
            target_avg_bits: 目标平均比特数 (默认4.0)
            bit_allocation_method: 比特分配方法 ('quantile' 或 'budget')
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
        
        # === Hessian 处理 ===
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

        # === 改进: 多维度重要性评估 ===
        use_enhanced_quantization = (
            hasattr(self, 'quantizer') and 
            self.activation_amps is not None
        )
        
        bit_allocation = None
        importance_scores = None
        component_scores = None
        
        if use_enhanced_quantization:
            print(f"\n[{self.layer_name}] 计算多维度重要性分数...")
            importance_scores, component_scores = self.compute_importance_scores(W, Hinv)
            
            print(f"[{self.layer_name}] 分配量化比特数...")
            bit_allocation = self.allocate_bits(
                importance_scores, 
                target_avg_bits=target_avg_bits,
                method=bit_allocation_method
            )
            
            # 统计比特分布
            unique_bits, counts = torch.unique(bit_allocation, return_counts=True)
            print(f"[{self.layer_name}] 比特分布: ", end="")
            for bit, count in zip(unique_bits, counts):
                print(f"{int(bit)}bit({int(count)}通道) ", end="")
            avg_bits = torch.mean(bit_allocation.float()).item()
            print(f"| 平均: {avg_bits:.2f} bits")
            
            # 更新统计信息
            if self.stats_collector is not None:
                self.stats_collector.update(bit_allocation, importance_scores, self.layer_name)

        mask = None

        # === 逐块剪枝 + 量化 ===
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            # 剪枝掩码计算
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

                # 基础剪枝
                q = w.clone()
                q[mask1[:, i]] = 0

                # === 改进: 动态精度量化 ===
                if use_enhanced_quantization:
                    # 根据该列的重要性动态调整量化比特
                    target_bits = int(bit_allocation[global_idx].item())
                    self.quantizer.maxq = torch.tensor(2 ** target_bits - 1, device=self.dev)
                    
                    # ✅ 修复: 为当前通道重新计算scale和zero
                    self.quantizer.find_params(q.unsqueeze(1), weight=True)
                    
                    # 量化
                    q = quantize(
                        q.unsqueeze(1), 
                        self.quantizer.scale, 
                        self.quantizer.zero, 
                        self.quantizer.maxq
                    ).flatten()
                elif hasattr(self, 'quantizer'):
                    # 原始固定比特量化
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

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        elapsed_time = time.time() - tick
        total_error = torch.sum(Losses).item()
        
        print(f'[{self.layer_name}] 时间: {elapsed_time:.2f}s | 误差: {total_error:.4f}')

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        """释放资源"""
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.activation_amps = None
        self.activation_variance = None
        torch.cuda.empty_cache()

