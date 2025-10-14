"""
通道分组算法

基于互信息矩阵进行通道聚类分组
"""

import numpy as np
import torch
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


class ChannelGrouping:
    """通道分组器"""
    
    def __init__(self, n_groups=10, method='spectral'):
        """
        Args:
            n_groups: 分组数量
            method: 'spectral', 'hierarchical', 'kmeans'
        """
        self.n_groups = n_groups
        self.method = method
        self.groups = None
        self.group_info = None
    
    def fit(self, MI_matrix, importance_scores=None):
        """
        基于互信息矩阵进行分组
        
        Args:
            MI_matrix: [n_channels, n_channels] 互信息/相似度矩阵
            importance_scores: [n_channels] 可选的重要性分数
        
        Returns:
            groups: [n_channels] 每个通道的组ID
        """
        n_channels = MI_matrix.shape[0]
        
        print(f"开始通道分组: {n_channels} 通道 → {self.n_groups} 组")
        
        # 转换为相似度矩阵（确保非负）
        affinity = np.abs(MI_matrix)
        
        # 处理NaN和Inf
        affinity = np.nan_to_num(affinity, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 归一化
        affinity_min = affinity.min()
        affinity_max = affinity.max()
        if affinity_max - affinity_min > 1e-8:
            affinity = (affinity - affinity_min) / (affinity_max - affinity_min + 1e-8)
        else:
            # 如果所有值都相同，使用单位矩阵
            affinity = np.eye(n_channels)
        
        if self.method == 'spectral':
            # 谱聚类（适合复杂结构）
            clustering = SpectralClustering(
                n_clusters=self.n_groups,
                affinity='precomputed',
                assign_labels='kmeans',
                random_state=42
            )
            self.groups = clustering.fit_predict(affinity)
            
        elif self.method == 'hierarchical':
            # 层次聚类（可解释性强）
            # 转换为距离矩阵
            distance = 1.0 - affinity
            distance = np.maximum(distance, 0)  # 确保非负
            
            clustering = AgglomerativeClustering(
                n_clusters=self.n_groups,
                linkage='average'
            )
            self.groups = clustering.fit_predict(distance)
            
        elif self.method == 'kmeans':
            # K-means（快速但需要特征表示）
            # 使用MI矩阵的行作为特征
            clustering = KMeans(
                n_clusters=self.n_groups,
                random_state=42,
                n_init=10
            )
            self.groups = clustering.fit_predict(MI_matrix)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # 计算分组质量
        try:
            score = silhouette_score(affinity, self.groups, metric='precomputed')
            print(f"分组质量 (Silhouette Score): {score:.4f}")
        except:
            print("无法计算Silhouette Score")
        
        # 统计分组信息
        self.group_info = self._analyze_groups(importance_scores)
        
        return self.groups
    
    def _analyze_groups(self, importance_scores=None):
        """分析分组统计信息"""
        group_info = []
        
        for group_id in range(self.n_groups):
            channels = np.where(self.groups == group_id)[0]
            
            info = {
                'group_id': group_id,
                'size': len(channels),
                'channels': channels.tolist()
            }
            
            if importance_scores is not None:
                group_importance = importance_scores[channels]
                info['avg_importance'] = float(group_importance.mean())
                info['max_importance'] = float(group_importance.max())
                info['min_importance'] = float(group_importance.min())
            
            group_info.append(info)
        
        # 按平均重要性排序
        if importance_scores is not None:
            group_info = sorted(group_info, key=lambda x: x['avg_importance'], reverse=True)
        
        return group_info
    
    def print_summary(self):
        """打印分组摘要"""
        if self.group_info is None:
            print("尚未进行分组")
            return
        
        print("\n" + "="*80)
        print("通道分组摘要")
        print("="*80)
        
        for info in self.group_info:
            gid = info['group_id']
            size = info['size']
            
            print(f"\n组 {gid}: {size} 通道")
            
            if 'avg_importance' in info:
                print(f"  平均重要性: {info['avg_importance']:.4f}")
                print(f"  重要性范围: [{info['min_importance']:.4f}, {info['max_importance']:.4f}]")
            
            # 只显示前10个通道
            channels_str = str(info['channels'][:10])
            if len(info['channels']) > 10:
                channels_str = channels_str[:-1] + ', ...]'
            print(f"  通道索引: {channels_str}")
        
        print("\n" + "="*80)
    
    def allocate_bits(self, target_avg_bits=4.0, bit_options=[2, 3, 4, 6, 8]):
        """
        为每组分配比特数
        
        Args:
            target_avg_bits: 目标平均比特数
            bit_options: 可选的比特数
        
        Returns:
            group_bits: dict {group_id: bits}
        """
        if self.group_info is None:
            raise ValueError("需要先执行fit()")
        
        # 计算总通道数
        total_channels = sum(info['size'] for info in self.group_info)
        
        # 策略1: 基于重要性分配
        if 'avg_importance' in self.group_info[0]:
            # 按重要性排序（已排序）
            importances = np.array([info['avg_importance'] for info in self.group_info])
            sizes = np.array([info['size'] for info in self.group_info])
            
            # 归一化重要性
            importances_norm = importances / importances.sum()
            
            # 分配比特数
            group_bits = {}
            total_bits_used = 0
            
            for i, info in enumerate(self.group_info):
                gid = info['group_id']
                size = info['size']
                imp = importances_norm[i]
                
                # 根据重要性选择比特数
                if imp > 0.15:  # 高重要性
                    bits = 8
                elif imp > 0.10:
                    bits = 6
                elif imp > 0.05:
                    bits = 4
                elif imp > 0.02:
                    bits = 3
                else:
                    bits = 2
                
                group_bits[gid] = bits
                total_bits_used += bits * size
            
            # 调整以满足目标平均比特数
            current_avg = total_bits_used / total_channels
            
            print(f"\n比特分配:")
            print(f"  目标平均: {target_avg_bits:.2f} bits")
            print(f"  当前平均: {current_avg:.2f} bits")
            
            # 简单调整（可以优化）
            if abs(current_avg - target_avg_bits) > 0.5:
                print(f"  ⚠ 需要调整比特分配")
        
        else:
            # 策略2: 均匀分配
            group_bits = {info['group_id']: target_avg_bits for info in self.group_info}
        
        return group_bits
    
    def visualize(self, MI_matrix, save_path=None):
        """可视化分组结果"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if self.groups is None:
            print("尚未进行分组")
            return
        
        # 按分组排序
        sorted_indices = np.argsort(self.groups)
        MI_sorted = MI_matrix[sorted_indices][:, sorted_indices]
        
        # 绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 原始互信息矩阵
        sns.heatmap(MI_matrix, cmap='viridis', ax=ax1, cbar_kws={'label': 'Mutual Information'})
        ax1.set_title('Original MI Matrix')
        ax1.set_xlabel('Channel Index')
        ax1.set_ylabel('Channel Index')
        
        # 排序后的互信息矩阵（显示分组结构）
        sns.heatmap(MI_sorted, cmap='viridis', ax=ax2, cbar_kws={'label': 'Mutual Information'})
        ax2.set_title(f'Grouped MI Matrix ({self.n_groups} groups)')
        ax2.set_xlabel('Channel Index (sorted by group)')
        ax2.set_ylabel('Channel Index (sorted by group)')
        
        # 添加分组边界
        group_boundaries = []
        current_pos = 0
        for gid in sorted(np.unique(self.groups)):
            size = np.sum(self.groups == gid)
            current_pos += size
            group_boundaries.append(current_pos)
        
        for boundary in group_boundaries[:-1]:
            ax2.axhline(y=boundary, color='red', linewidth=1, linestyle='--')
            ax2.axvline(x=boundary, color='red', linewidth=1, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()


def auto_select_n_groups(MI_matrix, min_groups=5, max_groups=50, step=5):
    """
    自动选择最佳分组数
    
    使用Silhouette Score评估
    """
    print("自动选择最佳分组数...")
    
    scores = []
    n_groups_range = range(min_groups, max_groups + 1, step)
    
    for n_groups in n_groups_range:
        grouping = ChannelGrouping(n_groups=n_groups, method='spectral')
        groups = grouping.fit(MI_matrix)
        
        try:
            score = silhouette_score(MI_matrix, groups, metric='precomputed')
            scores.append(score)
            print(f"  n_groups={n_groups}: score={score:.4f}")
        except:
            scores.append(-1)
    
    # 选择最佳
    best_idx = np.argmax(scores)
    best_n_groups = list(n_groups_range)[best_idx]
    best_score = scores[best_idx]
    
    print(f"\n最佳分组数: {best_n_groups} (score={best_score:.4f})")
    
    return best_n_groups, scores


if __name__ == '__main__':
    # 测试
    print("测试通道分组...")
    
    # 生成测试数据
    n_channels = 50
    
    # 创建块状互信息矩阵（模拟相关通道）
    MI_matrix = np.random.rand(n_channels, n_channels) * 0.2
    
    # 添加块结构
    for i in range(0, n_channels, 10):
        for j in range(i, min(i+10, n_channels)):
            for k in range(i, min(i+10, n_channels)):
                MI_matrix[j, k] = 0.8 + np.random.rand() * 0.2
                MI_matrix[k, j] = MI_matrix[j, k]
    
    # 对角线设为1
    np.fill_diagonal(MI_matrix, 1.0)
    
    # 创建重要性分数
    importance_scores = np.random.rand(n_channels)
    
    # 分组
    grouping = ChannelGrouping(n_groups=5, method='spectral')
    groups = grouping.fit(MI_matrix, importance_scores)
    
    # 打印摘要
    grouping.print_summary()
    
    # 分配比特
    group_bits = grouping.allocate_bits(target_avg_bits=4.0)
    print(f"\n比特分配: {group_bits}")

