"""
互信息计算模块

实现通道间互信息的高效计算，用于量化分组
"""

import torch
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import warnings

warnings.filterwarnings('ignore')


class MutualInfoEstimator:
    """互信息估计器"""
    
    def __init__(self, method='histogram', n_bins=50):
        """
        Args:
            method: 'histogram', 'knn', 'correlation'
            n_bins: 直方图方法的bin数量
        """
        self.method = method
        self.n_bins = n_bins
    
    def compute_entropy(self, x):
        """计算熵"""
        hist, _ = np.histogram(x, bins=self.n_bins, density=True)
        hist = hist + 1e-10  # 避免log(0)
        return entropy(hist)
    
    def compute_joint_entropy(self, x, y):
        """计算联合熵"""
        hist_2d, _, _ = np.histogram2d(x, y, bins=self.n_bins, density=True)
        hist_2d = hist_2d + 1e-10
        return entropy(hist_2d.flatten())
    
    def compute_mi_histogram(self, x, y):
        """基于直方图的互信息估计"""
        H_x = self.compute_entropy(x)
        H_y = self.compute_entropy(y)
        H_xy = self.compute_joint_entropy(x, y)
        
        MI = H_x + H_y - H_xy
        
        # 归一化互信息 (NMI)
        if H_x + H_y > 0:
            NMI = 2.0 * MI / (H_x + H_y)
        else:
            NMI = 0.0
        
        return MI, NMI
    
    def compute_mi_correlation(self, x, y):
        """基于相关系数的近似（快速但不精确）"""
        corr = np.corrcoef(x, y)[0, 1]
        # 使用 -0.5 * log(1 - corr^2) 近似
        if abs(corr) < 0.9999:
            MI_approx = -0.5 * np.log(1 - corr ** 2)
        else:
            MI_approx = 10.0  # 高相关
        
        return MI_approx, abs(corr)
    
    def compute_mi_knn(self, x, y, k=3):
        """基于KNN的互信息估计（更准确但较慢）"""
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            x = x.reshape(-1, 1)
            y = y.reshape(-1)
            
            MI = mutual_info_regression(x, y, n_neighbors=k, random_state=42)[0]
            
            # 归一化
            H_x = self.compute_entropy(x.flatten())
            H_y = self.compute_entropy(y)
            
            if H_x + H_y > 0:
                NMI = 2.0 * MI / (H_x + H_y)
            else:
                NMI = 0.0
            
            return MI, NMI
        except:
            # 如果失败，回退到直方图方法
            return self.compute_mi_histogram(x, y)
    
    def compute(self, x, y):
        """
        计算两个变量的互信息
        
        Args:
            x, y: numpy arrays of shape [N]
        
        Returns:
            MI: 互信息
            NMI: 归一化互信息
        """
        # 展平
        x = x.flatten()
        y = y.flatten()
        
        # 检查
        assert len(x) == len(y), "x和y长度必须相同"
        
        if self.method == 'histogram':
            return self.compute_mi_histogram(x, y)
        elif self.method == 'correlation':
            return self.compute_mi_correlation(x, y)
        elif self.method == 'knn':
            return self.compute_mi_knn(x, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")


def compute_pairwise_mi_matrix(activations, method='correlation', batch_size=None):
    """
    计算激活矩阵的成对互信息矩阵
    
    Args:
        activations: torch.Tensor [batch, seq_len, n_channels] 或 [samples, n_channels]
        method: 'histogram', 'knn', 'correlation'
        batch_size: 分批计算以节省内存
    
    Returns:
        MI_matrix: numpy array [n_channels, n_channels]
        NMI_matrix: numpy array [n_channels, n_channels]
    """
    # 转换为numpy并reshape
    if isinstance(activations, torch.Tensor):
        activations = activations.cpu().numpy()
    
    # 展平为 [samples, n_channels]
    if len(activations.shape) == 3:
        batch, seq_len, n_channels = activations.shape
        activations = activations.reshape(-1, n_channels)
    else:
        _, n_channels = activations.shape
    
    print(f"计算互信息矩阵: {n_channels} 通道, {activations.shape[0]} 样本")
    
    # 初始化
    MI_matrix = np.zeros((n_channels, n_channels))
    NMI_matrix = np.zeros((n_channels, n_channels))
    
    estimator = MutualInfoEstimator(method=method)
    
    # 计算成对互信息
    count = 0
    total = n_channels * (n_channels - 1) // 2
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            if i == j:
                # 对角线：自身的熵
                MI_matrix[i, j] = estimator.compute_entropy(activations[:, i])
                NMI_matrix[i, j] = 1.0
            else:
                # 非对角线：互信息
                MI, NMI = estimator.compute(activations[:, i], activations[:, j])
                MI_matrix[i, j] = MI
                MI_matrix[j, i] = MI
                NMI_matrix[i, j] = NMI
                NMI_matrix[j, i] = NMI
                
                count += 1
                if count % 1000 == 0:
                    print(f"  进度: {count}/{total} ({count/total*100:.1f}%)")
    
    print(f"互信息矩阵计算完成！")
    print(f"  MI范围: [{MI_matrix.min():.4f}, {MI_matrix.max():.4f}]")
    print(f"  NMI范围: [{NMI_matrix.min():.4f}, {NMI_matrix.max():.4f}]")
    
    return MI_matrix, NMI_matrix


def compute_mi_matrix_fast(activations, method='correlation'):
    """
    快速计算互信息矩阵（使用相关系数近似）
    
    适用于大规模通道数
    """
    if isinstance(activations, torch.Tensor):
        activations = activations.cpu().numpy()
    
    # 展平为 [samples, n_channels]
    if len(activations.shape) == 3:
        activations = activations.reshape(-1, activations.shape[-1])
    
    print(f"快速计算互信息矩阵（相关系数法）: {activations.shape[1]} 通道")
    
    # 计算相关系数矩阵
    corr_matrix = np.corrcoef(activations.T)  # [n_channels, n_channels]
    
    # 处理NaN值（常量通道或零方差通道）
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 转换为互信息近似
    corr_matrix = np.clip(corr_matrix, -0.9999, 0.9999)  # 避免log(0)
    MI_matrix = -0.5 * np.log(1 - corr_matrix ** 2)
    
    # 再次处理可能的NaN
    MI_matrix = np.nan_to_num(MI_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 归一化互信息
    NMI_matrix = np.abs(corr_matrix)
    
    print(f"互信息矩阵计算完成（快速模式）")
    print(f"  MI范围: [{MI_matrix.min():.4f}, {MI_matrix.max():.4f}]")
    print(f"  NMI范围: [{NMI_matrix.min():.4f}, {NMI_matrix.max():.4f}]")
    
    return MI_matrix, NMI_matrix


if __name__ == '__main__':
    # 测试
    print("测试互信息计算...")
    
    # 生成测试数据
    n_samples = 1000
    n_channels = 10
    
    # 创建一些相关和不相关的通道
    x = np.random.randn(n_samples, n_channels)
    x[:, 1] = x[:, 0] + 0.1 * np.random.randn(n_samples)  # 通道1和0高度相关
    x[:, 2] = x[:, 0] + 0.5 * np.random.randn(n_samples)  # 通道2和0中等相关
    
    # 测试成对互信息
    estimator = MutualInfoEstimator(method='correlation')
    MI_01, NMI_01 = estimator.compute(x[:, 0], x[:, 1])
    MI_02, NMI_02 = estimator.compute(x[:, 0], x[:, 2])
    MI_03, NMI_03 = estimator.compute(x[:, 0], x[:, 3])
    
    print(f"\n通道0-1 (高相关): MI={MI_01:.4f}, NMI={NMI_01:.4f}")
    print(f"通道0-2 (中相关): MI={MI_02:.4f}, NMI={NMI_02:.4f}")
    print(f"通道0-3 (低相关): MI={MI_03:.4f}, NMI={NMI_03:.4f}")
    
    # 测试矩阵计算
    print("\n测试互信息矩阵计算...")
    MI_matrix, NMI_matrix = compute_mi_matrix_fast(x)
    print(f"矩阵形状: {MI_matrix.shape}")
    print(f"对角线均值: {np.diag(MI_matrix).mean():.4f}")

