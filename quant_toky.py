import numpy as np
import torch
import torch.nn as nn


def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))
        # 互信息分组和动态缩放相关参数
        self.use_mutual_info = False
        self.num_clusters = 4
        self.dynamic_scaling = False

    def configure(
            self,
            bits, perchannel=False, sym=True,
            mse=False, norm=2.4, grid=100, maxshrink=.8,
            grouprows=1, use_mutual_info=False, num_clusters=4, dynamic_scaling=False
        ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.grouprows = grouprows
        # 初始化新增参数
        self.use_mutual_info = use_mutual_info
        self.num_clusters = num_clusters
        self.dynamic_scaling = dynamic_scaling

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        # 处理按通道量化的维度
        if self.perchannel:
            if weight:
                x = x.flatten(1)
                if self.grouprows > 1:
                    x = x.reshape((x.shape[0] // self.grouprows, -1))
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3]).flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        # 计算数据的最大/最小值（避免全零导致的除零错误）
        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp_neg = xmin < 0
            if torch.any(tmp_neg):
                xmin[tmp_neg] = -xmax[tmp_neg]
        tmp_zero = (xmin == 0) & (xmax == 0)
        xmin[tmp_zero] = -1
        xmax[tmp_zero] = +1

        # 初始量化参数（scale和zero）
        self.scale = (xmax - xmin) / self.maxq
        self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2) if self.sym else torch.round(-xmin / self.scale)

        # -------------------------- 核心修复：互信息计算（替换torch.histogram2d） --------------------------
        if self.use_mutual_info and weight:
            sample_size = min(200, x.shape[1])  # 采样减少计算量
            sample_indices = torch.randperm(x.shape[1], device=dev)[:sample_size]
            x_sample = x[:, sample_indices]

            mutual_info_matrix = torch.zeros(sample_size, sample_size, device=dev)
            for i in range(sample_size):
                for j in range(i, sample_size):  # 上三角计算，对称填充
                    # 1. 转移到CPU并转为NumPy数组（适配旧PyTorch版本）
                    wi_np = x_sample[:, i].cpu().detach().numpy()
                    wj_np = x_sample[:, j].cpu().detach().numpy()

                    # 2. 用NumPy计算直方图（替代torch.histogram和torch.histogram2d）
                    # 一维直方图（wi）
                    hist_i_np, bins_i_np = np.histogram(wi_np, bins=32)
                    # 一维直方图（wj）
                    hist_j_np, bins_j_np = np.histogram(wj_np, bins=32)
                    # 二维直方图（wi, wj）
                    hist_ij_np, bins_i_joint_np, bins_j_joint_np = np.histogram2d(wi_np, wj_np, bins=32)

                    # 3. 转回PyTorch张量并移回原设备
                    hist_i = torch.tensor(hist_i_np, device=dev, dtype=torch.float32)
                    hist_j = torch.tensor(hist_j_np, device=dev, dtype=torch.float32)
                    hist_ij = torch.tensor(hist_ij_np, device=dev, dtype=torch.float32)

                    # 4. 归一化（避免log2(0)）
                    hist_i = hist_i / (hist_i.sum() + 1e-10)
                    hist_j = hist_j / (hist_j.sum() + 1e-10)
                    hist_ij = hist_ij / (hist_ij.sum() + 1e-10)

                    # 5. 计算熵和互信息
                    H_i = -torch.sum(hist_i * torch.log2(hist_i + 1e-10))
                    H_j = -torch.sum(hist_j * torch.log2(hist_j + 1e-10))
                    H_ij = -torch.sum(hist_ij * torch.log2(hist_ij + 1e-10))
                    mutual_info = H_i + H_j - H_ij

                    mutual_info_matrix[i, j] = mutual_info
                    mutual_info_matrix[j, i] = mutual_info  # 对称矩阵填充

            # 聚类（PyTorch原生KMeans，不依赖外部库）
            from torch.cluster import kmeans
            mutual_info_flat = mutual_info_matrix.flatten().unsqueeze(1).float()
            cluster_ids, _ = kmeans(mutual_info_flat, self.num_clusters, distance_function='euclidean')

            # 扩展聚类结果到全部特征
            full_cluster_ids = torch.zeros(x.shape[1], device=dev, dtype=torch.long)
            for i in range(sample_size):
                full_cluster_ids[sample_indices[i]] = cluster_ids[i]
            # 未采样特征按余弦相似度分配簇
            for j in range(x.shape[1]):
                if j not in sample_indices:
                    similarities = []
                    for i in sample_indices:
                        sim = torch.cosine_similarity(x[:, i].unsqueeze(0), x[:, j].unsqueeze(0)).item()
                        similarities.append(sim)
                    closest_idx = torch.argmax(torch.tensor(similarities)).item()
                    full_cluster_ids[j] = full_cluster_ids[sample_indices[closest_idx]]

            # 为每个簇分配动态比特并计算量化参数
            for cluster_id in range(self.num_clusters):
                mask = (full_cluster_ids == cluster_id)
                if not torch.any(mask):
                    continue
                x_cluster = x[:, mask]
                # 按簇重要性分配比特（高重要性→高比特）
                cluster_importance = torch.mean(torch.abs(x_cluster))
                global_importance = torch.mean(torch.abs(x))
                if cluster_importance > global_importance * 1.2:
                    bits = min(8, self.maxq.bit_length())
                elif cluster_importance < global_importance * 0.5:
                    bits = max(2, self.maxq.bit_length() // 2)
                else:
                    bits = self.maxq.bit_length()
                cluster_maxq = torch.tensor(2 ** bits - 1, device=dev)

                # 计算簇的scale和zero
                c_xmin = torch.minimum(x_cluster.min(1)[0], torch.zeros_like(x_cluster.min(1)[0]))
                c_xmax = torch.maximum(x_cluster.max(1)[0], torch.zeros_like(x_cluster.max(1)[0]))
                if self.sym:
                    c_xmax = torch.maximum(torch.abs(c_xmin), c_xmax)
                    tmp_neg = c_xmin < 0
                    if torch.any(tmp_neg):
                        c_xmin[tmp_neg] = -c_xmax[tmp_neg]
                c_scale = (c_xmax - c_xmin) / cluster_maxq
                c_zero = torch.full_like(c_scale, (cluster_maxq + 1) / 2) if self.sym else torch.round(-c_xmin / c_scale)

                # 确保形状匹配（避免广播错误）
                if len(self.scale.shape) > 1 and len(c_scale.shape) == 1:
                    c_scale = c_scale.unsqueeze(1)
                    c_zero = c_zero.unsqueeze(1)
                target_shape = self.scale[:, mask].shape
                if c_scale.shape[0] != target_shape[0]:
                    c_scale = c_scale.repeat(target_shape[0] // c_scale.shape[0], 1)[:target_shape[0]]
                    c_zero = c_zero.repeat(target_shape[0] // c_zero.shape[0], 1)[:target_shape[0]]

                # 赋值量化参数
                self.scale[:, mask] = c_scale
                self.zero[:, mask] = c_zero

        # -------------------------- 原有MSE优化和动态缩放逻辑 --------------------------
        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                err = torch.sum(torch.abs(q - x) ** self.norm, 1)
                tmp_best = err < best
                if torch.any(tmp_best):
                    best[tmp_best] = err[tmp_best]
                    self.scale[tmp_best] = scale1[tmp_best]
                    self.zero[tmp_best] = zero1[tmp_best]

        if self.dynamic_scaling and weight and not self.use_mutual_info:
            s = torch.ones(x.shape[0], device=dev, requires_grad=True)
            optimizer = torch.optim.Adam([s], lr=1e-2)
            for _ in range(30):
                optimizer.zero_grad()
                scaled_x = x * s.unsqueeze(1)
                q_scaled = quantize(scaled_x, self.scale, self.zero, self.maxq)
                q = q_scaled / s.unsqueeze(1)
                loss = torch.mean((x - q) ** 2)
                loss.backward()
                optimizer.step()
            self.scale *= s.unsqueeze(1)
            self.zero *= s.unsqueeze(1)

        # 调整量化参数形状以匹配原始权重
        if not self.perchannel:
            repeat_dim = shape[0] if weight else (shape[1] if len(shape) != 3 else shape[2])
            self.scale = self.scale.repeat(repeat_dim)
            self.zero = self.zero.repeat(repeat_dim)

        if weight:
            if self.grouprows > 1:
                self.scale = self.scale.unsqueeze(1).repeat(1, self.grouprows)
                self.zero = self.zero.unsqueeze(1).repeat(1, self.grouprows)
            self.scale = self.scale.reshape([-1] + [1] * (len(shape) - 1))
            self.zero = self.zero.reshape([-1] + [1] * (len(shape) - 1))
            return

        # 调整激活值量化参数形状
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        return quantize(x, self.scale, self.zero, self.maxq) if self.ready() else x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)