# 读取.mat数据集并预处理    

# 导入库
import os
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class HyperspectralDataset(Dataset):
    def __init__(self, data_path, label_path, patch_size=5, train=True, train_split=0.1):
        # 加载数据集
        self.data, self.labels = self.load_data(data_path, label_path)
        self.patch_size = patch_size # 图像块大小
        self.train = train
        self.train_split = train_split # 训练集比例

        # 数据集划分
        self.train_data, self.test_data, self.train_labels, self.test_labels = self.split_data() # 划分数据集
        self.num_channels = self.data.shape[2] # 光谱通道数
        self.num_classes = np.unique(self.labels).size # 获取类别数

    def load_data(self, data_path, label_path):
        # 读取 .mat文件
        data = sio.loadmat(data_path)
        labels = sio.loadmat(label_path)

        # 解析数据
        data_key = list(data.keys())[-1] # 获取数据键
        label_key = list(labels.keys())[-1] # 获取标签键

        data = data[data_key].astype(np.float32)  # 提取数据并转换为float32类型
        labels = labels[label_key].astype(np.int64)  # 提取标签并转换为int64类型

        # 归一化数据
        scaler = MinMaxScaler() # 创建归一化器
        h, w, c =data.shape
        data = data.reshape(-1,c) # 转化为2维
        data = scaler.fit_transform(data)
        data = data.reshape(h, w, c) # 还原形状

        self.label_shape = labels.shape
        return data, labels
    
    def split_data(self):
        # 划分数据集函数

        indices = np.array(np.nonzero(self.labels)).T  # (N, 2) 形状
        num_samples = indices.shape[0]  # 计算有效样本数

        # 打乱索引
        perm = np.random.permutation(num_samples) # 生成随机排列
        train_size = int(num_samples * self.train_split) # 计算非零样本的数量

        # 划分训练集和测试集索引
        train_indices = indices[perm[:train_size], :]
        test_indices = indices[perm[train_size:], :]

        # 训练集
        train_data = np.array([self.extract_patch(x, y) for x, y in train_indices])
        train_labels = self.labels[tuple(train_indices.T)] # 获取对应标签

        # 测试集
        test_data = np.array([self.extract_patch(x, y) for x, y in test_indices])
        test_labels = self.labels[tuple(test_indices.T)]

        return train_data, test_data, train_labels, test_labels
    
    def extract_patch(self, x, y):
        half_size = self.patch_size // 2
        h, w, c = self.data.shape

        x_min = max(0, x - half_size)
        x_max = min(h, x + half_size + 1)
        y_min = max(0, y - half_size)
        y_max = min(w, y + half_size + 1)

        # 修正无效补丁的情况，确保补丁大小有效
        if x_min == x_max or y_min == y_max:
            print(f"Warning: Invalid patch size at ({x}, {y}): x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
            return np.zeros((self.patch_size, self.patch_size, c), dtype=np.float32)  # 返回零矩阵
        
        patch = self.data[x_min:x_max, y_min:y_max, :]

        # 如果补丁尺寸不够，填充补丁
        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
            padded_patch = np.zeros((self.patch_size, self.patch_size, c), dtype=np.float32)
            padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
            patch = padded_patch

        return patch

    def __len__(self):
        # 返回数据集大小
        return len(self.train_data) if self.train else len(self.test_data)

    def compute_spatial_feature(self, inputs):
        # 这里根据需求计算 spatial 特征，比如取均值、卷积等
        spatial_feature = torch.mean(inputs, dim=0, keepdim=True)  # 例如取均值
        return spatial_feature

    def __getitem__(self, idx):
        if self.train:
            spectral_data = torch.tensor(self.train_data[idx]).permute(2, 0, 1)  # [176, 5, 5]
            spatial_data = torch.tensor(self.train_data[idx]).permute(2, 0, 1)  # 这里改为完整通道
            label = torch.tensor(self.train_labels[idx])
        else:
            spectral_data = torch.tensor(self.test_data[idx]).permute(2, 0, 1)  # [176, 5, 5]
            spatial_data = torch.tensor(self.test_data[idx]).permute(2, 0, 1)  # 这里改为完整通道
            label = torch.tensor(self.test_labels[idx])
        
        # 确保不返回背景类
        if label == 0:
           return None 
        
        return spectral_data, spatial_data, label
        
    @staticmethod
    def get_dataloader(data_path, label_path, batch_size=32, patch_size=5, train=True):
        # 静态方法：创建数据加载器
        dataset = HyperspectralDataset(data_path, label_path, patch_size, train) # 创建数据集实例
        return DataLoader(dataset, batch_size=batch_size, shuffle=True) # 返回数据加载器
