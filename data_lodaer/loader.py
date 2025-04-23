# 导入库
import scipy.io as sio
import numpy  as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

class HSI(Dataset):
    def __init__(self, data_path, label_path, patch_size=5, train=True, train_split=0.1, use_pca=False, n_components=10):
        # 加载数据集
        self.data, self.label = self.load_data(data_path, label_path, use_pca, n_components)
       
        self.patch_size = patch_size
        self.train = train
        self.train_split = train_split

        # 数据集划分
        self.train_data, self.test_data, self.train_label, self.test_label = self.split_data()
        self.num_channels = self.data.shape[2]
        self.num_classes = np.unique(self.label).size

    def load_data(self, data_path, label_path, use_pca, n_components):
        # 读取.mat文件
        data = sio.loadmat(data_path)
        label = sio.loadmat(label_path)

        # 解析数据
        data_key = list(data.keys())[-1]  # 获取数据键
        label_key = list(label.keys())[-1]

        data = data[data_key].astype(np.float32)  # 提取数据并转换为float32类型
        label = label[label_key].astype(np.int64)

        # 归一化数据
        scaler = MinMaxScaler()  # 创建归一化器
        height, width, channels = data.shape
        data = data.reshape(-1, channels)
        data = scaler.fit_transform(data)
        data = data.reshape(height, width, channels)

        # PCA降维
        if use_pca:
            pca = PCA(n_components=n_components)  # 使用n_components指定降维后的维度
            data = data.reshape(-1, channels)  # 先将数据展平
            data = pca.fit_transform(data)  # 进行PCA降维
            data = data.reshape(height, width, n_components)  # 恢复成原始数据的尺寸（高，宽，降维后的通道数）

        self.label_shape = label.shape

        return data, label

    def split_data(self):
        # 划分数据集函数

        # 获取非零标签的索引
        indices = np.array(np.nonzero(self.label)).T
        indices = indices[self.label[tuple(indices.T)] != 0]
        num_samples = indices.shape[0]

        # 生成随机排列
        perm = np.random.permutation(num_samples)  # 生成随机排列
        train_size = int(num_samples * self.train_split)  # 计算训练集大小

        # 划分索引
        train_indices = indices[perm[:train_size], :]
        test_indices = indices[perm[train_size:], :]

        # 生成训练集
        train_data = np.array([self.extract_patch(x, y) for x, y in train_indices])
        train_label = self.label[tuple(train_indices.T)]

        # 生成测试集
        test_data = np.array([self.extract_patch(x, y) for x, y in test_indices])
        test_label = self.label[tuple(test_indices.T)]

        return train_data, test_data, train_label, test_label
    
    def extract_patch(self, x, y):
        half_size = self.patch_size // 2
        height, width, channels = self.data.shape

        x_min = max(0, x - half_size)
        x_max = min(height, x + half_size + 1)
        y_min = max(0, y - half_size)
        y_max = min(width, y + half_size + 1)

        # 检查patch是否有效
        if x_min == x_max or y_min == y_max:
            print(f"Warning: Invalid patch size as ({x}, {y}): x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
            return np.zeros((self.patch_size, self.patch_size, channels), dtype=np.float32)  # 返回零矩阵
        
        # 提取patch
        patch = self.data[x_min:x_max, y_min:y_max, :]

        # 处理边界情况
        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
            padded_patch = np.zeros((self.patch_size, self.patch_size, channels), dtype=np.float32)
            padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
            patch = padded_patch

        return patch
    
    def __len__(self):
        # 返回数据集大小
        return len(self.train_data) if self.train else len(self.test_data)
    
    def compute_spatial_feature(self, inputs):
        # 根据需求计算spatial特征，比如取均值，卷积
        spatial_feature = torch.mean(inputs, dim=0, keepdim=True)  # 取均值
        return spatial_feature
    
    def __getitem__(self, index):
        # 索引访问函数，用于从数据集中获取单个样本，返回光谱数据、空间数据和标签
        if self.train:
            # permute(2, 0, 1)将原本的(H, W, C)变为(C, H, W)，符合pytorch处理图像数据的格式：通道优先
            spectral_data = torch.tensor(self.train_data[index]).permute(2, 0, 1)
            spatial_data = torch.tensor(self.train_data[index]).permute(2, 0, 1)
            label = torch.tensor(self.train_label[index])
        else:
            spectral_data = torch.tensor(self.test_data[index]).permute(2, 0, 1)
            spatial_data = torch.tensor(self.test_data[index]).permute(2, 0, 1)
            label = torch.tensor(self.test_label[index])
    
        return spectral_data, spatial_data, label

    @staticmethod
    def get_dataloader(data_path, label_path, batch_size=32, patch_size=5, train=True, use_pca=False, n_components=10):
        # 静态方法：创建数据加载器
        dataset = HSI(data_path, label_path, patch_size, train, use_pca=use_pca, n_components=n_components)
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
        )
