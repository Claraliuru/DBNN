"""对比模型：SSRN"""

import torch.nn as nn
import torch.nn.functional as F

# 光谱残差块
class SpectralResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2 # 设置padding
        # 第一个3D残差块，仅作用于光谱维度
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size, 1, 1), padding=(padding, 0, 0))
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二个3D残差块
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(kernel_size, 1, 1), padding=(padding, 0, 0))
        self.bn2 = nn.BatchNorm3d(out_channels)
        # 如果输入和输出通道数不同，则使用1x1卷积进行下采样，否则直接保留输入
        self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x) # 残差连接支路
        out = self.relu(self.bn1(self.conv1(x))) # 第一层卷积 + BN + ReLU
        out = self.bn2(self.conv2(out)) # 第二层卷积 + BN
        out += identity # 加上残差
        return self.relu(out) # 最后ReLU激活

# 空间残差块
class SpatialResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        padding = kernel_size // 2
        # 第一个2D卷积核，用于空间特征提取
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二个2D卷积核
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 残差支路：1x1卷积或恒等映射
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

# SSRN 网络
class SSRN(nn.Module):
    def __init__(self, in_channels, num_classes, p=0.5):
        super().__init__()
        # 两个连续的光谱残差块，输入为1通道（前面会unsqueeze）
        self.spectral_block1 = SpectralResBlock(1, 24)
        self.spectral_block2 = SpectralResBlock(24, 24)
        # 过渡层：将光谱特征聚合成128维空间特征
        self.transition = nn.Conv3d(24, 128, kernel_size=(in_channels, 1, 1))  # 融合所有光谱维度

        # 两个空间残差块，提取空间特征
        self.spatial_block1 = SpatialResBlock(128, 128)
        self.spatial_block2 = SpatialResBlock(128, 128)
        # 平均池化，将空间特征压缩为 1x1
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 分类器部分：Flatten -> Linear -> ReLU -> Dropout -> Linear
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(64, num_classes)
        )

    def forward(self, spectral_input, spatial_input=None):
        x = spectral_input.unsqueeze(1)  # 输入: [B, C, H, W] → [B, 1, C, H, W]
        # 光谱残差提取
        x = self.spectral_block1(x)
        x = self.spectral_block2(x)
        # 聚合光谱维度 → [B, 128, 1, H, W]
        x = self.transition(x)  # [B, 128, 1, H, W]
        # 去掉中间的“1”维度 → [B, 128, H, W]
        x = x.squeeze(2)        # [B, 128, H, W]
        # 空间残差提取
        x = self.spatial_block1(x)
        x = self.spatial_block2(x)
        # 自适应池化到1×1大小 → [B, 128, 1, 1]
        x = self.pool(x)        # [B, 128, 1, 1]
        # 输入分类器，输出结果
        x = self.classifier(x)
        return x
