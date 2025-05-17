import torch.nn as nn
import torch.nn.functional as F

# 单一尺度3x3卷积模块
class SimpleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x): # 前向传播：先卷积再ReLU激活
        return F.relu(self.conv(x))

# 单分支网络：只包含2D卷积的光谱处理分支
class Spectral2DCNN(nn.Module):
    def __init__(self, in_channels, num_classes, p):
        super().__init__()
        self.conv1 = SimpleConv2D(in_channels, 128) # 将输入通道转为 128 维特征图
        self.pool = nn.AdaptiveAvgPool2d(1)  # 使用全局平均池化，输出形状变为 [B, 128, 1, 1]

        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, spatial_input=None):
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 展平为 [B, 128]
        out = self.classifier(x)
        return out
