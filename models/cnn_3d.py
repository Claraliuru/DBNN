import torch
import torch.nn as nn
import torch.nn.functional as F

# 3D卷积模块
class SimpleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x))

# 只包含3D卷积的光谱处理分支
class Spectral3DCNN(nn.Module):
    def __init__(self, in_channels, num_classes, p):
        super().__init__()
        self.conv1 = SimpleConv3D(in_channels, 32)
        self.conv2 = SimpleConv3D(32, 128)
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # 输出形状变为 [B, 128, 1, 1, 1]

        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, spatial_input=None):
        x = x.unsqueeze(2)  # 将 [B, C, H, W] 转为 [B, C, 1, H, W] 以适配3D卷积
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # 展平为 [B, 128]
        out = self.classifier(x)
        return out
