import torch
import torch.nn as nn
import torch.nn.functional as F

# 多尺度3D卷积模块
class MultiScaleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv_fuse = nn.Conv3d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x_concat = torch.cat([x1, x2, x3], dim=1)
        return self.conv_fuse(x_concat)

# 单分支网络：只包含3D卷积的光谱处理分支
class Spectral3DCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.msconv1 = MultiScaleConv3D(in_channels, 32)
        self.msconv2 = MultiScaleConv3D(32, 64)
        self.msconv3 = MultiScaleConv3D(64, 128)
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # 输出形状变为 [B, 128, 1, 1, 1]

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, spatial_input=None):
        x = x.unsqueeze(2)  # 将 [B, C, H, W] 转为 [B, C, 1, H, W] 以适配3D卷积
        x = self.msconv1(x)
        x = self.msconv2(x)
        x = self.msconv3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # 展平为 [B, 128]
        out = self.classifier(x)
        return out
