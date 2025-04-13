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

# 光谱分支
class SpectralBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.msconv1 = MultiScaleConv3D(in_channels, 32)
        self.msconv2 = MultiScaleConv3D(32, 64)
        self.msconv3 = MultiScaleConv3D(64, 128)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = x.unsqueeze(2)  # (B, C, 1, H, W)
        x = self.msconv1(x)
        x = self.msconv2(x)
        x = self.msconv3(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)

# 多尺度2D卷积模块
class MultiScaleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv_fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x_concat = torch.cat([x1, x2, x3], dim=1)
        return self.conv_fuse(x_concat)

# 空间分支
class SpatialBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.msconv1 = MultiScaleConv2D(in_channels, 32)
        self.msconv2 = MultiScaleConv2D(32, 64)
        self.msconv3 = MultiScaleConv2D(64, 128)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True), num_layers=2
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.msconv1(x)
        x = self.msconv2(x)
        x = self.msconv3(x)
        x = self.global_pool(x)            # (B, 128, 1, 1)
        x = x.view(x.size(0), 1, -1)       # (B, 1, 128)
        x = self.transformer(x)           # Transformer expects (B, Seq, D)
        return x.squeeze(1)               # (B, 128)

# 最终的双分支分类网络（不含注意力、不含Dropout）
class DBNN_noatt(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.spectral_branch = SpectralBranch(in_channels)
        self.spatial_branch = SpatialBranch(in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, spectral_input, spatial_input):
        spectral_feat = self.spectral_branch(spectral_input)  # (B, 128)
        spatial_feat = self.spatial_branch(spatial_input)     # (B, 128)
        fused_feat = spectral_feat + spatial_feat             # 简单相加融合
        output = self.classifier(fused_feat)
        return output
