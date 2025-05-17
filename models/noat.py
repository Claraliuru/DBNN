"""无Attention Fusion模型"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 多尺度2D卷积
class MultiScaleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = torch.cat([x1, x2, x3], dim=1)
        x = self.fuse(x_cat)
        x = self.relu(self.bn(x))
        return x

# 多尺度3D卷积
class MultiScaleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch2 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch3 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.fuse = nn.Conv3d(out_channels * 3, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = torch.cat([x1, x2, x3], dim=1)
        x = self.fuse(x_cat)
        x = self.relu(self.bn(x))
        return x

# 2D残差
class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = MultiScaleConv2D(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MultiScaleConv2D(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

# 3D残差
class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = MultiScaleConv3D(in_channels, out_channels)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MultiScaleConv3D(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

# 引导滤波
class GuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-2):
        super().__init__()
        self.radius = radius
        self.eps = eps

    def box_filter(self, x):
        kernel_size = 2 * self.radius + 1
        return F.avg_pool2d(x, kernel_size, stride=1, padding=self.radius)

    def forward(self, I, p):
        mean_I = self.box_filter(I)
        mean_p = self.box_filter(p)
        mean_Ip = self.box_filter(I * p)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.box_filter(I * I)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I

        mean_a = self.box_filter(a)
        mean_b = self.box_filter(b)

        q = mean_a * I + mean_b
        return q

# 光谱分支
class SpectralBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_input = MultiScaleConv3D(in_channels, 32)
        self.res1 = ResBlock3D(32, 64)
        self.res2 = ResBlock3D(64, 128)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = x.unsqueeze(2)  # [B, C, 1, H, W]
        x = self.conv_input(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)

# 空间分支
class SpatialBranch(nn.Module):
    def __init__(self, in_channels, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.guided_filter = GuidedFilter(radius=2, eps=1e-3)
        self.conv_input = MultiScaleConv2D(in_channels, 32)
        self.res1 = ResBlock2D(32, 64)
        self.res2 = ResBlock2D(64, embed_dim)

        # 融合 guided_out 的通道匹配卷积层
        self.fuse_guided = nn.Conv2d(1 + 32, 32, kernel_size=1)  # 输入通道是拼接后的 1+32

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        with torch.no_grad():
            gray = x.mean(dim=1, keepdim=True)
            guided_out = self.guided_filter(gray, gray)

        x = self.conv_input(x)

        x = torch.cat([x, guided_out], dim=1)
        x = self.fuse_guided(x)

        x = self.res1(x)
        x = self.res2(x)

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return x, gray, guided_out

# 无Attention
class NOAT(nn.Module):
    def __init__(self, in_channels, num_classes, p=0.1):
        super().__init__()
        self.spectral_branch = SpectralBranch(in_channels)
        self.spatial_branch = SpatialBranch(in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(32, num_classes)
        )

    def forward(self, spectral_input, spatial_input, return_gray=False):
        spectral_feat = self.spectral_branch(spectral_input)
        spatial_feat, gray_before, gray_after = self.spatial_branch(spatial_input)
        fused_feat = spectral_feat + spatial_feat # 相加
        output = self.classifier(fused_feat)
        if return_gray:
            return output, gray_before, gray_after
        return output
