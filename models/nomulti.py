""" 单一3x3尺度卷积"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 2d残差块
class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
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

# 3D残差块
class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
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

class SpectralBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_input = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
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

class SpatialBranch(nn.Module):
    def __init__(self, in_channels, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.guided_filter = GuidedFilter(radius=2, eps=1e-3)
        self.conv_input = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.res1 = ResBlock2D(32, 64)
        self.res2 = ResBlock2D(64, embed_dim)

        self.fuse_guided = nn.Conv2d(1 + 32, 32, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        with torch.no_grad():
            gray = x.mean(dim=1, keepdim=True)
            guided_out = self.guided_filter(gray, gray)

        x = self.conv_input(x)  # [B, 32, H, W]
        x = torch.cat([x, guided_out], dim=1)  # [B, 33, H, W]
        x = self.fuse_guided(x)  # [B, 32, H, W]

        x = self.res1(x)
        x = self.res2(x)

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return x, gray, guided_out

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attn_spectral = nn.Linear(feature_dim, 1)
        self.attn_spatial = nn.Linear(feature_dim, 1)

    def forward(self, spectral_feat, spatial_feat):
        attn_s = torch.sigmoid(self.attn_spectral(spectral_feat))
        attn_p = torch.sigmoid(self.attn_spatial(spatial_feat))
        fused_feat = attn_s * spectral_feat + attn_p * spatial_feat
        return fused_feat

class NOMULTI(nn.Module):
    def __init__(self, in_channels, num_classes, p=0.5):
        super().__init__()
        self.spectral_branch = SpectralBranch(in_channels)
        self.spatial_branch = SpatialBranch(in_channels)
        self.attention_fusion = AttentionFusion(128)
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
        fused_feat = self.attention_fusion(spectral_feat, spatial_feat)
        output = self.classifier(fused_feat)
        if return_gray:
            return output, gray_before, gray_after
        return output
