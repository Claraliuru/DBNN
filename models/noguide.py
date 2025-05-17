"""无引导滤波模型"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SpatialBranch(nn.Module):
    def __init__(self, in_channels, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.conv_input = MultiScaleConv2D(in_channels, 32)
        self.res1 = ResBlock2D(32, 64)
        self.res2 = ResBlock2D(64, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.conv_input(x)  # [B, 32, H, W]
        x = self.res1(x)
        x = self.res2(x)

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return x

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

class NOGUIDE(nn.Module):
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
        spatial_feat = self.spatial_branch(spatial_input)
        fused_feat = self.attention_fusion(spectral_feat, spatial_feat)
        output = self.classifier(fused_feat)
        return output
