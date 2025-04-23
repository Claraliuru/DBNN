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
        x = x.unsqueeze(2)  # [B, C, 1, H, W]
        x = self.msconv1(x)
        x = self.msconv2(x)
        x = self.msconv3(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)

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
    def __init__(self, in_channels, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.guided_filter = GuidedFilter(radius=2, eps=1e-3)
        self.msconv1 = MultiScaleConv2D(in_channels, 32)
        self.msconv2 = MultiScaleConv2D(32, 64)
        self.msconv3 = MultiScaleConv2D(64, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, C, H, W]
        with torch.no_grad():
            gray = x.mean(dim=1, keepdim=True)
            guided_out = self.guided_filter(gray, gray)

        x = self.msconv1(x)
        x = self.msconv2(x)
        x = self.msconv3(x)

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x = self.transformer(x)
        x = x.mean(dim=1)

        return x, gray, guided_out  # 返回灰度图和引导滤波图用于可视化

# 注意力融合模块
class AttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.attn_spectral = nn.Linear(feature_dim, 1)
        self.attn_spatial = nn.Linear(feature_dim, 1)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

    def forward(self, spectral_feat, spatial_feat):
        attn_s = torch.sigmoid(self.attn_spectral(spectral_feat))
        attn_p = torch.sigmoid(self.attn_spatial(spatial_feat))
        fused_feat = attn_s * spectral_feat + attn_p * spatial_feat
        fused_feat, _ = self.multihead_attn(fused_feat.unsqueeze(0), fused_feat.unsqueeze(0), fused_feat.unsqueeze(0))
        return fused_feat.squeeze(0)

# 完整模型
class DBNN_gf(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.spectral_branch = SpectralBranch(in_channels)
        self.spatial_branch = SpatialBranch(in_channels)
        self.attention_fusion = AttentionFusion(128)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
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