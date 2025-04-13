import torch
import torch.nn as nn
import torch.nn.functional as F

# 多尺度3D卷积模块
class MultiScaleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)  # 1×1
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)  # 3×3
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)  # 5×5
        self.conv_fuse = nn.Conv3d(out_channels * 3, out_channels, kernel_size=1)  # 通道融合

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
        x = x.unsqueeze(2)
        x = self.msconv1(x)
        x = self.msconv2(x)
        x = self.msconv3(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)
    
"""空间分支"""
# 引入传统引导滤波
class GuildFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8):
        super().__init__()
        self.radius = radius
        self.eps = eps

    def box_filter(self, x, radius):
        """
        计算均值滤波，使用卷积代替窗口求和
        """
        kernel_size = 2 * radius + 1
        channels = x.shape[1]
        weight = torch.ones(channels, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        return F.conv2d(x, weight, padding=radius, groups=channels)
    
    def forward(self, x, guide):
        """引导滤波计算过程"""
        # 计算均值
        mean_x = self.box_filter(x, self.radius)
        mean_g = self.box_filter(guide, self.radius)
        mean_xg = self.box_filter(x * guide, self.radius)

        # 计算协方差和方差
        cov_xg = mean_xg - mean_x * mean_g
        mean_gg = self.box_filter(guide * guide, self.radius)
        var_g = mean_gg - mean_g * mean_g

        # 计算线性系数a和b
        a = cov_xg / (var_g + self.eps)
        b = mean_x - a * mean_g

        # 计算滤波后的输出
        mean_a = self.box_filter(a, self.radius)
        mean_b = self.box_filter(b, self.radius)
        output = mean_a * guide + mean_b

        return output
    
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

class SpatialBranch_gf(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.guided_filter = GuildFilter(radius=1)
        self.msconv1 = MultiScaleConv2D(in_channels, 32)
        self.msconv2 = MultiScaleConv2D(32, 64)
        self.msconv3 = MultiScaleConv2D(64, 128)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True), num_layers=2
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x, guide):
        """先进行引导滤波，再进入多尺度卷积"""
        x = self.guided_filter(x, guide)  # 预处理
        x = self.msconv1(x)
        x = self.msconv2(x)
        x = self.msconv3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), 1, -1)
        x = self.transformer(x)
        return x.squeeze(1)
    
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
    
# 双分支神经网络
class DBNN_gf(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.spectral_branch = SpectralBranch(in_channels)
        self.spatial_branch = SpatialBranch_gf(in_channels)
        self.attention_fusion = AttentionFusion(128)
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, spectral_input, spatial_input, guide):
        spectral_feat = self.spectral_branch(spectral_input)
        spatial_feat = self.spatial_branch(spatial_input, guide)  # 传入 guide
        fused_feat = self.attention_fusion(spectral_feat, spatial_feat)
        output = self.classifier(fused_feat)
        return output