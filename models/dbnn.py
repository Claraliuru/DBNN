import torch
import torch.nn as nn
import torch.nn.functional as F

# 引导滤波模块
class GuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8):
        super(GuidedFilter, self).__init__()
        self.radius = radius
        self.eps = eps

    def forward(self, x, guide):
        # 简化实现，实际可以使用更高效的引导滤波算法
        return x

# 多尺度3D卷积模块
class MultiScaleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)  # 1×1
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)  # 3×3
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)  # 5×5
        self.conv_fuse = nn.Conv3d(out_channels * 3, out_channels, kernel_size=1)  # 通道融合

    def forward(self, x):
        x1 = F.relu(self.conv1(x))  # [batch, out_channels, depth, height, width]
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x_concat = torch.cat([x1, x2, x3], dim=1)  # 拼接不同尺度特征
        return self.conv_fuse(x_concat)  # 1x1x1 卷积进行通道融合

# 光谱分支
class SpectralBranch(nn.Module):
    def __init__(self, in_channels):
        super(SpectralBranch, self).__init__()
        self.msconv1 = MultiScaleConv3D(in_channels, 32)
        self.msconv2 = MultiScaleConv3D(32, 64)
        self.msconv3 = MultiScaleConv3D(64, 128)
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # 全局池化

    def forward(self, x):
        x = x.unsqueeze(2)  # 增加深度维度 [batch, channels, depth=1, height, width]
        x = self.msconv1(x)
        x = self.msconv2(x)
        x = self.msconv3(x)
        x = self.global_pool(x)  # [batch, 128, 1, 1, 1]
        return x.view(x.size(0), -1)  # 展平成 [batch, 128]

# 多尺度2D卷积模块
class MultiScaleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv_fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        self.guided_filter = GuidedFilter()  # 添加引导滤波层

    def forward(self, x):
        x1 = F.relu(self.conv1(x))  # [batch, out_channels, height, width]
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x_concat = torch.cat([x1, x2, x3], dim=1)  # 拼接不同尺度特征
        fused_feat = self.conv_fuse(x_concat)
        guided_feat = self.guided_filter(fused_feat, fused_feat)  # 引导滤波增强空间特征
        return guided_feat

# 空间分支
class SpatialBranch(nn.Module):
    def __init__(self, in_channels):
        super(SpatialBranch, self).__init__()
        self.msconv1 = MultiScaleConv2D(in_channels, 32)
        self.msconv2 = MultiScaleConv2D(32, 64)
        self.msconv3 = MultiScaleConv2D(64, 128)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True), num_layers=2
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化

    def forward(self, x):
        x = self.msconv1(x)  # 多尺度卷积 + 引导滤波
        x = self.msconv2(x)
        x = self.msconv3(x)
        x = self.global_pool(x)  # [batch, 128, 1, 1]
        x = x.view(x.size(0), 1, -1)  # [batch, 1, 128] -> 符合 Transformer 输入要求
        x = self.transformer(x)  # 通过 Transformer 编码
        return x.squeeze(1)  # [batch, 128]

# 注意力融合模块
class AttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super(AttentionFusion, self).__init__()
        self.attn_spectral = nn.Linear(feature_dim, 1)  # 光谱特征注意力权重
        self.attn_spatial = nn.Linear(feature_dim, 1)  # 空间特征注意力权重
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

    def forward(self, spectral_feat, spatial_feat):
        attn_s = torch.sigmoid(self.attn_spectral(spectral_feat))  # 计算光谱特征注意力权重
        attn_p = torch.sigmoid(self.attn_spatial(spatial_feat))  # 计算空间特征注意力权重
        fused_feat = attn_s * spectral_feat + attn_p * spatial_feat  # 加权融合
        # 使用多头注意力进一步增强特征
        fused_feat, _ = self.multihead_attn(fused_feat.unsqueeze(0), fused_feat.unsqueeze(0), fused_feat.unsqueeze(0))
        return fused_feat.squeeze(0)

# 双分支神经网络
class DBNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DBNN, self).__init__()
        self.spectral_branch = SpectralBranch(in_channels)  # 光谱分支
        self.spatial_branch = SpatialBranch(in_channels)  # 空间分支
        self.attention_fusion = AttentionFusion(128)  # 注意力融合模块
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # 增加一层全连接
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, spectral_input, spatial_input):
        spectral_feat = self.spectral_branch(spectral_input)  # 提取光谱特征
        spatial_feat = self.spatial_branch(spatial_input)  # 提取空间特征
        fused_feat = self.attention_fusion(spectral_feat, spatial_feat)  # 融合特征
        output = self.classifier(fused_feat)  # 分类
        return output