import torch
import torch.nn as nn
import torch.nn.functional as F

# 单尺度3D卷积模块（仅使用3x3）
class SingleScaleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x))

# 光谱分支
class SpectralBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = SingleScaleConv3D(in_channels, 32)
        self.conv2 = SingleScaleConv3D(32, 64)
        self.conv3 = SingleScaleConv3D(64, 128)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)

# 单尺度2D卷积模块（仅使用3x3）
class SingleScaleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x))

# 空间分支
class SpatialBranch(nn.Module):
    def __init__(self, in_channels, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.msconv1 = SingleScaleConv2D(in_channels, 32)
        self.msconv2 = SingleScaleConv2D(32, 64)
        self.msconv3 = SingleScaleConv2D(64, embed_dim)  # 输出维度要等于 Transformer 输入维度

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.msconv1(x)     # [B, 32, H, W]
        x = self.msconv2(x)     # [B, 64, H, W]
        x = self.msconv3(x)     # [B, 128, H, W]

        B, C, H, W = x.shape
        x = x.view(B, C, H * W)     # [B, C, H*W]
        x = x.permute(0, 2, 1)      # [B, H*W, C]  -> token 序列

        x = self.transformer(x)     # [B, H*W, C] -> 处理空间依赖

        x = x.mean(dim=1)           # 所有 token 平均池化 -> [B, C]
        return x


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
class DBNN_3x3(nn.Module):
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
    
    def forward(self, spectral_input, spatial_input):
        spectral_feat = self.spectral_branch(spectral_input)
        spatial_feat = self.spatial_branch(spatial_input)
        fused_feat = self.attention_fusion(spectral_feat, spatial_feat)
        output = self.classifier(fused_feat)
        return output
