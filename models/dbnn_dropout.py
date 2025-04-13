import torch
import torch.nn as nn
import torch.nn.functional as F

# 多尺度3D卷积模块（加了BatchNorm和Dropout）
class MultiScaleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm3d(out_channels)
        
        self.conv_fuse = nn.Conv3d(out_channels * 3, out_channels, kernel_size=1)
        self.dropout = nn.Dropout3d(p=0.3)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        x3 = F.relu(self.bn3(self.conv3(x)))
        x_concat = torch.cat([x1, x2, x3], dim=1)
        x_concat = self.dropout(x_concat)
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
        x = x.unsqueeze(2)  # 添加深度维度
        x = self.msconv1(x)
        x = self.msconv2(x)
        x = self.msconv3(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)

# 多尺度2D卷积模块（加了BatchNorm和Dropout）
class MultiScaleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv_fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        x3 = F.relu(self.bn3(self.conv3(x)))
        x_concat = torch.cat([x1, x2, x3], dim=1)
        x_concat = self.dropout(x_concat)
        return self.conv_fuse(x_concat)

# 空间分支（加了Dropout）
class SpatialBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.msconv1 = MultiScaleConv2D(in_channels, 32)
        self.msconv2 = MultiScaleConv2D(32, 64)
        self.msconv3 = MultiScaleConv2D(64, 128)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, dropout=0.3, batch_first=True),
            num_layers=2
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
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
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)

    def forward(self, spectral_feat, spatial_feat):
        attn_s = torch.sigmoid(self.attn_spectral(spectral_feat))
        attn_p = torch.sigmoid(self.attn_spatial(spatial_feat))
        fused_feat = attn_s * spectral_feat + attn_p * spatial_feat
        fused_feat, _ = self.multihead_attn(fused_feat.unsqueeze(1), fused_feat.unsqueeze(1), fused_feat.unsqueeze(1))
        return fused_feat.squeeze(1)

# 双分支神经网络（DBNN）加了Dropout
class DBNN_dropout(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.spectral_branch = SpectralBranch(in_channels)
        self.spatial_branch = SpatialBranch(in_channels)
        self.attention_fusion = AttentionFusion(128)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, spectral_input, spatial_input):
        spectral_feat = self.spectral_branch(spectral_input)
        spatial_feat = self.spatial_branch(spatial_input)
        fused_feat = self.attention_fusion(spectral_feat, spatial_feat)
        output = self.classifier(fused_feat)
        return output
