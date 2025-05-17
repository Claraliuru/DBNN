"""主模型DBNN"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 多尺度卷积：2D
class MultiScaleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 三种不同尺度的卷积核
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        # 用1x1卷积融合三个尺度的特征
        self.fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1) 
        self.bn = nn.BatchNorm2d(out_channels) # 批归一化
        self.relu = nn.ReLU(inplace=True) # 激活函数

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = torch.cat([x1, x2, x3], dim=1) # 沿着通道维拼接
        x = self.fuse(x_cat) # 融合
        x = self.relu(self.bn(x)) # 批归一化 + 激活
        return x # 返回融合后的特征图

# 多尺度卷积：3D
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

# 2D残差模块，使用多尺度卷积代替普通卷积
class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = MultiScaleConv2D(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MultiScaleConv2D(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 如果通道数不一致，则用1x1卷积变换跳跃链接
        self.skip = nn.Sequential() # 跳跃连接默认空
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x) # 残差路径
        out = self.relu(self.bn1(self.conv1(x))) # 主干路径第一层
        out = self.bn2(self.conv2(out)) # 主干路径第二层
        return self.relu(out + identity) # 返回残差加和后激活的结果

# 3D残差模块
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
        self.radius = radius # 盒滤波窗口半径
        self.eps = eps # 稳定因子，避免除零的小常数

    def box_filter(self, x): # 定义和滤波操作，利用平均池化实现
        kernel_size = 2 * self.radius + 1 # 窗口大小
        return F.avg_pool2d(x, kernel_size, stride=1, padding=self.radius) # 平均池化

    def forward(self, I, p):
        # I为引导图，p为输入图像
        mean_I = self.box_filter(I) # 计算I的局部均值
        mean_p = self.box_filter(p) # 计算p的局部均值
        mean_Ip = self.box_filter(I * p) # I*P的局部均值
        cov_Ip = mean_Ip - mean_I * mean_p # 计算协方差

        mean_II = self.box_filter(I * I) # 计算I*I的局部均值
        var_I = mean_II - mean_I * mean_I # 计算方差

        a = cov_Ip / (var_I + self.eps) # 计算线性系数a
        b = mean_p - a * mean_I # 计算线性系数b

        mean_a = self.box_filter(a) # a的局部均值
        mean_b = self.box_filter(b)

        q = mean_a * I + mean_b # 输出滤波结果q
        return q

# 光谱分支，处理三维光谱数据，输出全局光谱特征向量
class SpectralBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_input = MultiScaleConv3D(in_channels, 32) # 输入多尺度3D卷积，输出32通道
        self.res1 = ResBlock3D(32, 64) # 第一个3D残差块，输出64通道
        self.res2 = ResBlock3D(64, 128)  # 第二个3D残差块，输出128通道
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # 全局平均池化，输出尺寸变为1x1x1

    def forward(self, x):
        x = x.unsqueeze(2)  # 变为[B, C, 1, H, W]
        x = self.conv_input(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.global_pool(x) # 全局池化，尺寸变为[B, 128, 1, 1, 1]
        return x.view(x.size(0), -1) # 展平为[B, 128]的特征向量

# 空间分支，输入二维空间特征图，使用引导滤波和Transformer编码器
class SpatialBranch(nn.Module):
    def __init__(self, in_channels, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.guided_filter = GuidedFilter(radius=2, eps=1e-3) # 初始化引导滤波器
        self.conv_input = MultiScaleConv2D(in_channels, 32) # 输入多尺度二维卷积
        self.res1 = ResBlock2D(32, 64) # 2D残差块，输出64通道
        self.res2 = ResBlock2D(64, embed_dim) # 输出embed_dim通道

        # 融合 guided_out 的通道匹配卷积层
        self.fuse_guided = nn.Conv2d(1 + 32, 32, kernel_size=1)  # 用1x1卷积融合引导图和特征图
        # 定义Transformer编码器层，batch_first=True使输入为[batch, seq, feature]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        with torch.no_grad():
            gray = x.mean(dim=1, keepdim=True) # 计算灰度图
            guided_out = self.guided_filter(gray, gray) # 对灰度图进行引导滤波，输出引导图

        x = self.conv_input(x)  # 多尺度二维卷积[B, 32, H, W]
        x = torch.cat([x, guided_out], dim=1)  # 将引导图与卷积特征拼接[B, 33, H, W]
        x = self.fuse_guided(x)  # [B, 32, H, W] -> 融合信息回32通道
        x = self.res1(x)
        x = self.res2(x)
        B, C, H, W = x.shape # 获取张量尺寸
        x = x.view(B, C, H * W).permute(0, 2, 1) # 展平为序列形式 [B, HW, C]
        x = self.transformer(x) # 通过Transformer编码器
        x = x.mean(dim=1) # 对序列长度维度求平均池化
        return x, gray, guided_out  # 返回空间特征及引导滤波前后图

# 注意力融合模块
class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attn_spectral = nn.Linear(feature_dim, 1) # 光谱特征注意力映射
        self.attn_spatial = nn.Linear(feature_dim, 1) # 空间特征注意力映射

    def forward(self, spectral_feat, spatial_feat):
        attn_s = torch.sigmoid(self.attn_spectral(spectral_feat)) # 计算光谱分支注意力分数
        attn_p = torch.sigmoid(self.attn_spatial(spatial_feat)) # 计算空间分支注意力分数
        fused_feat = attn_s * spectral_feat + attn_p * spatial_feat # 加权求和融合特征
        return fused_feat # 返回融合结果

# 整体无PCA模型结构
class NOPCA(nn.Module):
    def __init__(self, in_channels, num_classes, p=0.1):
        super().__init__()
        self.spectral_branch = SpectralBranch(in_channels) # 光谱分支
        self.spatial_branch = SpatialBranch(in_channels) # 空间分支
        self.attention_fusion = AttentionFusion(128) # 融合模块
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

class DBNN(nn.Module):
    def __init__(self, in_channels, num_classes, p=0.5):
        super().__init__()
        self.spectral_branch = SpectralBranch(in_channels)
        self.spatial_branch = SpatialBranch(in_channels)
        self.attention_fusion = AttentionFusion(128)
        # 分类器，3层全连接网络，中间含ReLU激活和Dropout防过拟合
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
        # 如果设置return_gray=True，返回分类结果和灰度图（滤波前后），否则只返回分类结果
        if return_gray:
            return output, gray_before, gray_after
        return output
