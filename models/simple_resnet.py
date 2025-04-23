import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# 光谱分支：单尺度 3D CNN（保持不变）
class SimpleSpectralBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = x.unsqueeze(2)  # [B, C, 1, H, W]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        return x.view(x.size(0), -1)

# 空间分支：使用 ResNet18 提取空间特征
class ResNetSpatialBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 使用 torchvision 提供的 ResNet18，并调整输入层和输出层
        base_model = resnet18(weights=None)
        if in_channels != 3:
            # 修改第一层输入通道数（默认是3）
            base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])  # 去掉最后的全连接层和全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 保持输出一致
        self.output_dim = 512  # ResNet18 的最后一个特征层是 512

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)

# 简化版 DBNN：拼接两个分支特征后分类
class Simple_Resnet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.spectral_branch = SimpleSpectralBranch(in_channels)
        self.spatial_branch = ResNetSpatialBranch(in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(128 + self.spatial_branch.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, spectral_input, spatial_input):
        spectral_feat = self.spectral_branch(spectral_input)
        spatial_feat = self.spatial_branch(spatial_input)
        fused_feat = torch.cat([spectral_feat, spatial_feat], dim=1)
        output = self.classifier(fused_feat)
        return output
