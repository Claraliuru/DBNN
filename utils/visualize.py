import numpy as np
import torch
import string
import matplotlib.pyplot as plt
from data_lodaer.loader import HSI
from models.dbnn import DBNN
from utils.helper import load_model
import yaml
import os


class Visualize:
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def generate_classification_map(self, model, dataset):
        """
        生成分类结果图
        model: 训练好的模型
        dataset: 数据集对象
        return: 分类结果图（H, W）
        """
        model.to(self.device)
        model.eval()
        h, w, _ = dataset.data.shape
        classification_map = np.zeros((h, w), dtype=np.int64) # 初始化

        with torch.no_grad():
            for x in range(h):
                for y in range(w):
                    if dataset.label[x, y] == 0:
                        classification_map[x, y] = 0
                    else:
                        patch = dataset.extract_patch(x, y) # 提取补丁
                        patch = torch.tensor(patch).permute(2, 0, 1).unsqueeze(0).to(self.device)  # [1, C, H, W]
                        output = model(patch, patch)
                        _, predicted = torch.max(output, 1)
                        classification_map[x, y] = predicted.item()

        return classification_map
    
    def visualize_ground_truth_and_classification(self, ground_truth, classification_map, title="Classification Map"):
        """
        可视化真值图和分类结果图
        ground_truth: 真值图
        classification_map: 分类结果图
        title: 图像标题
        """
        plt.figure(figsize=(12, 6))

        # 真值图
        plt.subplot(1, 2, 1)
        plt.imshow(ground_truth, cmap='jet', vmin=0, vmax=np.max(ground_truth))
        plt.title("(a) Grpund Truth", y=-0.1)
        plt.axis('off')

        # 分类结果图
        plt.subplot(1, 2, 2)
        plt.imshow(classification_map, cmap='jet', vmin=0, vmax=np.max(ground_truth))
        plt.title("(b) {title}", y=-0.1)
        plt.axis('off')

        plt.show()

    def visualize_comparison(self, ground_truth, model_maps:dict):
        """
        真值图 + 多个模型的分类图
        model_maps: dict，键为模型名，值为分类图（H, W）
        """
        num_models = len(model_maps)
        total = num_models + 1 # 模型数加真值图

        plt.figure(figsize=(6 * total, 6))

        if total > 3:
            row = 2
        else:
            row = 1

        alphabet = list(string.ascii_lowercase)
        # 真值图
        plt.subplot(row, total, 1)
        plt.imshow(ground_truth, cmap='jet', vmin=0, vmax=np.max(ground_truth))
        plt.title(f"({alphabet[0]})", y=-0.1)
        plt.axis('off')

        # 模型结果
        for i, (_,cls_map) in enumerate(model_maps.items(), start=2):
            plt.subplot(row, total, i)
            plt.imshow(cls_map, cmap='jet', vmin=0, vmax=np.max(ground_truth))
            plt.title(f"({alphabet[i-1]})", y=-0.1)
            plt.axis('off')
        
        plt.show()    