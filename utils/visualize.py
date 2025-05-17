import numpy as np
import torch
import matplotlib.pyplot as plt
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
        classification_map = np.zeros((h, w), dtype=np.int64) # 初始化， H,W,C

        with torch.no_grad():
            # 遍历整张图像中所有像素位置。若该位置没有标签（背景），则预测结果设为 0
            for x in range(h):
                for y in range(w):
                    if dataset.label[x, y] == 0:
                        classification_map[x, y] = 0
                    else:
                        #否则提取包含该像素的 patch，并转成 PyTorch tensor 格式（转置为 [C, H, W]，再扩展为 batch 维度 [1, C, H, W]）。
                        patch = dataset.extract_patch(x, y) # 提取补丁
                        patch = torch.tensor(patch).permute(2, 0, 1).unsqueeze(0).to(self.device)  # [1, C, H, W]
                        output = model(patch, patch)
                        _, predicted = torch.max(output, 1)
                        classification_map[x, y] = predicted.item()

        return classification_map
    
    def save_single_map(self, img, title="", save_path=None):
        """
        保存单张分类图或真值图
        img: 图像矩阵
        title: 图像标题
        save_path: 保存路径（含文件名）
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='jet', vmin=0, vmax=np.max(img))
        if title:
            plt.title(title, y=-0.1)
        plt.axis('off')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def visualize_comparison(self, ground_truth, cls_map, save_name, save_dir=None):
        """
        真值图 + 多个模型的分类图
        model_maps: dict，键为模型名，值为分类图（H, W）
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        gt_path = os.path.join(save_dir, "ground_truth.png") if save_dir else None
        self.save_single_map(ground_truth, save_path=gt_path)

        model_path = os.path.join(save_dir, f"{save_name}.png") if save_dir else None
        self.save_single_map(cls_map, save_path=model_path)
    