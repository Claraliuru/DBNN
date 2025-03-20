import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets.data_loader import HyperspectralDataset
from models.dbnn import DBNN
from utils.helper import load_model
import yaml
import os

class Visualizer:
    def __init__(self, config):
        """
        初始化 Visualizer
        :param config: 配置文件
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_classification_map(self, model, dataset):
        """
        生成分类结果图
        :param model: 训练好的模型
        :param dataset: 数据集对象
        :return: 分类结果图 (H, W)
        """
        model.eval()
        h, w, _ = dataset.data.shape
        classification_map = np.zeros((h, w), dtype=np.int64)

        with torch.no_grad():
            for x in range(h):
                for y in range(w):
                    if dataset.labels[x, y] == 0:
                        classification_map[x, y] = 0
                    else:
                        patch = dataset.extract_patch(x, y)
                        patch = torch.tensor(patch).permute(2, 0, 1).unsqueeze(0).to(self.device)  # [1, C, H, W]
                        output = model(patch, patch)
                        _, predicted = torch.max(output, 1)
                        classification_map[x, y] = predicted.item()

        return classification_map

    def visualize_ground_truth_and_classification(self, ground_truth, classification_map, title="Classification Map"):
        """
        可视化真值图和分类结果图
        :param ground_truth: 真值图
        :param classification_map: 分类结果图
        :param title: 图像标题
        """
        plt.figure(figsize=(12, 6))

        # 真值图
        plt.subplot(1, 2, 1)
        plt.imshow(ground_truth, cmap='jet', vmin=0, vmax=np.max(ground_truth))
        plt.title("(a) Ground Truth", y=-0.1)
        plt.axis('off')

        # 分类结果图
        plt.subplot(1, 2, 2)
        plt.imshow(classification_map, cmap='jet', vmin=0, vmax=np.max(ground_truth))
        plt.title(f"(b) {title}", y=-0.1)
        plt.axis('off')

        # 保存图像
        output_dir = self.config["FIGURE_DIR"]
        data_name = self.config["DATA_NAME"]
        plt.savefig(os.path.join(output_dir, f'{data_name}_{title.lower().replace(" ", "_")}.png'))
        plt.show()

    def visualize_comparison(self, ground_truth, dnn_map, svm_pca_map):
        """
        可视化真值图、DNN 分类结果图和 SVM+PCA 分类结果图
        :param ground_truth: 真值图
        :param dnn_map: DNN 分类结果图
        :param svm_pca_map: SVM+PCA 分类结果图
        """
        plt.figure(figsize=(18, 6))

        # 真值图
        plt.subplot(1, 3, 1)
        plt.imshow(ground_truth, cmap='jet', vmin=0, vmax=np.max(ground_truth))
        plt.title("(a)", y=-0.1)
        plt.axis('off')

        # SVM+PCA 分类结果图
        plt.subplot(1, 3, 2)
        plt.imshow(svm_pca_map, cmap='jet', vmin=0, vmax=np.max(ground_truth))
        plt.title("(b)", y=-0.1)
        plt.axis('off')

        # DNN 分类结果图
        plt.subplot(1, 3, 3)
        plt.imshow(dnn_map, cmap='jet', vmin=0, vmax=np.max(ground_truth))
        plt.title("(c)", y=-0.1)
        plt.axis('off')



        # 保存图像
        output_dir = self.config["FIGURE_DIR"]
        data_name = self.config["DATA_NAME"]
        plt.savefig(os.path.join(output_dir, f'{data_name}_comparison.png'))
        plt.show()