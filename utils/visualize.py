import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets.data_loader import HyperspectralDataset
from models.dbnn import DBNN
from utils.helper import load_model
import yaml
import os
# 读取配置
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载测试数据
test_dataset = HyperspectralDataset(
    data_path=config["DATA_PATH"],
    label_path=config["LABEL_PATH"],
    patch_size=config["PATCH_SIZE"],
    train=False
)

# 加载模型
model = DBNN(in_channels=test_dataset.num_channels, num_classes=test_dataset.num_classes).to(DEVICE)
model_path = config["MODEL_PATH_TEST"]
load_model(model, model_path)

# 生成分类结果图像
def generate_classification_map(model, dataset, device):
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
                    patch = torch.tensor(patch).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, C, H, W]
                    output = model(patch, patch)
                    _, predicted = torch.max(output, 1)
                    classification_map[x, y] = predicted.item()

    return classification_map

# 可视化真值图和分类结果图
def visualize_ground_truth_and_classification(ground_truth, classification_map):
    plt.figure(figsize=(12, 6))

    # 真值图
    plt.subplot(1, 2, 1)
    plt.imshow(ground_truth, cmap='jet', vmin=0, vmax=np.max(ground_truth))
    plt.title("(a)", y=-0.1)
    plt.axis('off')

    # 分类结果图
    output_dir = config["FIGURE_DIR"]
    data_name = config["DATA_NAME"]
    plt.subplot(1, 2, 2)
    plt.imshow(classification_map, cmap='jet', vmin=0, vmax=np.max(ground_truth))
    plt.title("(b)", y=-0.1)
    plt.axis('off')

    plt.savefig(os.path.join(output_dir, f'{data_name}_figure.png'))
    plt.show()