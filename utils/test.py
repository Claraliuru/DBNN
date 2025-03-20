import torch
from torch.utils.data import DataLoader
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
in_channels = test_dataset.num_channels
num_classes = test_dataset.num_classes

test_loader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"], shuffle= False)

# 加载模型
model = DBNN(in_channels=in_channels, num_classes=num_classes).to(DEVICE)
model_path = config["MODEL_PATH_TEST"]
print(f"地址：{model_path}")
load_model(model, model_path)

print("模型路径:", model_path)
print("文件是否存在:", os.path.exists(model_path))
print("是文件吗？", os.path.isfile(model_path))

# 测试模型
def test():
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels  =inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"测试集准确率：{accuracy:.2f}%")
