import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.data_loader import HyperspectralDataset
from models.dbnn import DBNN
from utils import save_model
import yaml
import os 

# 读取配置
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device is {DEVICE}.")

# 加载数据集
train_dataset = HyperspectralDataset(
    data_path=config["DATA_PATH"],
    label_path=config["LABEL_PATH"],
    patch_size=config["PATCH_SIZE"],
    train=True # 加载训练集
)
in_channels = train_dataset.num_channels
num_classes = train_dataset.num_classes

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True)

# 初始化模型
model = DBNN(in_channels=in_channels, num_classes=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss() # 定义交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"]) # 定义Adam优化器

# 训练模型
def train():
    # 加载数据集
    train_dataset = HyperspectralDataset(
        data_path=config["DATA_PATH"],
        label_path=config["LABEL_PATH"],
        patch_size=config["PATCH_SIZE"],
        train=True
    )
    in_channels = train_dataset.num_channels
    num_classes = train_dataset.num_classes

    # 打印一个样本的形状
    sample, label = train_dataset[0]
    print(f"样本形状：{sample.shape}")  # 预期输出: [height, width, num_channels]

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True)

    # 初始化模型
    model = DBNN(in_channels=in_channels, num_classes=num_classes).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])  # 定义Adam优化器

    # 训练模型
    model.train()
    for epoch in range(config["EPOCHS"]):
        running_loss = 0.0  # 累计损失值
        correct, total = 0, 0  # 正确样本数和总样本数

        for inputs, spatial_input, labels in train_loader:
            inputs, spatial_input, labels = inputs.to(DEVICE), spatial_input.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs, spatial_input)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            running_loss += loss.item()  # 累加损失值
            _, predicted = torch.max(outputs, 1)  # 获取预测结果
            total += labels.size(0)  # 累加总样本数
            correct += (predicted == labels).sum().item()  # 累计预测正确的样本数

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{config['EPOCHS']}], Loss: {running_loss: .4f}, Accuracy: {accuracy:.2f}%")

        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            save_model(model, os.path.join(config["MODEL_PATH"], f"dbnn_epoch_{epoch+1}.pth"))

    print("训练完成！")
