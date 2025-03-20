# main.py
import torch
import yaml
from torch.utils.data import DataLoader
from datasets.data_loader import HyperspectralDataset
from models.dbnn import DBNN
import numpy as np
from utils.helper import save_model, load_model
from utils.evaluate import evaluate_model
from utils.visualize import generate_classification_map, visualize_ground_truth_and_classification
import matplotlib.pyplot as plt
import os

# 读取配置
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # 加载数据集
    train_dataset = HyperspectralDataset(
        data_path=config["DATA_PATH"],
        label_path=config["LABEL_PATH"],
        patch_size=config["PATCH_SIZE"],
        train=True  # 加载训练集
    )
    in_channels = train_dataset.num_channels
    num_classes = train_dataset.num_classes

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True)

    # 初始化模型
    model = DBNN(in_channels=in_channels, num_classes=num_classes).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])  # 定义Adam优化器

    # 训练模型
    model.train()
    acc = []
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
        acc.append(0.01 * accuracy)
        print(f"Epoch [{epoch+1}/{config['EPOCHS']}], Loss: {running_loss: .4f}, Accuracy: {accuracy:.2f}%")

        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            save_model(model, os.path.join(config["MODEL_PATH"], f"dbnn_epoch_{epoch+1}.pth"))

    print("训练完成！")
 
    output_dir = config["FIGURE_DIR"]
    data_name = config["DATA_NAME"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.figure(figsize=(10,6))
    plt.plot(range(1, epoch + 2), acc, label='Train Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training over Epochs')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.2, 1.0)
    plt.savefig(os.path.join(output_dir, f'{data_name}_accuracy_plot.png'))  # 保存准确率折线图
    plt.show()

def evaluate():
    # 加载测试数据集
    test_dataset = HyperspectralDataset(
        data_path=config["DATA_PATH"],
        label_path=config["LABEL_PATH"],
        patch_size=config["PATCH_SIZE"],
        train=False
    )
    in_channels = test_dataset.num_channels
    num_classes = test_dataset.num_classes
    print(f"测试参数：{in_channels}, {num_classes}")
    test_loader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"], shuffle=False)

    # 初始化模型并加载训练好的权重
    model = DBNN(in_channels=in_channels, num_classes=num_classes).to(DEVICE)
    load_model(model, config["MODEL_PATH_TEST"])

    # 评估模型
    OA, AA, kappa = evaluate_model(model, test_loader, DEVICE, num_classes)
    print(f"OA (Overall Accuracy): {OA:.4f}")
    print(f"AA (Average Accuracy): {AA:.4f}")
    print(f"Kappa Coefficient: {kappa:.4f}")

    # 评估模型并绘制准确率折线图
    model.eval()
    accuracy_list = []  # 用来保存每个batch的准确率

    with torch.no_grad():
        for inputs, spatial_input, labels in test_loader:
            inputs, spatial_input, labels = inputs.to(DEVICE), spatial_input.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs, spatial_input)
            
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / labels.size(0)
            accuracy_list.append(0.01 * accuracy)
def visualize():
    # 加载测试数据集
    test_dataset = HyperspectralDataset(
        data_path=config["DATA_PATH"],
        label_path=config["LABEL_PATH"],
        patch_size=config["PATCH_SIZE"],
        train=False
    )

    in_channels = test_dataset.num_channels
    num_classes = test_dataset.num_classes

    # 初始化模型并加载训练好的权重
    model = DBNN(in_channels=in_channels, num_classes=num_classes).to(DEVICE)
    checkpoint_dir = config["MODEL_PATH_TEST"]
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    load_model(model, checkpoint_dir)

    # 生成分类图
    classification_map = generate_classification_map(model, test_dataset, DEVICE)
    ground_truth = test_dataset.labels
    visualize_ground_truth_and_classification(ground_truth, classification_map)

if __name__ == "__main__":
    train()
    evaluate()
    visualize()
