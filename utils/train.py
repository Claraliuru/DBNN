import torch
import torch.nn as nn
import torch.optim as optim
from utils.helper import save_model
import os

def train(config, device, train_loader, model, model_name, model_path):
    # 创建模型保存目录
    save_dir = os.path.join(model_path, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # 初始化
    model = model.to(device)
    model.train()
    epoch = 0
    min_loss = float("inf")
    max_epochs = config["max_epochs"]
    learning_rate = config["learning_rate"]
    
    print(f"[Train Start] learning_rate: {learning_rate}.")

    while epoch < max_epochs:
        running_loss = 0.0
        correct, total = 0, 0
        epoch += 1
        

        for inputs, spatial_input, labels in train_loader:
            inputs, spatial_input, labels = inputs.to(device), spatial_input.to(device), labels.to(device)
            
            # 对于SVMPCA，直接调用其train
            if not hasattr(model, 'parameters'):
                model.train(inputs.cpu().numpy(), labels.cpu().numpy())
                continue
            
            # 清空梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs, spatial_input)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新模型参数
            optimizer.step()

            # 累加损失值
            running_loss += loss.item()
            # 获取预测结果
            _, predicted = torch.max(outputs, 1)
            # 累加总样本数
            total += labels.size(0)
            # 累计预测正确的样本数
            correct += (predicted == labels).sum().item()

        accuuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        print(f"[{model_name}] Epoch: {epoch}, Loss:{avg_loss:.4f}, Accuracy:{accuuracy:.3f}%")

        # 保存最优模型或者每10轮保存一次
        if epoch % 10 == 0 and avg_loss < min_loss:
            save_model(model, os.path.join(save_dir, f"{model_name}_epoch_{epoch}.pth"))
            if avg_loss < min_loss:
                min_loss = avg_loss
        
        if avg_loss < learning_rate and epoch % 10 == 0:
            break
        
    print(f"[Train Done] Final loss: {avg_loss:.4f}, Accuracy: {accuuracy:.2f}%")
    return epoch