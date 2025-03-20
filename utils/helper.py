import torch
import os

def save_model(model, path):
    # 保存Pytorch模型
    # param model: 训练好的模型
    # param path: 保存路径
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"模型已保存：{path}")

def load_model(model, path):
    # 加载Pytorch模型
    # param model: 需要加载参数的模型
    # param path: 模型参数文件路径
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"模型已加载: {path}")
    else:
        print(f"模型文件不存在: {path}")