import numpy as np
import torch
from sklearn.metrics import confusion_matrix

def calculate_accuracy(predictions, labels, nnum_classes):
    
    # 将预测结果和评价标签拉成一维
    predictions = predictions.flatten()
    labels = labels.flatten()

    # 创建混淆矩阵
    cm = confusion_matrix(labels, predictions, labels=np.arange(nnum_classes))

    # 计算OA
    OA = np.trace(cm) / np.sum(cm)

    # 计算AA
    AA = np.mean(np.diag(cm) / np.maximum(cm.sum(axis=1), 1e-8)) # 防止除以0

    # 计算Kappa
    total = np.sum(cm)
    row_sum = np.sum(cm, axis=1) # 每类真实值总数
    col_sum = np.sum(cm, axis=0) # 每类预测值总数
    expected = np.outer(row_sum, col_sum) / total # 期望混淆矩阵（随机情况下）

    kappa_numerator = total * np.trace(cm) - np.sum(expected)
    # 分子：实际一致的样本数量（对角线之和）与“随机一致性期望”的差值。
    # np.trace(cm): 实际预测正确的样本数之和
    # np.sum(expected): 随机情况下，理论上预测正确的样本数（期望一致性）
    # total * np.trace(cm): 放大实际一致数量，与期望一致数量做差
    kappa_denomintor = total**2 - np.sum(expected)
    # 分母：在完全一致的理想状态下与期望一致状态之间的最大可能差值
    # total**2: 理论上所有样本都被正确预测的极限情况
    # np.sum(expected): 随机预测下的期望一致数量
    kappa = kappa_numerator / kappa_denomintor
    # 最终的 Kappa 系数：实际一致性改进程度 / 最大可能改进程度
    # 结果范围为 [-1, 1]，越接近1表示一致性越好，0表示与随机无异，负数表示比随机还差
    return OA, AA, kappa

def evaluate(model, test_loader, device, num_classes):
    model.to(device)
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad(): # 禁用梯度计算，加速推理，节省内存

        for  inputs, spatial_input, labels in test_loader:
            # 将数据移到对应设备
            inputs, spatial_input, labels = inputs.to(device), spatial_input.to(device),labels.to(device)

            # 前向传播，得到输出
            outputs = model(inputs, spatial_input)
            # 获取每个样本预测的类别（取最大值对应的索引）
            _, predicted = torch.max(outputs, 1)

            # 将当前 batch 的预测和真实标签转为 numpy 格式，保存到列表中
            all_predictions.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 拼接所有批次的预测结果与真实标签
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    # 计算并返回OA, AA, Kappa
    OA, AA, kappa = calculate_accuracy(all_predictions, all_labels, num_classes)
    return OA, AA, kappa