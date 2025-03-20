import numpy as np
import torch
from sklearn.metrics import confusion_matrix

def calculate_accuracy(predictions, labels, num_classes):

    # 将预测结果和平面标签拉成一维
    predictions = predictions.flatten()
    labels = labels.flatten()

    # 创建混淆矩阵
    cm = confusion_matrix(labels, predictions, labels=np.arange(num_classes))

    # 计算OA
    OA = np.trace(cm) / np.sum(cm)

    # 计算AA
    AA = np.mean(np.diag(cm) / np.maximum(cm.sum(axis=1), 1e-8))  # 防止除以零


    # 计算Kappa
    total = np.sum(cm)
    row_sum = np.sum(cm, axis=1)
    col_sum = np.sum(cm, axis=0)
    expected = np.outer(row_sum, col_sum) / total
    kappa_numerator = total * np.trace(cm) - np.sum(expected)
    kappa_denomintor = total**2 - np.sum(expected)
    kappa = kappa_numerator / kappa_denomintor

    return OA, AA, kappa

def evaluate_model(model, test_loader, device, num_classes):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for inputs, spatial_input, labels in test_loader:
            inputs, spatial_input, labels = inputs.to(device), spatial_input.to(device),labels.to(device)

            outputs = model(inputs, spatial_input)
            _, predicted = torch.max(outputs, 1)

            all_predictions.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 拼接所有批次的预测结果与真实标签
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    # 计算并返回OA, AA, Kappa
    OA, AA, kappa = calculate_accuracy(all_predictions, all_labels, num_classes)
    return OA, AA, kappa