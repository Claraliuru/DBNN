"""对比模型：SVM+PCA"""

import yaml
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
import matplotlib.pyplot as plt
import os

def load_hsi_data(data_path, label_path):
    # 数据加载
    data = sio.loadmat(data_path) # 加载数据
    label = sio.loadmat(label_path) # 加载标签

    data_key = list(data.keys())[-1] # 取出数据键
    label_key = list(label.keys())[-1] # 取出标签键

    data = data[data_key] # 获取实际数据
    labels = label[label_key] # 获取实际标签

    return data.astype(np.float32), labels.astype(np.int64) # 转换为适当类型

def apply_pca(data, n_components):
    h, w, c = data.shape # 获取图像高宽和通道数
    reshaped = data.reshape(-1, c) # 变为二维 [N, C]，N=H×W
    pca = PCA(n_components=n_components) # 创建 PCA 对象
    reduced = pca.fit_transform(reshaped) # 进行 PCA 降维
    return reduced.reshape(h, w, n_components) # 恢复为三维 [H, W, C']

def split_train_test(X, y, train_split=0.1):
    np.random.seed(0) # 固定随机种子，便于复现
    h, w, c = X.shape
    X_flat = X.reshape(-1, c) # 拉平成 [N, C]
    y_flat = y.flatten() # 标签也拉平成一维

    valid_indices = np.where(y_flat > 0)[0] # 找到所有有标注的位置（非0）
    num_train = int(train_split * len(valid_indices)) # 训练样本数量
    perm = np.random.permutation(valid_indices) # 随机打乱索引
    train_idx = perm[:num_train] # 训练索引
    test_idx = perm[num_train:] # 测试索引

    X_train, y_train = X_flat[train_idx], y_flat[train_idx] # 划分训练集
    X_test, y_test = X_flat[test_idx], y_flat[test_idx] # 划分测试集
    
    return X_train, y_train, X_test, y_test, y_flat, test_idx

def classify_svm(X_train, y_train, X_test):
    clf = SVC(C=100, gamma='scale', kernel='rbf', class_weight='balanced') # 初始化支持向量机（RBF核）
    clf.fit(X_train, y_train) # 拟合训练数据
    preds = clf.predict(X_test) # 预测测试数据
    return clf, preds

def evaluate(preds, y_test):
    oa = accuracy_score(y_test, preds) # OA
    kappa = cohen_kappa_score(y_test, preds) # Kappa 系数
    report = classification_report(y_test, preds, digits=4, output_dict=True) # 每类的详细报告
    aa = np.mean([report[str(label)]['recall'] for label in np.unique(y_test)]) # Average Accuracy
    return oa, aa, kappa

def visualize_result(y_pred_all, gt_shape, save_path=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(y_pred_all.reshape(gt_shape), cmap='jet') # 显示预测图
    plt.axis('off')  # 关闭坐标轴
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) # 创建输出目录
        plt.savefig(save_path, bbox_inches='tight') # 保存图像
    plt.close()

def get_dataset_config(dataset_name, config):
    """获取指定数据集配置"""
    for ds in config["datasets"]:
        if ds["name"] == dataset_name:
            return ds # 找到匹配的数据集配置
    raise ValueError(f"数据集{dataset_name} 不存在于配置中")

def main():
    with open("configs/config.yaml", "r", encoding="Utf-8") as f:
        config = yaml.safe_load(f)
    for dataset in config["datasets"]:
        dataset_name = dataset["name"]
        dataset_config = get_dataset_config(dataset_name, config)
        print(f"\n=== 当前数据集：{dataset_name} ===")

        data_path = dataset_config["data_path"]
        label_path = dataset_config["label_path"]
        
        data, labels = load_hsi_data(data_path, label_path)
        data_pca = apply_pca(data, dataset_config["n_components"])

        oa_list, aa_list, kappa_list = [], [], []

        for run in range(5):
            print(f"\n-- 第 {run+1} 次运行 --")
            X_train, y_train, X_test, y_test, y_all, test_idx = split_train_test(data_pca, labels, train_split=dataset_config["train_split"])

            _, preds = classify_svm(X_train, y_train, X_test)
            oa, aa, kappa = evaluate(preds, y_test)

            oa_list.append(oa)
            aa_list.append(aa)
            kappa_list.append(kappa)
            
            model_name = "SVM"
            # 只保存第一次的可视化图
            if run == 0:
                h, w = labels.shape
                y_pred_all = np.zeros(h * w, dtype=int)
                y_pred_all[test_idx] = preds
                visualize_result(y_pred_all, labels.shape,
                                 save_path=f"{dataset_config['figure_dir']}/svm_pca.png")
                result = {
                    "train_split": dataset_config["train_split"],
                    "weight_decay": dataset_config["weight_decay"],
                    "OA": float(oa),
                    "AA": float(aa),
                    "Kappa": float(kappa),
                }
                # 尝试从已有文件中读取数据
                if os.path.exists(config["model_file"]):
                    with open(config["model_file"], "r", encoding="utf-8") as file:
                        model_data = yaml.safe_load(file) or {}
                else:
                    model_data = {}

                # 确保结构存在
                if dataset_name not in model_data:
                    model_data[dataset_name] = {}
                if model_name not in model_data[dataset_name]:
                    model_data[dataset_name][model_name] = []

                # 添加新结果
                model_data[dataset_name][model_name].append(result)

                # 写回文件
                with open(config["model_file"], "w", encoding="utf-8") as file:
                    yaml.dump(model_data, file, allow_unicode=True)
            print(f"[SVM on {dataset_name}]:\nOA: {oa:.4f}, AA: {aa:.4f}, Kappa: {kappa:.4f}")

        print("\n--- 五次运行的平均结果 ---")
        print(f"平均 OA: {np.mean(oa_list):.4f}")
        print(f"平均 AA: {np.mean(aa_list):.4f}")
        print(f"平均 Kappa: {np.mean(kappa_list):.4f}")

if __name__ == "__main__":
    main()
