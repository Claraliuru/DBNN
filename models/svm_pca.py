import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import joblib
import os

class SVMPCA:
    def __init__(self, n_components=30, background_value=0):
        """
        初始化 SVM+PCA 模型
        :param n_components: PCA 降维后的维度
        :param background_value: 背景标签值
        """
        self.n_components = n_components
        self.background_value = background_value
        self.pca = PCA(n_components=n_components)
        self.svm = SVC(kernel='rbf', C=1.0, gamma='scale')

    def train(self, data, labels):
        """
        训练 SVM+PCA 模型
        :param data: 高光谱数据 (H, W, C) 或 (N, H, W, C)
        :param labels: 标签数据 (H, W) 或 (N, H, W)
        """
        # 处理输入数据形状
        if data.ndim == 3:  # 单幅图像 (H, W, C)
            data = data[np.newaxis, ...]  # 变为 (1, H, W, C)
            labels = labels[np.newaxis, ...]  # 变为 (1, H, W)
        
        # 展平数据 (N*H*W, C)
        data_flat = data.reshape(-1, data.shape[-1])
        
        # 展平标签并过滤背景
        labels_flat = labels.flatten()
        mask = labels_flat != self.background_value
        data_flat = data_flat[mask]
        labels_flat = labels_flat[mask]
        
        # PCA 降维
        self.pca.fit(data_flat)
        data_pca = self.pca.transform(data_flat)
        
        # 训练 SVM
        self.svm.fit(data_pca, labels_flat)

    def predict(self, data):
        """
        预测数据
        :param data: 高光谱数据 (H, W, C) 或 (N, H, W, C)
        :return: 预测标签 (H, W) 或 (N, H, W)
        """
        original_shape = data.shape
        if data.ndim == 3:  # 单幅图像
            data = data[np.newaxis, ...]  # 变为 (1, H, W, C)
            need_squeeze = True
        else:
            need_squeeze = False
        
        # 保存原始背景掩膜（假设第一个波段为0是背景）
        background_mask = (data[..., 0] == 0)
        
        # 展平数据 (N*H*W, C)
        data_flat = data.reshape(-1, data.shape[-1])
        
        # PCA 降维
        data_pca = self.pca.transform(data_flat)
        
        # SVM 预测
        pred_flat = self.svm.predict(data_pca)
        
        # 恢复形状
        pred = pred_flat.reshape(data.shape[:-1])
        
        # 恢复背景区域
        pred[background_mask] = self.background_value
        
        if need_squeeze:
            pred = pred.squeeze(0)
        
        return pred
    
    def evaluate(self, true_labels, pred_labels):
        true_labels = np.array(true_labels).flatten()
        pred_labels = np.array(pred_labels).flatten()

        mask = true_labels != self.background_value
        true_labels = true_labels[mask]
        pred_labels = pred_labels[mask]

        # 计算整体精度 OA
        OA = accuracy_score(true_labels, pred_labels)

        # 计算每类精度，再取平均 AA
        num_classes = len(np.unique(true_labels))
        class_acc = []
        for c in range(num_classes):
            idx = (true_labels == c)
            if np.sum(idx) == 0:
                continue
            acc = accuracy_score(true_labels[idx], pred_labels[idx])
            class_acc.append(acc)
        AA = np.mean(class_acc)

        # 计算 Kappa
        kappa = cohen_kappa_score(true_labels, pred_labels)

        return OA, AA, kappa
    
    def save_model(self, path):
        """保存模型（兼容.pth格式）"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'pca': self.pca,
            'svm': self.svm,
            'config': {
                'n_components': self.n_components,
                'background_value': self.background_value
            }
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        model_dict = joblib.load(path)
        self.pca = model_dict['pca']
        self.svm = model_dict['svm']
        if 'config' in model_dict:
            self.n_components = model_dict['config'].get('n_components', 30)
            self.background_value = model_dict['config'].get('background_value', 0)