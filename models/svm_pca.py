# models/svm_pca.py
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SVMPCA:
    def __init__(self, n_components=30, C=1.0, kernel='rbf', gamma='scale', random_state=None):
        """
        PCA + SVM 高光谱分类模型
        
        参数:
            n_components: PCA降维后的维度
            C: SVM的惩罚参数
            kernel: SVM核函数类型 ('linear', 'rbf', 'poly'等)
            gamma: 核函数系数 ('scale', 'auto'或数值)
            random_state: 随机种子
        """
        self.n_components = n_components
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state
        
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.scaler = StandardScaler()
        self.svm = SVC(C=C, kernel=kernel, gamma=gamma, random_state=random_state)
    
    def train(self, data, labels):
        """
        训练PCA+SVM模型
        
        参数:
            data: 高光谱数据 (n_samples, n_features)
            labels: 对应标签 (n_samples,)
        """
        # 将数据展平为2D (如果是3D空间数据)
        if data.ndim > 2:
            original_shape = data.shape
            data = data.reshape(-1, original_shape[-1])
            labels = labels.flatten()
        
        # 移除背景类别(0)的样本
        mask = labels != 0
        data = data[mask]
        labels = labels[mask]
        
        # 数据标准化
        data = self.scaler.fit_transform(data)
        
        # PCA降维
        pca_features = self.pca.fit_transform(data)
        
        # 训练SVM
        self.svm.fit(pca_features, labels)
    
    def predict(self, data):
        """
        预测分类结果，并保持背景区域为0
        
        参数:
            data: 高光谱数据 (可以是2D或3D)
        
        返回:
            预测的类别标签 (与输入空间形状相同)，背景保持为0
        """
        original_shape = data.shape
        is_3d = data.ndim > 2
        
        # 保存原始背景掩膜
        if is_3d:
            # 对于3D数据，计算第一个波段的0值区域作为背景
            background_mask = (data[..., 0] == 0)  # 假设背景在第一个波段为0
            data_2d = data.reshape(-1, original_shape[-1])
        else:
            background_mask = (data[..., 0] == 0)  # 对于2D数据
            data_2d = data.copy()
        
        # 标准化 + PCA + SVM预测
        data_2d = self.scaler.transform(data_2d)
        pca_features = self.pca.transform(data_2d)
        pred = self.svm.predict(pca_features)
        
        # 恢复形状
        if is_3d:
            pred = pred.reshape(original_shape[:-1])
        else:
            pred = pred.reshape(original_shape[:1])
        
        # 将背景区域强制设为0
        pred[background_mask] = 0
        
        return pred
    
    def evaluate(self, true_labels, pred_labels):
        """
        评估模型性能
        
        参数:
            true_labels: 真实标签
            pred_labels: 预测标签
            
        返回:
            OA (总体精度), AA (平均精度), kappa (kappa系数)
        """
        # 展平并移除背景类别
        true_labels = true_labels.flatten()
        pred_labels = pred_labels.flatten()
        mask = true_labels != 0
        
        true_labels = true_labels[mask]
        pred_labels = pred_labels[mask]
        
        # 计算OA
        OA = accuracy_score(true_labels, pred_labels)
        
        # 计算AA (每个类别的平均精度)
        unique_classes = np.unique(true_labels)
        AA = 0
        for cls in unique_classes:
            cls_mask = true_labels == cls
            if np.sum(cls_mask) > 0:
                AA += accuracy_score(true_labels[cls_mask], pred_labels[cls_mask])
        AA /= len(unique_classes)
        
        # 计算kappa系数
        kappa = cohen_kappa_score(true_labels, pred_labels)
        
        return OA, AA, kappa
    
    def save_model(self, path):
        """保存模型到文件"""
        joblib.dump({
            'pca': self.pca,
            'scaler': self.scaler,
            'svm': self.svm
        }, path)
    
    def load_model(self, path):
        """从文件加载模型"""
        model_dict = joblib.load(path)
        self.pca = model_dict['pca']
        self.scaler = model_dict['scaler']
        self.svm = model_dict['svm']