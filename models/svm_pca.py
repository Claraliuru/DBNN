import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score

class SVMPCA:
    def __init__(self, n_components=30):
        """
        初始化 SVM+PCA 模型
        :param n_components: PCA 降维后的维度
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.svm = SVC(kernel='rbf', C=1.0, gamma='scale')

    def train(self, train_data, train_labels):
        """
        训练 SVM+PCA 模型
        :param train_data: 训练数据 (N, H, W, C)
        :param train_labels: 训练标签 (N,)
        """
        # 将数据展平为 (N, H*W*C)
        train_data_flat = train_data.reshape(train_data.shape[0], -1)

        # PCA 降维
        self.pca.fit(train_data_flat)
        train_data_pca = self.pca.transform(train_data_flat)

        # 训练 SVM 模型
        self.svm.fit(train_data_pca, train_labels)

    def predict(self, test_data):
        """
        预测测试数据
        :param test_data: 测试数据 (M, H, W, C)
        :return: 预测标签 (M,)
        """
        # 将数据展平为 (M, H*W*C)
        test_data_flat = test_data.reshape(test_data.shape[0], -1)

        # PCA 降维
        test_data_pca = self.pca.transform(test_data_flat)

        # SVM 预测
        return self.svm.predict(test_data_pca)

    def evaluate(self, test_labels, test_pred):
        """
        评估模型性能
        :param test_labels: 测试集真实标签
        :param test_pred: 测试集预测标签
        :return: OA, AA, Kappa
        """
        # 计算 OA (Overall Accuracy)
        oa = accuracy_score(test_labels, test_pred)

        # 计算 AA (Average Accuracy)
        conf_matrix = confusion_matrix(test_labels, test_pred)
        aa = np.mean(conf_matrix.diagonal() / conf_matrix.sum(axis=1))

        # 计算 Kappa 系数
        kappa = cohen_kappa_score(test_labels, test_pred)

        return oa, aa, kappa

    def generate_classification_map(self, dataset):
        """
        生成分类结果图
        :param dataset: 数据集对象
        :return: 分类结果图 (H, W)
        """
        h, w, _ = dataset.data.shape
        classification_map = np.zeros((h, w), dtype=np.int64)

        for x in range(h):
            for y in range(w):
                if dataset.labels[x, y] == 0:
                    classification_map[x, y] = 0
                else:
                    patch = dataset.extract_patch(x, y)
                    patch_flat = patch.reshape(1, -1)  # 展平为 (1, H*W*C)
                    patch_pca = self.pca.transform(patch_flat)  # PCA 降维
                    pred = self.svm.predict(patch_pca)  # SVM 预测
                    classification_map[x, y] = pred[0]

        return classification_map