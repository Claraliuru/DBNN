import torch
import yaml
from torch.utils.data import DataLoader
from data_lodaer.loader import HSI
from models.dbnn import DBNN
from models.dbnn_gf import DBNN_gf
from models.svm_pca import SVMPCA
from utils.helper import save_model, load_model
from utils.train import train
from utils.evaluate import evaluate
from utils.visualize import Visualize
from matplotlib import pyplot as plt
import os

def get_dataset_config(dataset_name, config):
    for ds in config["datasets"]:
        if ds["name"] == dataset_name:
            return ds
    raise ValueError(f"数据集{dataset_name} 不存在于配置中")

# 模型构建函数
def build_model(name, in_channels, num_classes):
    match name:
        case "DBNN":
            return DBNN(in_channels, num_classes)
        case "DBNN_gf":
            return DBNN_gf(in_channels, num_classes)
        case "SVMPCA":
            return SVMPCA(n_components=30)
        case _:
            raise ValueError(f"Unknown model: {name}")

# 训练函数
def train_model(model_name, dataset_cfg):
    # 加载数据集
    train_dataset = HSI(
        data_path=dataset_cfg["data_path"],
        label_path=dataset_cfg["label_path"],
        patch_size=dataset_cfg["patch_size"],
        train=True
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    model = build_model(model_name, in_channels=train_dataset.num_channels, num_classes=train_dataset.num_classes)
    epoch = train(config=config, device=device, train_loader=train_loader, model=model, model_name=model_name, model_path=dataset_cfg["model_path"])
    return epoch

# 评估函数
def evaluate_model(model_name, epoch, dataset_cfg, num_classes=0):
    # 加载测试集
    test_dataset = HSI(
        data_path=dataset_cfg["data_path"],
        label_path=dataset_cfg["label_path"],
        patch_size=dataset_cfg["patch_size"],
        train=False
    )
    in_channels = test_dataset.num_channels
    num_classes = test_dataset.num_classes
    print(f"测试数据：{in_channels}, {num_classes}")

    # 创造数据加载器
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = build_model(model_name, in_channels=in_channels, num_classes=num_classes)
    test_path = os.path.join(dataset_cfg["model_path"], model_name, f"{model_name}_epoch_{epoch}.pth")
    if not os.path.isfile(test_path):
        print(f"模型文件不存在: {test_path}")
        return
    load_model(model, test_path)

    # 评估模型
    OA, AA, kappa = evaluate(model, test_loader, device, num_classes)
    print(f"[{model_name} on {dataset_cfg['name']}] OA: {OA:.4f}, AA: {AA:.4f}, Kappa: {kappa:.4f}")

# 可视化函数
def visualize_map(model_name, epoch, dataset_cfg):
    test_dataset = HSI(
        data_path=dataset_cfg["data_path"],
        label_path=dataset_cfg["label_path"],
        patch_size=dataset_cfg["patch_size"],
        train=False
    )
    ground_truth = test_dataset.label
    model = build_model(model_name, test_dataset.num_channels, test_dataset.num_classes)
    model_path = os.path.join(dataset_cfg["model_path"], model_name, f"{model_name}_epoch_{epoch}.pth")
    if not os.path.isfile(model_path):
        print(f"可视化模型文件不存在：{model_path}")
        return
    load_model(model, model_path)
    model.to(device)
    model.eval()

    visualizer = Visualize(config=config, device=device)
    cls_map = visualizer.generate_classification_map(model, test_dataset)
    out_dir = dataset_cfg["figure_dir"]
    os.makedirs(out_dir, exist_ok=True)
    visualizer.visualize_comparison(ground_truth, {model_name:cls_map})
    plt.savefig(os.path.join(out_dir, f'{dataset_cfg["name"]}.png'))
if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device is {device}")

    for dataset in config["datasets"]:
        dataset_name = dataset["name"]
        dataset_config = get_dataset_config(dataset_name, config)
        print(f"\n=== 当前数据集：{dataset_name} ===")

        for model_info in config["models"]:
            model_name = model_info["name"]
            print(f"\n--- 当前模型：{model_name} ---")

            merged_config = {**config, **dataset_config}
            epoch = train_model(model_name, merged_config)
            evaluate_model(model_name, epoch, merged_config)
            visualize_map(model_name, epoch, merged_config)