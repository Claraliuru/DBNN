import torch
import yaml
from torch.utils.data import DataLoader
from data_lodaer.loader import HSI
from utils.helper import load_model
from utils.train import train
from utils.evaluate import evaluate
from utils.visualize import Visualize
from datetime import datetime
import os

from models.cnn_3d import Spectral3DCNN
from models.simple_dbnn import SimpleDBNN
from models.simple_resnet import Simple_Resnet
from models.dbnn_multinn import DBNN_multinn
from models.dbnn_no_trans import DBNN_noTrans
from models.dbnn_noatt import DBNN_noatt
from models.dbnn_3x3 import DBNN_3x3
from models.dbnn import DBNN
from models.dbnn_gf import DBNN_gf

def save_model_epochs(model_epochs, filename="configs/model_epochs.yaml"):
    with open(filename, "w") as file:
        yaml.dump(model_epochs, file, default_flow_style=False)

def load_model_epochs(filename="configs/model_epochs.yaml"):
    if os.path.exists(filename):
        with open(filename, "r") as file:
            return yaml.safe_load(file)
    else:
        return {}

def get_dataset_config(dataset_name, config):
    for ds in config["datasets"]:
        if ds["name"] == dataset_name:
            return ds
    raise ValueError(f"数据集{dataset_name} 不存在于配置中")

# 模型构建函数
def build_model(name, in_channels, num_classes):
    match name:
        case "SimpleDBNN":
            return SimpleDBNN(in_channels, num_classes)
        case "Spectral3DCNN":
            return Spectral3DCNN(in_channels, num_classes)
        case "Simple_Resnet":
            return Simple_Resnet(in_channels, num_classes)
        case "DBNN_multinn":
            return DBNN_multinn(in_channels, num_classes)
        case "DBNN_noTrans":
            return DBNN_noTrans(in_channels, num_classes)
        case "DBNN_noatt":
            return DBNN_noatt(in_channels, num_classes)
        case "DBNN_3x3":
            return DBNN_3x3(in_channels, num_classes)
        case "DBNN":
            return DBNN(in_channels, num_classes)
        case "DBNN_gf":
            return DBNN_gf(in_channels, num_classes)
        case _:
            raise ValueError(f"Unknown model: {name}")

# 训练函数
def train_model(model_name, dataset_cfg, config, device):
    # 加载数据集
    train_dataset = HSI(
        data_path=dataset_cfg["data_path"],
        label_path=dataset_cfg["label_path"],
        patch_size=dataset_cfg["patch_size"],
        use_pca=config["use_pca"],
        n_components=config["n_components"],
        train=True
    )
    model = build_model(model_name, in_channels=train_dataset.num_channels, num_classes=train_dataset.num_classes)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    epoch = train(config=config, device=device, train_loader=train_loader, model=model, model_name=model_name, model_path=dataset_cfg["model_path"])

    # 保存模型时添加PCA标记
    model_suffix = "_PCA" if config["use_pca"] else ""
    model_filename = os.path.join(dataset_cfg["model_path"], model_name, f"{model_name}{model_suffix}_epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_filename)
    return epoch

# 评估函数
def evaluate_model(model_name, epoch, dataset_cfg, config, device, num_classes=None):
    use_pca = config["use_pca"]
    # 加载测试集
    test_dataset = HSI(
        data_path=dataset_cfg["data_path"],
        label_path=dataset_cfg["label_path"],
        patch_size=dataset_cfg["patch_size"],
        use_pca=use_pca,
        n_components=config["n_components"],
        train=False
    )

    model = build_model(model_name, in_channels=test_dataset.num_channels, num_classes=test_dataset.num_classes)
    # 根据是否使用PCA来修改文件名
    model_suffix = "_PCA" if config["use_pca"] else ""
    # 加载模型
    test_path = os.path.join(dataset_cfg["model_path"], model_name, f"{model_name}{model_suffix}_epoch_{epoch}.pth")
    print(test_path)
    if not os.path.isfile(test_path):
        print(f"模型文件不存在: {test_path}")
        return
    load_model(model.to(device), test_path)

    # 评估
    num_classes = test_dataset.num_classes
    # 创造数据加载器
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    OA, AA, kappa = evaluate(model, test_loader, device, num_classes)
    
    print(f"[{model_name} on {dataset_cfg['name']}] OA: {OA:.4f}, AA: {AA:.4f}, Kappa: {kappa:.4f}")
    
    return {
        "epoch": epoch,
        "OA": float(OA),
        "AA": float(AA),
        "Kappa": float(kappa),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "use_pca": use_pca
    }

# 可视化函数
def visualize_all_models(dataset_cfg, model_epochs, config, device):
    test_dataset = HSI(
        data_path=dataset_cfg["data_path"],
        label_path=dataset_cfg["label_path"],
        patch_size=config["patch_size"],
        use_pca=config["use_pca"],
        n_components=config["n_components"],
        train=False
    )

    ground_truth = test_dataset.label
    out_dir = dataset_cfg["figure_dir"]
    os.makedirs(out_dir, exist_ok=True)
    model_suffix = "_PCA" if config["use_pca"] else ""

    visualizer = Visualize(config=config, device=device)
    model_maps = {}

    # 仅可视化每个模型最后一次结果
    sorted_model_epochs = {
        model: entries[-1]["epoch"] for model, entries in model_epochs.items() if entries
    }

    for model_name, epoch in sorted_model_epochs.items():
        model = build_model(model_name, test_dataset.num_channels, 
                            test_dataset.num_classes)
        model_path = os.path.join(dataset_cfg["model_path"], model_name, f"{model_name}{model_suffix}_epoch_{epoch}.pth")

        # 检查模型文件是否存在
        if not os.path.isfile(model_path):
            print(f"模型文件不存在: {model_path}，跳过该模型的可视化")
            continue  # 跳过没有模型文件的模型

        load_model(model, model_path)
        model.to(device)
        model.eval()
        cls_map = visualizer.generate_classification_map(model, test_dataset)

        model_maps[model_name] = cls_map

    # 可视化所有模型的对比图
    if model_maps:  # 确保有模型进行比较
        visualizer.visualize_comparison(ground_truth, model_maps, 
            save_path=os.path.join(out_dir, f'{dataset_cfg["name"]}.png'))
    else:
        print("没有可用的模型进行可视化。")


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device is {device}")

    retrain_all = input("是否重新训练所有模型？(y/N): ").strip().lower() == "y"

    model_epochs_file = config.get("model_epochs_file", "configs/model_epochs.yaml")
    model_epochs = load_model_epochs(model_epochs_file) or {}

    for dataset in config["datasets"]:
        dataset_name = dataset["name"]

        if dataset_name not in model_epochs:
            model_epochs[dataset_name] = {}

        dataset_config = get_dataset_config(dataset_name, config)
        print(f"\n=== 当前数据集：{dataset_name} ===")

        for model_info in config["models"]:
            model_name = model_info["name"]
            print(f"\n--- 当前模型：{model_name} ---")

            merged_config = {**config, **dataset_config}

            # 判断是否需要训练
            history = model_epochs[dataset_name].get(model_name, [])
            if retrain_all or not history:
                epoch = train_model(model_name, merged_config, config, device)
            else:
                epoch = history[-1]["epoch"]
                print(f"使用已有模型 epoch {epoch}")

            eval_result = evaluate_model(model_name, epoch, merged_config, config, device)
            if eval_result:
                model_epochs[dataset_name].setdefault(model_name, []).append(eval_result)

        # 保存训练记录
        save_model_epochs(model_epochs, model_epochs_file)

        # 可视化所有模型
        visualize_all_models(dataset_config, model_epochs[dataset_name], config, device)
