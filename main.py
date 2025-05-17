"""主函数"""

# 导入库
import torch
import yaml
from torch.utils.data import DataLoader
from data_lodaer.loader import HSI
from utils.helper import load_model
from utils.train import train
from utils.evaluate import evaluate
from utils.visualize import Visualize
import os

# ===主模型 ===
from models.dbnn import DBNN

# === 对比试验 ===
from models.cnn_2d import Spectral2DCNN
from models.cnn_3d import Spectral3DCNN
from models.ssrn import SSRN

# === 消融实验 ===
from models.dbnn import NOPCA
from models.noguide import NOGUIDE
from models.nomulti import NOMULTI
from models.nores import NORES
from models.noat import NOAT
from models.notr import NOTR

def save_model_epochs(model_epochs, filename):
    """保存训练轮次信息到YAML文件"""
    os.makedirs(os.path.dirname(filename), exist_ok=True) # 创建文件夹
    with open(filename, "w", encoding="utf-8") as file:
        yaml.dump(model_epochs, file, default_flow_style=False)

def load_model_epochs(filename):
    """加载训练轮次信息，如果文件不存在则返回字典"""
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) # 加载已存在的训练轮次记录
    else:
        return {}

def get_dataset_config(dataset_name, config):
    """获取指定数据集配置"""
    for ds in config["datasets"]:
        if ds["name"] == dataset_name:
            return ds # 找到匹配的数据集配置
    raise ValueError(f"数据集{dataset_name} 不存在于配置中")

# 模型构建函数
def build_model(name, in_channels, num_classes, p):
    match name:
        # 主模型
        case "DBNN":
            return DBNN(in_channels, num_classes, p)
        # 对比试验
        case "Spectral2DCNN":
            return Spectral2DCNN(in_channels, num_classes, p)
        case "Spectral3DCNN":
            return Spectral3DCNN(in_channels, num_classes, p)
        case "SSRN":
            return SSRN(in_channels, num_classes, p)
        # 消融实验
        case "NOPCA":
            return NOPCA(in_channels, num_classes, p)
        case "NOGUIDE":
            return NOGUIDE(in_channels, num_classes, p)
        case "NOMULTI":
            return NOMULTI(in_channels, num_classes, p)
        case "NORES":
            return NORES(in_channels, num_classes, p)
        case "NOAT":
            return NOAT(in_channels, num_classes, p)
        case "NOTR":
            return NOTR(in_channels, num_classes, p)
        
        case _:
            raise ValueError(f"Unknown model: {name}")

# 训练函数
def train_model(model_name, dataset_cfg, config, device):
    # 加载数据集
    train_dataset = HSI(
        data_path=dataset_cfg["data_path"],
        label_path=dataset_cfg["label_path"],
        patch_size=dataset_cfg["patch_size"],
        train_split=dataset_cfg["train_split"],
        n_components=dataset_cfg["n_components"],
        use_pca=config["use_pca"],
        train=True
    )

    # 构建模型
    model = build_model(model_name, in_channels=train_dataset.num_channels, num_classes=train_dataset.num_classes, p=dataset_cfg["p"])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=dataset_cfg["batch_size"], shuffle=True)
    # 训练并返回最终epoch
    epoch = train(config=config, datasets_cfg=dataset_cfg, device=device, train_loader=train_loader, model=model, model_name=model_name, model_path=dataset_cfg["model_path"])

    # 保存模型时添加PCA标记
    model_suffix = "_PCA" if config["use_pca"] else ""
    model_filename = os.path.join(dataset_cfg["model_path"], model_name, f"{model_name}{model_suffix}_epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_filename) # 保存模型参数
    return epoch

# 评估函数
def evaluate_model(model_name, epoch, dataset_cfg, config, device):
    # 加载测试集
    test_dataset = HSI(
        data_path=dataset_cfg["data_path"],
        label_path=dataset_cfg["label_path"],
        patch_size=dataset_cfg["patch_size"],
        train_split=dataset_cfg["train_split"],
        n_components=dataset_cfg["n_components"],
        use_pca=config["use_pca"],
        train=False
    )

    # 构建模型
    model = build_model(model_name, in_channels=test_dataset.num_channels, num_classes=test_dataset.num_classes, p=dataset_cfg["p"])
    # 根据是否使用PCA来修改文件名
    model_suffix = "_PCA" if config["use_pca"] else ""
    # 加载模型
    test_path = os.path.join(dataset_cfg["model_path"], model_name, f"{model_name}{model_suffix}_epoch_{epoch}.pth")
    if not os.path.isfile(test_path):
        print(f"模型文件不存在: {test_path}")
        return
    load_model(model.to(device), test_path)

    # 创造数据加载器
    test_loader = DataLoader(test_dataset, batch_size=dataset_cfg["batch_size"], shuffle=False)
    # 评估
    OA, AA, kappa = evaluate(model, test_loader, device, test_dataset.num_classes)
    
    print(f"[{model_name} on {dataset_cfg['name']}]\n OA: {OA:.4f}, AA: {AA:.4f}, Kappa: {kappa:.4f}")
    
    return {
        "train_split": dataset_cfg["train_split"],
        "weight_decay": dataset_cfg["weight_decay"],
        "p": dataset_cfg["p"],
        "OA": float(OA),
        "AA": float(AA),
        "Kappa": float(kappa),
        # "use_pca": use_pca
    }

def visualize_all_models(dataset_cfg, model_epochs, config, device):
    train_split = dataset_cfg["train_split"]
    weight_decay = dataset_cfg["weight_decay"]
    p = dataset_cfg['p']
    test_dataset = HSI(
        data_path=dataset_cfg["data_path"],
        label_path=dataset_cfg["label_path"],
        patch_size=dataset_cfg["patch_size"],
        train_split=train_split,
        n_components=dataset_cfg["n_components"],
        use_pca=config["use_pca"],
        train=False
    )

    ground_truth = test_dataset.label # 获取原始标签
    out_dir = dataset_cfg["figure_dir"] # 图像保存路径
    os.makedirs(out_dir, exist_ok=True)
    model_suffix = "_PCA" if config["use_pca"] else ""
    target_epoch = config["max_epochs"]
    visualizer = Visualize(config=config, device=device)

    # 遍历每个模型并生成分类图
    for model_name in model_epochs.keys():  # 直接遍历字典键
        model_path = os.path.join(
            dataset_cfg["model_path"], 
            model_name, 
            f"{model_name}{model_suffix}_epoch_{target_epoch}.pth"  # 使用target_epoch
        )

        if not os.path.isfile(model_path):
            print(f"模型文件不存在: {model_path}，跳过该模型的可视化")
            continue

        # 构建并加载模型
        model = build_model(
            model_name, 
            test_dataset.num_channels, 
            test_dataset.num_classes, 
            p=p
        )
        load_model(model, model_path)
        model.to(device).eval()

        # 生成并保存分类图
        cls_map = visualizer.generate_classification_map(model, test_dataset)
        if cls_map is not None:
            save_name = f'{model_name}_ts{train_split}_wd{weight_decay}_p{p}_epoch_{target_epoch}'
            visualizer.visualize_comparison(
                ground_truth, 
                cls_map, 
                save_name=save_name,
                save_dir=out_dir
            )
            print(f"已生成 {model_name} 的分类图")
        else:
            print(f"模型 {model_name} 的分类图生成失败")

if __name__ == "__main__":
    # 读取配置文件
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device is {device}")

    # 用户决定是否重新训练
    retrain_all = input("是否重新训练所有模型？(y/N): ").strip().lower() == "y"
    # retrain_all = "n"

    model_epochs_file = config["model_file"]
    model_epochs = load_model_epochs(model_epochs_file) or {}

    # 遍历所有数据集
    for dataset in config["datasets"]:
        dataset_name = dataset["name"]

        if dataset_name not in model_epochs:
            model_epochs[dataset_name] = {}

        dataset_config = get_dataset_config(dataset_name, config)
        print(f"\n=== 当前数据集：{dataset_name} ===")

        # 遍历所有模型
        for model_info in config["models"]:
            model_name = model_info["name"]
            print(f"\n--- 当前模型：{model_name} ---")

            # 判断是否需要训练
            history = model_epochs[dataset_name].get(model_name, [])
            if retrain_all or not history:
                epoch = train_model(model_name, dataset_config, config, device)
            else:
                epoch=config["max_epochs"]

            # 评估模型
            eval_result = evaluate_model(model_name, epoch, dataset_config, config, device)
            if eval_result:
                model_epochs[dataset_name].setdefault(model_name, []).append(eval_result)

        # 保存训练记录
        save_model_epochs(model_epochs, model_epochs_file)

        # 可视化所有模型
        # visualize_all_models(dataset_config, model_epochs[dataset_name], config, device)