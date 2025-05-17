import yaml
import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt

def load_yaml_data(file_path):
    """Load data from YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def calculate_average_metrics(data):
    """Calculate average, max, and min metrics grouped by dataset, model, and parameters."""
    results = defaultdict(list)
    
    for dataset_name, models in data.items():
        for model_name, experiments in models.items():
            for exp in experiments:
                key = (dataset_name, model_name, exp['train_split'], exp['weight_decay'])
                results[key].append({
                    'OA': exp['OA'],
                    'AA': exp['AA'],
                    'Kappa': exp['Kappa']
                })
    
    # Calculate averages and min/max
    averages = []
    for key, metrics_list in results.items():
        dataset, model, train_split, weight_decay = key
        count = len(metrics_list)
        
        oa_list = [m['OA'] for m in metrics_list]
        aa_list = [m['AA'] for m in metrics_list]
        kappa_list = [m['Kappa'] for m in metrics_list]
        
        averages.append({
            'Dataset': dataset,
            'Model': model,
            # 'p': p,
            'train_split': train_split,
            # 'weight_decay': weight_decay,
            'avg_OA': sum(oa_list) / count,
            # 'max_OA': max(oa_list),
            # 'min_OA': min(oa_list),
            'avg_AA': sum(aa_list) / count,
            # 'max_AA': max(aa_list),
            # 'min_AA': min(aa_list),
            'avg_Kappa': sum(kappa_list) / count,
            # 'max_Kappa': max(kappa_list),
            # 'min_Kappa': min(kappa_list),
            'num_runs': count
        })
    
    return averages

def save_to_csv(data, config):
    """Save averaged results to CSV files grouped by dataset."""
    os.makedirs(config["file_dir"], exist_ok=True)
    
    # Group by dataset first
    datasets = defaultdict(list)
    for row in data:
        datasets[row['Dataset']].append(row)
    
    # Save separate CSV for each dataset
    for dataset_name, rows in datasets.items():
        filename = os.path.join(config["file_dir"], f'{dataset_name}_{config["file_name"]}.csv')
        fieldnames = [
            'Dataset', 'Model', 'train_split', #'p', 'weight_decay',
            'avg_OA',# 'max_OA', 'min_OA',
            'avg_AA', #'max_AA', 'min_AA',
            'avg_Kappa',# 'max_Kappa', 'min_Kappa',
            'num_runs'
        ]

        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Saved averages for {dataset_name} to {filename}")
import matplotlib.pyplot as plt

def plot_dnn_train_split_effect(averages, model_name, save_dir):
    """绘制 DBNN 模型在不同 train_split 下 OA 的变化图"""
    os.makedirs(save_dir, exist_ok=True)

    # 筛选 DBNN 的数据
    filtered = [row for row in averages if row["Model"] == model_name]

    # 按 train_split 分组求 OA 均值
    results = defaultdict(list)
    for row in filtered:
        split = row["train_split"]
        results[split].append(row["avg_OA"])

    # 排序并平均
    splits = sorted(results.keys())
    avg_oa = [sum(results[s]) / len(results[s]) for s in splits]

    # 绘图
    plt.figure()
    plt.plot(splits, avg_oa, marker='o', linestyle='-', color='b', label='Average OA')
    plt.xlabel("Train Split")
    plt.ylabel("Average OA")
    plt.title(f"{model_name} - OA vs Train Split")
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(save_dir, f"{model_name}_oa_vs_train_split.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()
    print(f"图像已保存到 {plot_path}")

def main():
    with open("configs/config.yaml", "r",encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # Load data
    data = load_yaml_data(config["model_file"])
    
    # Process data
    averages = calculate_average_metrics(data)
    
    # Save results
    save_to_csv(averages,config)
    if config["model_file"] == "outputs/ablation.yaml":
    # Plot OA vs train_split for DBNN
        plot_dnn_train_split_effect(averages, model_name="DBNN", save_dir=config["file_dir"])

if __name__ == '__main__':
    main()