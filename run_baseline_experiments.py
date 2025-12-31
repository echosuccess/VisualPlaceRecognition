#!/usr/bin/env python3
"""
Baseline实验运行脚本
用于系统化地运行VPR baseline实验 旧版本，目前不用了
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime

# 实验配置
VPR_METHODS = [
    {"method": "cosplace", "backbone": "ResNet18", "dim": 512, "image_size": "512 512"},
    {"method": "netvlad", "backbone": "VGG16", "dim": 4096, "image_size": "512 512"},
    {"method": "mixvpr", "backbone": "ResNet50", "dim": 4096, "image_size": "320 320"},
    {"method": "megaloc", "backbone": "Dinov2", "dim": 8448, "image_size": "224 224"},
]

DISTANCE_METRICS = ["l2", "dot_product"]

DATASETS = [
    {"name": "SF-XS-test", "database": "data/sf_xs/test/database", "queries": "data/sf_xs/test/queries"},
    {"name": "Tokyo-XS", "database": "data/tokyo_xs/test/database", "queries": "data/tokyo_xs/test/queries"},
    {"name": "SVOX-sun", "database": "data/svox/images/test/gallery", "queries": "data/svox/images/test/queries"},
    {"name": "SVOX-night", "database": "data/svox/images/test/gallery", "queries": "data/svox/images/test/queries_night"},
]

# 实验参数
NUM_WORKERS = 8
BATCH_SIZE = 32
NUM_PREDS = 20
MAX_QUERIES_VIZ = 3  # Only save visualizations for first 3 queries (for Colab)
RECALL_VALUES = [1, 5, 10, 20]


def run_experiment(vpr_method, distance_metric, dataset, log_base_dir="logs/baseline"):
    """运行单个实验"""
    
    # 创建日志目录名
    log_dir = f"{log_base_dir}/{vpr_method['method']}_{distance_metric}_{dataset['name']}"
    
    # 构建命令
    cmd = [
        "python", "VPR-methods-evaluation/main.py",
        "--num_workers", str(NUM_WORKERS),
        "--batch_size", str(BATCH_SIZE),
        "--log_dir", log_dir,
        "--method", vpr_method['method'],
        "--backbone", vpr_method['backbone'],
        "--descriptors_dimension", str(vpr_method['dim']),
        "--image_size", *vpr_method['image_size'].split(),
        "--database_folder", dataset['database'],
        "--queries_folder", dataset['queries'],
        "--num_preds_to_save", str(NUM_PREDS),
        "--max_queries_to_save", str(MAX_QUERIES_VIZ),
        "--recall_values", *[str(r) for r in RECALL_VALUES],
        "--distance_metric", distance_metric,
        "--save_for_uncertainty",
    ]
    
    print(f"\n{'='*80}")
    print(f"运行实验:")
    print(f"  方法: {vpr_method['method']}")
    print(f"  距离度量: {distance_metric}")
    print(f"  数据集: {dataset['name']}")
    print(f"  日志目录: {log_dir}")
    print(f"{'='*80}\n")
    
    try:
        # 运行命令
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return {"status": "success", "log_dir": log_dir}
    except subprocess.CalledProcessError as e:
        print(f"实验失败: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    """主函数：运行所有baseline实验"""
    
    print("\n" + "="*80)
    print("开始运行Baseline实验")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    results = []
    total_experiments = len(VPR_METHODS) * len(DISTANCE_METRICS) * len(DATASETS)
    current_exp = 0
    
    for vpr_method in VPR_METHODS:
        for distance_metric in DISTANCE_METRICS:
            for dataset in DATASETS:
                current_exp += 1
                print(f"\n进度: {current_exp}/{total_experiments}")
                
                result = run_experiment(vpr_method, distance_metric, dataset)
                results.append({
                    "method": vpr_method['method'],
                    "distance_metric": distance_metric,
                    "dataset": dataset['name'],
                    **result
                })
    
    # 保存实验结果摘要
    summary_file = Path("logs/baseline_experiments_summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_experiments": total_experiments,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("所有实验完成!")
    print(f"结果摘要已保存到: {summary_file}")
    print("="*80 + "\n")
    
    # 打印失败的实验
    failed = [r for r in results if r['status'] == 'failed']
    if failed:
        print("\n失败的实验:")
        for f in failed:
            print(f"  - {f['method']} / {f['distance_metric']} / {f['dataset']}")
    else:
        print("\n✅ 所有实验成功完成!")


if __name__ == "__main__":
    # 提示用户
    print("\n" + "="*80)
    print("Baseline实验运行脚本")
    print("="*80)
    print(f"\n将运行 {len(VPR_METHODS)} 个VPR方法 × {len(DISTANCE_METRICS)} 个距离度量 × {len(DATASETS)} 个数据集")
    print(f"总共: {len(VPR_METHODS) * len(DISTANCE_METRICS) * len(DATASETS)} 个实验\n")
    
    response = input("是否继续? (y/n): ")
    if response.lower() == 'y':
        main()
    else:
        print("已取消")

