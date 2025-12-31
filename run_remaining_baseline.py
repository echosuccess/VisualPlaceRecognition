#!/usr/bin/env python3
"""
运行剩余的Baseline实验
会检查哪些已完成，只运行未完成的实验
"""

import subprocess
from pathlib import Path
import json
from datetime import datetime

# 定义所有32个实验
ALL_EXPERIMENTS = [
    # CosPlace组 (8个)
    {"id": 1, "method": "cosplace", "backbone": "ResNet18", "dim": 512, "size": "512 512", "distance": "l2", "dataset": "SF-XS", "db": "data/sf_xs/test/database", "q": "data/sf_xs/test/queries"},
    {"id": 2, "method": "cosplace", "backbone": "ResNet18", "dim": 512, "size": "512 512", "distance": "l2", "dataset": "Tokyo-XS", "db": "data/tokyo_xs/test/database", "q": "data/tokyo_xs/test/queries"},
    {"id": 3, "method": "cosplace", "backbone": "ResNet18", "dim": 512, "size": "512 512", "distance": "l2", "dataset": "SVOX-sun", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries"},
    {"id": 4, "method": "cosplace", "backbone": "ResNet18", "dim": 512, "size": "512 512", "distance": "l2", "dataset": "SVOX-night", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries_night"},
    {"id": 5, "method": "cosplace", "backbone": "ResNet18", "dim": 512, "size": "512 512", "distance": "dot_product", "dataset": "SF-XS", "db": "data/sf_xs/test/database", "q": "data/sf_xs/test/queries"},
    {"id": 6, "method": "cosplace", "backbone": "ResNet18", "dim": 512, "size": "512 512", "distance": "dot_product", "dataset": "Tokyo-XS", "db": "data/tokyo_xs/test/database", "q": "data/tokyo_xs/test/queries"},
    {"id": 7, "method": "cosplace", "backbone": "ResNet18", "dim": 512, "size": "512 512", "distance": "dot_product", "dataset": "SVOX-sun", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries"},
    {"id": 8, "method": "cosplace", "backbone": "ResNet18", "dim": 512, "size": "512 512", "distance": "dot_product", "dataset": "SVOX-night", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries_night"},
    
    # NetVLAD组 (8个)
    {"id": 9, "method": "netvlad", "backbone": "VGG16", "dim": 4096, "size": "512 512", "distance": "l2", "dataset": "SF-XS", "db": "data/sf_xs/test/database", "q": "data/sf_xs/test/queries"},
    {"id": 10, "method": "netvlad", "backbone": "VGG16", "dim": 4096, "size": "512 512", "distance": "l2", "dataset": "Tokyo-XS", "db": "data/tokyo_xs/test/database", "q": "data/tokyo_xs/test/queries"},
    {"id": 11, "method": "netvlad", "backbone": "VGG16", "dim": 4096, "size": "512 512", "distance": "l2", "dataset": "SVOX-sun", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries"},
    {"id": 12, "method": "netvlad", "backbone": "VGG16", "dim": 4096, "size": "512 512", "distance": "l2", "dataset": "SVOX-night", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries_night"},
    {"id": 13, "method": "netvlad", "backbone": "VGG16", "dim": 4096, "size": "512 512", "distance": "dot_product", "dataset": "SF-XS", "db": "data/sf_xs/test/database", "q": "data/sf_xs/test/queries"},
    {"id": 14, "method": "netvlad", "backbone": "VGG16", "dim": 4096, "size": "512 512", "distance": "dot_product", "dataset": "Tokyo-XS", "db": "data/tokyo_xs/test/database", "q": "data/tokyo_xs/test/queries"},
    {"id": 15, "method": "netvlad", "backbone": "VGG16", "dim": 4096, "size": "512 512", "distance": "dot_product", "dataset": "SVOX-sun", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries"},
    {"id": 16, "method": "netvlad", "backbone": "VGG16", "dim": 4096, "size": "512 512", "distance": "dot_product", "dataset": "SVOX-night", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries_night"},
    
    # MixVPR组 (8个)
    {"id": 17, "method": "mixvpr", "backbone": "ResNet50", "dim": 4096, "size": "320 320", "distance": "l2", "dataset": "SF-XS", "db": "data/sf_xs/test/database", "q": "data/sf_xs/test/queries"},
    {"id": 18, "method": "mixvpr", "backbone": "ResNet50", "dim": 4096, "size": "320 320", "distance": "l2", "dataset": "Tokyo-XS", "db": "data/tokyo_xs/test/database", "q": "data/tokyo_xs/test/queries"},
    {"id": 19, "method": "mixvpr", "backbone": "ResNet50", "dim": 4096, "size": "320 320", "distance": "l2", "dataset": "SVOX-sun", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries"},
    {"id": 20, "method": "mixvpr", "backbone": "ResNet50", "dim": 4096, "size": "320 320", "distance": "l2", "dataset": "SVOX-night", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries_night"},
    {"id": 21, "method": "mixvpr", "backbone": "ResNet50", "dim": 4096, "size": "320 320", "distance": "dot_product", "dataset": "SF-XS", "db": "data/sf_xs/test/database", "q": "data/sf_xs/test/queries"},
    {"id": 22, "method": "mixvpr", "backbone": "ResNet50", "dim": 4096, "size": "320 320", "distance": "dot_product", "dataset": "Tokyo-XS", "db": "data/tokyo_xs/test/database", "q": "data/tokyo_xs/test/queries"},
    {"id": 23, "method": "mixvpr", "backbone": "ResNet50", "dim": 4096, "size": "320 320", "distance": "dot_product", "dataset": "SVOX-sun", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries"},
    {"id": 24, "method": "mixvpr", "backbone": "ResNet50", "dim": 4096, "size": "320 320", "distance": "dot_product", "dataset": "SVOX-night", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries_night"},
    
    # MegaLoc组 (8个)
    {"id": 25, "method": "megaloc", "backbone": "Dinov2", "dim": 8448, "size": "224 224", "distance": "l2", "dataset": "SF-XS", "db": "data/sf_xs/test/database", "q": "data/sf_xs/test/queries"},
    {"id": 26, "method": "megaloc", "backbone": "Dinov2", "dim": 8448, "size": "224 224", "distance": "l2", "dataset": "Tokyo-XS", "db": "data/tokyo_xs/test/database", "q": "data/tokyo_xs/test/queries"},
    {"id": 27, "method": "megaloc", "backbone": "Dinov2", "dim": 8448, "size": "224 224", "distance": "l2", "dataset": "SVOX-sun", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries"},
    {"id": 28, "method": "megaloc", "backbone": "Dinov2", "dim": 8448, "size": "224 224", "distance": "l2", "dataset": "SVOX-night", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries_night"},
    {"id": 29, "method": "megaloc", "backbone": "Dinov2", "dim": 8448, "size": "224 224", "distance": "dot_product", "dataset": "SF-XS", "db": "data/sf_xs/test/database", "q": "data/sf_xs/test/queries"},
    {"id": 30, "method": "megaloc", "backbone": "Dinov2", "dim": 8448, "size": "224 224", "distance": "dot_product", "dataset": "Tokyo-XS", "db": "data/tokyo_xs/test/database", "q": "data/tokyo_xs/test/queries"},
    {"id": 31, "method": "megaloc", "backbone": "Dinov2", "dim": 8448, "size": "224 224", "distance": "dot_product", "dataset": "SVOX-sun", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries"},
    {"id": 32, "method": "megaloc", "backbone": "Dinov2", "dim": 8448, "size": "224 224", "distance": "dot_product", "dataset": "SVOX-night", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries_night"},
]


def check_experiment_completed(exp):
    """检查实验是否已完成"""
    dataset_name = exp['dataset'].replace('-', '_').lower()
    
    # 检查两个可能的日志目录
    log_dirs = [
        Path(f"logs/baseline/{exp['method']}_{exp['distance']}_{dataset_name}_test"),
        Path(f"logs/logs/baseline/{exp['method']}_{exp['distance']}_{exp['dataset']}")
    ]
    
    for log_dir in log_dirs:
        if log_dir.exists():
            for subdir in log_dir.iterdir():
                if subdir.is_dir():
                    info_log = subdir / "info.log"
                    if info_log.exists():
                        try:
                            content = info_log.read_text(encoding='utf-8', errors='ignore')
                            if "R@1:" in content:
                                return True
                        except:
                            pass
    return False


def run_experiment(exp):
    """运行单个实验"""
    dataset_name = exp['dataset'].replace('-', '_').lower()
    log_dir = f"baseline/{exp['method']}_{exp['distance']}_{dataset_name}_test"
    
    cmd = [
        "python", "VPR-methods-evaluation/main.py",
        "--method", exp['method'],
        "--backbone", exp['backbone'],
        "--descriptors_dimension", str(exp['dim']),
        "--image_size", *exp['size'].split(),
        "--database_folder", exp['db'],
        "--queries_folder", exp['q'],
        "--distance_metric", exp['distance'],
        "--log_dir", log_dir,
        "--num_preds_to_save", "20",
        "--recall_values", "1", "5", "10", "20",
        "--save_for_uncertainty",
        "--num_workers", "8",
        "--batch_size", "32"
    ]
    
    print(f"\n{'='*80}")
    print(f"[{exp['id']}/32] {exp['method']} + {exp['distance']} + {exp['dataset']}")
    print(f"{'='*80}")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Experiment failed: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "="*80)
    print("Baseline实验进度检查")
    print("="*80)
    
    # 检查哪些已完成
    completed = []
    remaining = []
    
    for exp in ALL_EXPERIMENTS:
        if check_experiment_completed(exp):
            completed.append(exp)
        else:
            remaining.append(exp)
    
    print(f"\n已完成: {len(completed)}/32")
    print(f"剩余: {len(remaining)}/32")
    
    if completed:
        print("\n已完成的实验:")
        for exp in completed:
            print(f"  [{exp['id']}] {exp['method']} + {exp['distance']} + {exp['dataset']}")
    
    if not remaining:
        print("\n[SUCCESS] 所有实验已完成！")
        return
    
    print("\n" + "="*80)
    print(f"准备运行剩余{len(remaining)}个实验")
    print("="*80)
    
    response = input("\n是否继续运行剩余实验? (y/n): ")
    if response.lower() != 'y':
        print("\n已取消")
        print("\n如果要手动运行，可以逐个复制以下命令:")
        for exp in remaining:
            dataset_name = exp['dataset'].replace('-', '_').lower()
            print(f"\n# [{exp['id']}/32] {exp['method']} + {exp['distance']} + {exp['dataset']}")
            print(f"python VPR-methods-evaluation/main.py --method {exp['method']} --backbone {exp['backbone']} --descriptors_dimension {exp['dim']} --image_size {exp['size']} --database_folder {exp['db']} --queries_folder {exp['q']} --distance_metric {exp['distance']} --log_dir baseline/{exp['method']}_{exp['distance']}_{dataset_name}_test --num_preds_to_save 20 --recall_values 1 5 10 20 --save_for_uncertainty --num_workers 8 --batch_size 32")
        return
    
    # 运行剩余实验
    results = []
    for i, exp in enumerate(remaining):
        print(f"\n进度: {i+1}/{len(remaining)}")
        success = run_experiment(exp)
        results.append({
            "id": exp['id'],
            "method": exp['method'],
            "distance": exp['distance'],
            "dataset": exp['dataset'],
            "success": success
        })
    
    # 保存结果
    result_file = Path("logs/baseline_progress.json")
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total": 32,
            "completed_before": len(completed),
            "just_ran": len(remaining),
            "results": results
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("[SUCCESS] 实验运行完成！")
    print(f"进度已保存到: {result_file}")
    print("="*80)
    
    # 统计
    success_count = sum(1 for r in results if r['success'])
    failed_count = len(results) - success_count
    
    print(f"\n成功: {success_count}/{len(results)}")
    if failed_count > 0:
        print(f"失败: {failed_count}/{len(results)}")
        print("\n失败的实验:")
        for r in results:
            if not r['success']:
                print(f"  [{r['id']}] {r['method']} + {r['distance']} + {r['dataset']}")


if __name__ == "__main__":
    main()

