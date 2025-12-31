#!/usr/bin/env python3
"""列出所有剩余需要运行的实验命令"""
from pathlib import Path
import re

# 定义所有32个实验配置
ALL_EXPERIMENTS = [
    # CosPlace (8个)
    {"id": 1, "method": "cosplace", "backbone": "ResNet18", "dim": 512, "size": "512 512", "distance": "l2", "dataset": "SF-XS", "db": "data/sf_xs/test/database", "q": "data/sf_xs/test/queries"},
    {"id": 2, "method": "cosplace", "backbone": "ResNet18", "dim": 512, "size": "512 512", "distance": "l2", "dataset": "Tokyo-XS", "db": "data/tokyo_xs/test/database", "q": "data/tokyo_xs/test/queries"},
    {"id": 3, "method": "cosplace", "backbone": "ResNet18", "dim": 512, "size": "512 512", "distance": "l2", "dataset": "SVOX-sun", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries"},
    {"id": 4, "method": "cosplace", "backbone": "ResNet18", "dim": 512, "size": "512 512", "distance": "l2", "dataset": "SVOX-night", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries_night"},
    {"id": 5, "method": "cosplace", "backbone": "ResNet18", "dim": 512, "size": "512 512", "distance": "dot_product", "dataset": "SF-XS", "db": "data/sf_xs/test/database", "q": "data/sf_xs/test/queries"},
    {"id": 6, "method": "cosplace", "backbone": "ResNet18", "dim": 512, "size": "512 512", "distance": "dot_product", "dataset": "Tokyo-XS", "db": "data/tokyo_xs/test/database", "q": "data/tokyo_xs/test/queries"},
    {"id": 7, "method": "cosplace", "backbone": "ResNet18", "dim": 512, "size": "512 512", "distance": "dot_product", "dataset": "SVOX-sun", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries"},
    {"id": 8, "method": "cosplace", "backbone": "ResNet18", "dim": 512, "size": "512 512", "distance": "dot_product", "dataset": "SVOX-night", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries_night"},
    
    # NetVLAD (8个)
    {"id": 9, "method": "netvlad", "backbone": "VGG16", "dim": 4096, "size": "512 512", "distance": "l2", "dataset": "SF-XS", "db": "data/sf_xs/test/database", "q": "data/sf_xs/test/queries"},
    {"id": 10, "method": "netvlad", "backbone": "VGG16", "dim": 4096, "size": "512 512", "distance": "l2", "dataset": "Tokyo-XS", "db": "data/tokyo_xs/test/database", "q": "data/tokyo_xs/test/queries"},
    {"id": 11, "method": "netvlad", "backbone": "VGG16", "dim": 4096, "size": "512 512", "distance": "l2", "dataset": "SVOX-sun", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries"},
    {"id": 12, "method": "netvlad", "backbone": "VGG16", "dim": 4096, "size": "512 512", "distance": "l2", "dataset": "SVOX-night", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries_night"},
    {"id": 13, "method": "netvlad", "backbone": "VGG16", "dim": 4096, "size": "512 512", "distance": "dot_product", "dataset": "SF-XS", "db": "data/sf_xs/test/database", "q": "data/sf_xs/test/queries"},
    {"id": 14, "method": "netvlad", "backbone": "VGG16", "dim": 4096, "size": "512 512", "distance": "dot_product", "dataset": "Tokyo-XS", "db": "data/tokyo_xs/test/database", "q": "data/tokyo_xs/test/queries"},
    {"id": 15, "method": "netvlad", "backbone": "VGG16", "dim": 4096, "size": "512 512", "distance": "dot_product", "dataset": "SVOX-sun", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries"},
    {"id": 16, "method": "netvlad", "backbone": "VGG16", "dim": 4096, "size": "512 512", "distance": "dot_product", "dataset": "SVOX-night", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries_night"},
    
    # MixVPR (8个)
    {"id": 17, "method": "mixvpr", "backbone": "ResNet50", "dim": 4096, "size": "320 320", "distance": "l2", "dataset": "SF-XS", "db": "data/sf_xs/test/database", "q": "data/sf_xs/test/queries"},
    {"id": 18, "method": "mixvpr", "backbone": "ResNet50", "dim": 4096, "size": "320 320", "distance": "l2", "dataset": "Tokyo-XS", "db": "data/tokyo_xs/test/database", "q": "data/tokyo_xs/test/queries"},
    {"id": 19, "method": "mixvpr", "backbone": "ResNet50", "dim": 4096, "size": "320 320", "distance": "l2", "dataset": "SVOX-sun", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries"},
    {"id": 20, "method": "mixvpr", "backbone": "ResNet50", "dim": 4096, "size": "320 320", "distance": "l2", "dataset": "SVOX-night", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries_night"},
    {"id": 21, "method": "mixvpr", "backbone": "ResNet50", "dim": 4096, "size": "320 320", "distance": "dot_product", "dataset": "SF-XS", "db": "data/sf_xs/test/database", "q": "data/sf_xs/test/queries"},
    {"id": 22, "method": "mixvpr", "backbone": "ResNet50", "dim": 4096, "size": "320 320", "distance": "dot_product", "dataset": "Tokyo-XS", "db": "data/tokyo_xs/test/database", "q": "data/tokyo_xs/test/queries"},
    {"id": 23, "method": "mixvpr", "backbone": "ResNet50", "dim": 4096, "size": "320 320", "distance": "dot_product", "dataset": "SVOX-sun", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries"},
    {"id": 24, "method": "mixvpr", "backbone": "ResNet50", "dim": 4096, "size": "320 320", "distance": "dot_product", "dataset": "SVOX-night", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries_night"},
    
    # MegaLoc (8个)
    {"id": 25, "method": "megaloc", "backbone": "Dinov2", "dim": 8448, "size": "224 224", "distance": "l2", "dataset": "SF-XS", "db": "data/sf_xs/test/database", "q": "data/sf_xs/test/queries"},
    {"id": 26, "method": "megaloc", "backbone": "Dinov2", "dim": 8448, "size": "224 224", "distance": "l2", "dataset": "Tokyo-XS", "db": "data/tokyo_xs/test/database", "q": "data/tokyo_xs/test/queries"},
    {"id": 27, "method": "megaloc", "backbone": "Dinov2", "dim": 8448, "size": "224 224", "distance": "l2", "dataset": "SVOX-sun", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries"},
    {"id": 28, "method": "megaloc", "backbone": "Dinov2", "dim": 8448, "size": "224 224", "distance": "l2", "dataset": "SVOX-night", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries_night"},
    {"id": 29, "method": "megaloc", "backbone": "Dinov2", "dim": 8448, "size": "224 224", "distance": "dot_product", "dataset": "SF-XS", "db": "data/sf_xs/test/database", "q": "data/sf_xs/test/queries"},
    {"id": 30, "method": "megaloc", "backbone": "Dinov2", "dim": 8448, "size": "224 224", "distance": "dot_product", "dataset": "Tokyo-XS", "db": "data/tokyo_xs/test/database", "q": "data/tokyo_xs/test/queries"},
    {"id": 31, "method": "megaloc", "backbone": "Dinov2", "dim": 8448, "size": "224 224", "distance": "dot_product", "dataset": "SVOX-sun", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries"},
    {"id": 32, "method": "megaloc", "backbone": "Dinov2", "dim": 8448, "size": "224 224", "distance": "dot_product", "dataset": "SVOX-night", "db": "data/svox/images/test/gallery", "q": "data/svox/images/test/queries_night"},
]


def check_completed(exp):
    """检查实验是否已完成"""
    log_bases = [Path("logs/baseline"), Path("logs/logs/baseline")]
    
    # 构造所有可能的目录名格式（实际运行时的格式很不统一）
    dataset_lower = exp['dataset'].replace('-', '_').lower()
    dataset_orig = exp['dataset']
    
    possible_names = [
        # 实际观察到的格式
        f"{exp['method']}_{exp['distance']}_{dataset_orig}",  # cosplace_l2_Tokyo-XS
        f"{exp['method']}_{exp['distance']}_{dataset_orig}-test",  # cosplace_l2_SF-XS-test
        f"{exp['method']}_{exp['distance']}_{dataset_lower}_test",  # cosplace_l2_sf_xs_test
        # 其他可能的格式
        f"{exp['method']}_{exp['distance']}_{dataset_orig}_test",
        f"{exp['method']}_{exp['distance']}_{dataset_lower}",
    ]
    
    for base in log_bases:
        if not base.exists():
            continue
        for name in possible_names:
            log_dir = base / name
            if log_dir.exists():
                for subdir in log_dir.iterdir():
                    if subdir.is_dir():
                        info_log = subdir / "info.log"
                        if info_log.exists():
                            try:
                                content = info_log.read_text(encoding='utf-8', errors='ignore')
                                match = re.search(r'R@1:\s*([\d.]+)', content)
                                if match:
                                    return True, match.group(1)
                            except:
                                pass
    return False, None


def main():
    print("=" * 80)
    print("Baseline 实验状态检查")
    print("=" * 80)
    
    completed = []
    remaining = []
    
    for exp in ALL_EXPERIMENTS:
        is_completed, r1 = check_completed(exp)
        if is_completed:
            completed.append((exp, r1))
        else:
            remaining.append(exp)
    
    print(f"\n[OK] 已完成: {len(completed)}/32")
    for exp, r1 in completed:
        print(f"  [{exp['id']:2d}] {exp['method']:10s} + {exp['distance']:12s} + {exp['dataset']:12s} -> R@1 = {r1}%")
    
    print(f"\n[TODO] 剩余: {len(remaining)}/32")
    
    if not remaining:
        print("\n[SUCCESS] 所有实验已完成!")
        return
    
    print("\n" + "=" * 80)
    print(f"剩余实验命令（复制以下命令逐个运行）")
    print("=" * 80)
    
    for exp in remaining:
        dataset_name = exp['dataset'].replace('-', '_').lower()
        log_dir = f"baseline/{exp['method']}_{exp['distance']}_{dataset_name}_test"
        
        print(f"\n# [{exp['id']}/32] {exp['method']} + {exp['distance']} + {exp['dataset']}")
        print(f"python VPR-methods-evaluation/main.py \\")
        print(f"  --method {exp['method']} --backbone {exp['backbone']} \\")
        print(f"  --descriptors_dimension {exp['dim']} --image_size {exp['size']} \\")
        print(f"  --database_folder {exp['db']} --queries_folder {exp['q']} \\")
        print(f"  --distance_metric {exp['distance']} --log_dir {log_dir} \\")
        print(f"  --num_preds_to_save 20 --recall_values 1 5 10 20 \\")
        print(f"  --save_for_uncertainty --num_workers 8 --batch_size 32")


if __name__ == "__main__":
    main()

