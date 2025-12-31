#!/usr/bin/env python3
"""
自动运行所有剩余的Baseline实验
会按顺序执行，并显示进度
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 所有剩余实验的命令
EXPERIMENTS = [
    # CosPlace SVOX (4个)
    {
        "id": 3,
        "name": "cosplace + l2 + SVOX-sun",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "cosplace", "--backbone", "ResNet18",
            "--descriptors_dimension", "512", "--image_size", "512", "512",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries",
            "--distance_metric", "l2",
            "--log_dir", "baseline/cosplace_l2_svox_sun_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 4,
        "name": "cosplace + l2 + SVOX-night",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "cosplace", "--backbone", "ResNet18",
            "--descriptors_dimension", "512", "--image_size", "512", "512",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries_night",
            "--distance_metric", "l2",
            "--log_dir", "baseline/cosplace_l2_svox_night_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 7,
        "name": "cosplace + dot_product + SVOX-sun",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "cosplace", "--backbone", "ResNet18",
            "--descriptors_dimension", "512", "--image_size", "512", "512",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries",
            "--distance_metric", "dot_product",
            "--log_dir", "baseline/cosplace_dot_product_svox_sun_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 8,
        "name": "cosplace + dot_product + SVOX-night",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "cosplace", "--backbone", "ResNet18",
            "--descriptors_dimension", "512", "--image_size", "512", "512",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries_night",
            "--distance_metric", "dot_product",
            "--log_dir", "baseline/cosplace_dot_product_svox_night_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    
    # NetVLAD (8个)
    {
        "id": 9,
        "name": "netvlad + l2 + SF-XS",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "netvlad", "--backbone", "VGG16",
            "--descriptors_dimension", "4096", "--image_size", "512", "512",
            "--database_folder", "data/sf_xs/test/database",
            "--queries_folder", "data/sf_xs/test/queries",
            "--distance_metric", "l2",
            "--log_dir", "baseline/netvlad_l2_sf_xs_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 10,
        "name": "netvlad + l2 + Tokyo-XS",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "netvlad", "--backbone", "VGG16",
            "--descriptors_dimension", "4096", "--image_size", "512", "512",
            "--database_folder", "data/tokyo_xs/test/database",
            "--queries_folder", "data/tokyo_xs/test/queries",
            "--distance_metric", "l2",
            "--log_dir", "baseline/netvlad_l2_tokyo_xs_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 11,
        "name": "netvlad + l2 + SVOX-sun",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "netvlad", "--backbone", "VGG16",
            "--descriptors_dimension", "4096", "--image_size", "512", "512",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries",
            "--distance_metric", "l2",
            "--log_dir", "baseline/netvlad_l2_svox_sun_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 12,
        "name": "netvlad + l2 + SVOX-night",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "netvlad", "--backbone", "VGG16",
            "--descriptors_dimension", "4096", "--image_size", "512", "512",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries_night",
            "--distance_metric", "l2",
            "--log_dir", "baseline/netvlad_l2_svox_night_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 13,
        "name": "netvlad + dot_product + SF-XS",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "netvlad", "--backbone", "VGG16",
            "--descriptors_dimension", "4096", "--image_size", "512", "512",
            "--database_folder", "data/sf_xs/test/database",
            "--queries_folder", "data/sf_xs/test/queries",
            "--distance_metric", "dot_product",
            "--log_dir", "baseline/netvlad_dot_product_sf_xs_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 14,
        "name": "netvlad + dot_product + Tokyo-XS",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "netvlad", "--backbone", "VGG16",
            "--descriptors_dimension", "4096", "--image_size", "512", "512",
            "--database_folder", "data/tokyo_xs/test/database",
            "--queries_folder", "data/tokyo_xs/test/queries",
            "--distance_metric", "dot_product",
            "--log_dir", "baseline/netvlad_dot_product_tokyo_xs_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 15,
        "name": "netvlad + dot_product + SVOX-sun",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "netvlad", "--backbone", "VGG16",
            "--descriptors_dimension", "4096", "--image_size", "512", "512",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries",
            "--distance_metric", "dot_product",
            "--log_dir", "baseline/netvlad_dot_product_svox_sun_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 16,
        "name": "netvlad + dot_product + SVOX-night",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "netvlad", "--backbone", "VGG16",
            "--descriptors_dimension", "4096", "--image_size", "512", "512",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries_night",
            "--distance_metric", "dot_product",
            "--log_dir", "baseline/netvlad_dot_product_svox_night_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    
    # MixVPR (8个)
    {
        "id": 17,
        "name": "mixvpr + l2 + SF-XS",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "mixvpr", "--backbone", "ResNet50",
            "--descriptors_dimension", "4096", "--image_size", "320", "320",
            "--database_folder", "data/sf_xs/test/database",
            "--queries_folder", "data/sf_xs/test/queries",
            "--distance_metric", "l2",
            "--log_dir", "baseline/mixvpr_l2_sf_xs_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 18,
        "name": "mixvpr + l2 + Tokyo-XS",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "mixvpr", "--backbone", "ResNet50",
            "--descriptors_dimension", "4096", "--image_size", "320", "320",
            "--database_folder", "data/tokyo_xs/test/database",
            "--queries_folder", "data/tokyo_xs/test/queries",
            "--distance_metric", "l2",
            "--log_dir", "baseline/mixvpr_l2_tokyo_xs_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 19,
        "name": "mixvpr + l2 + SVOX-sun",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "mixvpr", "--backbone", "ResNet50",
            "--descriptors_dimension", "4096", "--image_size", "320", "320",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries",
            "--distance_metric", "l2",
            "--log_dir", "baseline/mixvpr_l2_svox_sun_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 20,
        "name": "mixvpr + l2 + SVOX-night",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "mixvpr", "--backbone", "ResNet50",
            "--descriptors_dimension", "4096", "--image_size", "320", "320",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries_night",
            "--distance_metric", "l2",
            "--log_dir", "baseline/mixvpr_l2_svox_night_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 21,
        "name": "mixvpr + dot_product + SF-XS",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "mixvpr", "--backbone", "ResNet50",
            "--descriptors_dimension", "4096", "--image_size", "320", "320",
            "--database_folder", "data/sf_xs/test/database",
            "--queries_folder", "data/sf_xs/test/queries",
            "--distance_metric", "dot_product",
            "--log_dir", "baseline/mixvpr_dot_product_sf_xs_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 22,
        "name": "mixvpr + dot_product + Tokyo-XS",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "mixvpr", "--backbone", "ResNet50",
            "--descriptors_dimension", "4096", "--image_size", "320", "320",
            "--database_folder", "data/tokyo_xs/test/database",
            "--queries_folder", "data/tokyo_xs/test/queries",
            "--distance_metric", "dot_product",
            "--log_dir", "baseline/mixvpr_dot_product_tokyo_xs_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 23,
        "name": "mixvpr + dot_product + SVOX-sun",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "mixvpr", "--backbone", "ResNet50",
            "--descriptors_dimension", "4096", "--image_size", "320", "320",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries",
            "--distance_metric", "dot_product",
            "--log_dir", "baseline/mixvpr_dot_product_svox_sun_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 24,
        "name": "mixvpr + dot_product + SVOX-night",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "mixvpr", "--backbone", "ResNet50",
            "--descriptors_dimension", "4096", "--image_size", "320", "320",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries_night",
            "--distance_metric", "dot_product",
            "--log_dir", "baseline/mixvpr_dot_product_svox_night_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    
    # MegaLoc剩余 (6个)
    {
        "id": 25,
        "name": "megaloc + l2 + SF-XS",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "megaloc", "--backbone", "Dinov2",
            "--descriptors_dimension", "8448", "--image_size", "224", "224",
            "--database_folder", "data/sf_xs/test/database",
            "--queries_folder", "data/sf_xs/test/queries",
            "--distance_metric", "l2",
            "--log_dir", "baseline/megaloc_l2_sf_xs_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 27,
        "name": "megaloc + l2 + SVOX-sun",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "megaloc", "--backbone", "Dinov2",
            "--descriptors_dimension", "8448", "--image_size", "224", "224",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries",
            "--distance_metric", "l2",
            "--log_dir", "baseline/megaloc_l2_svox_sun_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 28,
        "name": "megaloc + l2 + SVOX-night",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "megaloc", "--backbone", "Dinov2",
            "--descriptors_dimension", "8448", "--image_size", "224", "224",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries_night",
            "--distance_metric", "l2",
            "--log_dir", "baseline/megaloc_l2_svox_night_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 29,
        "name": "megaloc + dot_product + SF-XS",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "megaloc", "--backbone", "Dinov2",
            "--descriptors_dimension", "8448", "--image_size", "224", "224",
            "--database_folder", "data/sf_xs/test/database",
            "--queries_folder", "data/sf_xs/test/queries",
            "--distance_metric", "dot_product",
            "--log_dir", "baseline/megaloc_dot_product_sf_xs_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 30,
        "name": "megaloc + dot_product + Tokyo-XS",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "megaloc", "--backbone", "Dinov2",
            "--descriptors_dimension", "8448", "--image_size", "224", "224",
            "--database_folder", "data/tokyo_xs/test/database",
            "--queries_folder", "data/tokyo_xs/test/queries",
            "--distance_metric", "dot_product",
            "--log_dir", "baseline/megaloc_dot_product_tokyo_xs_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 31,
        "name": "megaloc + dot_product + SVOX-sun",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "megaloc", "--backbone", "Dinov2",
            "--descriptors_dimension", "8448", "--image_size", "224", "224",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries",
            "--distance_metric", "dot_product",
            "--log_dir", "baseline/megaloc_dot_product_svox_sun_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
    {
        "id": 32,
        "name": "megaloc + dot_product + SVOX-night",
        "cmd": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "megaloc", "--backbone", "Dinov2",
            "--descriptors_dimension", "8448", "--image_size", "224", "224",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries_night",
            "--distance_metric", "dot_product",
            "--log_dir", "baseline/megaloc_dot_product_svox_night_test",
            "--num_preds_to_save", "20", "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty", "--num_workers", "8", "--batch_size", "32"
        ]
    },
]


def run_all_experiments():
    """运行所有实验"""
    total = len(EXPERIMENTS)
    success_count = 0
    failed = []
    
    print("=" * 80)
    print(f"准备运行 {total} 个剩余实验")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{total}] 实验 #{exp['id']}: {exp['name']}")
        print(f"{'='*80}")
        
        try:
            result = subprocess.run(
                exp['cmd'],
                check=True,
                capture_output=False  # 显示实时输出
            )
            success_count += 1
            print(f"\n[SUCCESS] 实验 #{exp['id']} 完成!")
            
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] 实验 #{exp['id']} 失败: {e}")
            failed.append({
                "id": exp['id'],
                "name": exp['name'],
                "error": str(e)
            })
            
            # 询问是否继续
            response = input("\n实验失败！是否继续下一个? (y/n): ")
            if response.lower() != 'y':
                print("\n用户中止执行")
                break
        
        except KeyboardInterrupt:
            print("\n\n用户中断!")
            break
    
    # 最终统计
    print("\n" + "=" * 80)
    print("实验运行完成!")
    print("=" * 80)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"成功: {success_count}/{total}")
    print(f"失败: {len(failed)}/{total}")
    
    if failed:
        print("\n失败的实验:")
        for f in failed:
            print(f"  - #{f['id']}: {f['name']}")
    
    # 保存结果
    log_file = Path("logs/run_all_progress.txt")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"运行时间: {datetime.now().isoformat()}\n")
        f.write(f"成功: {success_count}/{total}\n")
        f.write(f"失败: {len(failed)}/{total}\n\n")
        if failed:
            f.write("失败的实验:\n")
            for exp in failed:
                f.write(f"  - #{exp['id']}: {exp['name']}\n")
    
    print(f"\n进度日志已保存到: {log_file}")


if __name__ == "__main__":
    # 检查是否确认运行
    print("\n即将运行所有 26 个剩余实验")
    print("预计需要 3-8 小时")
    print("\n你可以随时按 Ctrl+C 中断")
    
    response = input("\n确认开始运行? (y/n): ")
    if response.lower() == 'y':
        run_all_experiments()
    else:
        print("已取消")

