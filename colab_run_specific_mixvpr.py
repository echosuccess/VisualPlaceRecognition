#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab专用：运行4个特定的MixVPR+SVOX实验
"""

import subprocess
import os
import sys

# 设置UTF-8编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 要运行的4个实验
EXPERIMENTS = [
    {
        "name": "mixvpr_dot_product_svox_night_test",
        "command": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "mixvpr",
            "--backbone", "ResNet50",
            "--descriptors_dimension", "4096",
            "--image_size", "320", "320",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries_night",
            "--distance_metric", "dot_product",
            "--log_dir", "baseline/mixvpr_dot_product_svox_night_test",
            "--num_preds_to_save", "20",
            "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty",
            "--num_workers", "4",
            "--batch_size", "32",
        ],
    },
    {
        "name": "mixvpr_dot_product_svox_sun_test",
        "command": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "mixvpr",
            "--backbone", "ResNet50",
            "--descriptors_dimension", "4096",
            "--image_size", "320", "320",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries",
            "--distance_metric", "dot_product",
            "--log_dir", "baseline/mixvpr_dot_product_svox_sun_test",
            "--num_preds_to_save", "20",
            "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty",
            "--num_workers", "4",
            "--batch_size", "32",
        ],
    },
    {
        "name": "mixvpr_l2_svox_night_test",
        "command": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "mixvpr",
            "--backbone", "ResNet50",
            "--descriptors_dimension", "4096",
            "--image_size", "320", "320",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries_night",
            "--distance_metric", "l2",
            "--log_dir", "baseline/mixvpr_l2_svox_night_test",
            "--num_preds_to_save", "20",
            "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty",
            "--num_workers", "4",
            "--batch_size", "32",
        ],
    },
    {
        "name": "mixvpr_l2_svox_sun_test",
        "command": [
            "python", "VPR-methods-evaluation/main.py",
            "--method", "mixvpr",
            "--backbone", "ResNet50",
            "--descriptors_dimension", "4096",
            "--image_size", "320", "320",
            "--database_folder", "data/svox/images/test/gallery",
            "--queries_folder", "data/svox/images/test/queries",
            "--distance_metric", "l2",
            "--log_dir", "baseline/mixvpr_l2_svox_sun_test",
            "--num_preds_to_save", "20",
            "--max_queries_to_save", "3",
            "--recall_values", "1", "5", "10", "20",
            "--save_for_uncertainty",
            "--num_workers", "4",
            "--batch_size", "32",
        ],
    },
]


def check_completed(exp_name):
    """检查实验是否已完成"""
    log_dir = f"logs/logs/baseline/{exp_name}"
    if os.path.exists(log_dir):
        results_file = os.path.join(log_dir, "results.txt")
        if os.path.exists(results_file):
            return True
    return False


def run_experiment(exp):
    """运行单个实验"""
    name = exp["name"]
    command = exp["command"]
    
    print(f"\n{'='*70}")
    print(f"🚀 运行实验: {name}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=False,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        print(f"\n✅ 成功: {name}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 失败: {name}")
        print(f"错误: {e}\n")
        return False
    except Exception as e:
        print(f"\n❌ 失败: {name}")
        print(f"错误: {str(e)}\n")
        return False


def main():
    print("="*70)
    print("🎯 MixVPR + SVOX 实验运行器")
    print("="*70)
    
    # 检查哪些已完成
    total = len(EXPERIMENTS)
    completed_count = 0
    pending_experiments = []
    
    for exp in EXPERIMENTS:
        if check_completed(exp["name"]):
            print(f"✓ {exp['name']} (已完成)")
            completed_count += 1
        else:
            print(f"○ {exp['name']} (待运行)")
            pending_experiments.append(exp)
    
    print(f"\n进度: {completed_count}/{total} 已完成")
    
    if not pending_experiments:
        print("\n🎉 所有实验都已完成！")
        return
    
    print(f"\n将运行 {len(pending_experiments)} 个实验\n")
    
    success_count = 0
    failed_experiments = []
    
    for i, exp in enumerate(pending_experiments, 1):
        print(f"\n[{i}/{len(pending_experiments)}] 开始运行: {exp['name']}")
        
        if run_experiment(exp):
            success_count += 1
        else:
            failed_experiments.append(exp['name'])
    
    # 总结
    print("\n" + "="*70)
    print("📊 实验总结")
    print("="*70)
    print(f"✅ 成功: {success_count}/{len(pending_experiments)}")
    print(f"❌ 失败: {len(failed_experiments)}/{len(pending_experiments)}")
    
    if failed_experiments:
        print("\n失败的实验:")
        for name in failed_experiments:
            print(f"  - {name}")
    else:
        print("\n🎉 所有实验都成功完成！")


if __name__ == "__main__":
    main()
