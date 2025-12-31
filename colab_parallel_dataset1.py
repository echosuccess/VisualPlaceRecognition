#!/usr/bin/env python3
"""
Colab并行执行 - 数据集组1：SF-XS + Tokyo-XS
适合在Notebook 1中运行
"""

import subprocess
import sys

# 要运行的实验
experiments = [
    # CosPlace - SF-XS
    ["python", "VPR-methods-evaluation/main.py",
     "--method", "cosplace", "--backbone", "ResNet18",
     "--descriptors_dimension", "512", "--image_size", "512", "512",
     "--database_folder", "data/sf_xs/test/database",
     "--queries_folder", "data/sf_xs/test/queries",
     "--distance_metric", "l2",
     "--log_dir", "baseline/cosplace_l2_sf_xs_test",
     "--num_preds_to_save", "20", "--max_queries_to_save", "3",
     "--recall_values", "1", "5", "10", "20",
     "--save_for_uncertainty", "--num_workers", "4", "--batch_size", "32"],
    
    # CosPlace - SF-XS - Dot Product
    ["python", "VPR-methods-evaluation/main.py",
     "--method", "cosplace", "--backbone", "ResNet18",
     "--descriptors_dimension", "512", "--image_size", "512", "512",
     "--database_folder", "data/sf_xs/test/database",
     "--queries_folder", "data/sf_xs/test/queries",
     "--distance_metric", "dot_product",
     "--log_dir", "baseline/cosplace_dot_product_sf_xs_test",
     "--num_preds_to_save", "20", "--max_queries_to_save", "3",
     "--recall_values", "1", "5", "10", "20",
     "--save_for_uncertainty", "--num_workers", "4", "--batch_size", "32"],
    
    # CosPlace - Tokyo-XS
    ["python", "VPR-methods-evaluation/main.py",
     "--method", "cosplace", "--backbone", "ResNet18",
     "--descriptors_dimension", "512", "--image_size", "512", "512",
     "--database_folder", "data/tokyo_xs/test/database",
     "--queries_folder", "data/tokyo_xs/test/queries",
     "--distance_metric", "l2",
     "--log_dir", "baseline/cosplace_l2_tokyo_xs_test",
     "--num_preds_to_save", "20", "--max_queries_to_save", "3",
     "--recall_values", "1", "5", "10", "20",
     "--save_for_uncertainty", "--num_workers", "4", "--batch_size", "32"],
    
    # CosPlace - Tokyo-XS - Dot Product
    ["python", "VPR-methods-evaluation/main.py",
     "--method", "cosplace", "--backbone", "ResNet18",
     "--descriptors_dimension", "512", "--image_size", "512", "512",
     "--database_folder", "data/tokyo_xs/test/database",
     "--queries_folder", "data/tokyo_xs/test/queries",
     "--distance_metric", "dot_product",
     "--log_dir", "baseline/cosplace_dot_product_tokyo_xs_test",
     "--num_preds_to_save", "20", "--max_queries_to_save", "3",
     "--recall_values", "1", "5", "10", "20",
     "--save_for_uncertainty", "--num_workers", "4", "--batch_size", "32"],
]

def main():
    print("="*70)
    print("  🚀 并行执行组1：SF-XS + Tokyo-XS (CosPlace)")
    print("  实验数量: 4")
    print("  预计时间: 1-2小时")
    print("="*70)
    
    for i, cmd in enumerate(experiments, 1):
        log_dir = cmd[cmd.index("--log_dir") + 1]
        print(f"\n[{i}/{len(experiments)}] 运行: {log_dir}")
        print("-"*70)
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ 完成: {log_dir}")
        except subprocess.CalledProcessError as e:
            print(f"❌ 失败: {log_dir}")
            print(f"错误: {e}")
            response = input("继续下一个实验？ [Y/n]: ").strip().lower()
            if response == 'n':
                sys.exit(1)
    
    print("\n" + "="*70)
    print("🎉 所有实验完成！")
    print("="*70)

if __name__ == "__main__":
    main()
