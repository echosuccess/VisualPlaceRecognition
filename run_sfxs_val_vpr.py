#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行SF-XS val的VPR实验（Extension 6.1验证集）
"""

import subprocess
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime

def run_vpr_experiment(method, backbone, dim, image_size, log_name):
    """运行单个VPR实验"""
    
    cmd = [
        "python", "VPR-methods-evaluation/main.py",
        "--method", method,
        "--backbone", backbone,
        "--descriptors_dimension", str(dim),
        "--image_size", *image_size.split(),
        "--database_folder", "data/sf_xs/val/database",
        "--queries_folder", "data/sf_xs/val/queries",
        "--distance_metric", "l2",
        "--log_dir", f"baseline/{log_name}",
        "--num_preds_to_save", "20",
        "--max_queries_to_save", "3",
        "--recall_values", "1", "5", "10", "20",
        "--save_for_uncertainty",
        "--num_workers", "8",
        "--batch_size", "32"
    ]
    
    print(f"\n{'='*80}")
    print(f"运行: {log_name}")
    print(f"{'='*80}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ {log_name} 完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {log_name} 失败: {e}")
        return False

def save_checkpoint(completed_experiments, checkpoint_path):
    """保存checkpoint"""
    checkpoint_data = {
        'completed': completed_experiments,
        'timestamp': datetime.now().isoformat()
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    print(f"[CHECKPOINT] 已保存: {checkpoint_path}")

def load_checkpoint(checkpoint_path):
    """加载checkpoint"""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            return checkpoint_data.get('completed', [])
        except Exception as e:
            print(f"[WARN] 无法加载checkpoint: {e}")
    return []

def main():
    """主函数"""
    
    print("="*80)
    print("SF-XS val VPR实验（Extension 6.1验证集）")
    print("="*80)
    
    # Checkpoint路径
    checkpoint_path = Path("checkpoints/sfxs_val_vpr.json")
    
    # 检查数据文件夹
    val_data_path = Path("data/sf_xs/val")
    if not val_data_path.exists():
        print(f"\n❌ 错误: {val_data_path} 不存在！")
        print("请先准备SF-XS val数据集")
        return
    
    # 定义所有实验
    experiments = [
        {
            "method": "cosplace",
            "backbone": "ResNet18",
            "dim": "512",
            "image_size": "512 512",
            "log_name": "cosplace_l2_sfxs_val"
        },
        {
            "method": "netvlad",
            "backbone": "VGG16",
            "dim": "4096",
            "image_size": "512 512",
            "log_name": "netvlad_l2_sfxs_val"
        },
        {
            "method": "mixvpr",
            "backbone": "ResNet50",
            "dim": "4096",
            "image_size": "320 320",
            "log_name": "mixvpr_l2_sfxs_val"
        },
        {
            "method": "megaloc",
            "backbone": "Dinov2",
            "dim": "8448",
            "image_size": "224 224",
            "log_name": "megaloc_l2_sfxs_val"
        }
    ]
    
    print(f"\n将运行 {len(experiments)} 个VPR实验")
    print("\n实验列表:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['log_name']}")
    
    # 加载checkpoint
    checkpoint_completed = load_checkpoint(checkpoint_path)
    if checkpoint_completed:
        print(f"\n[CHECKPOINT] 从checkpoint恢复: {len(checkpoint_completed)} 个已完成实验")
    
    # 检查哪些已经完成（包括检查checkpoint和实际文件）
    print("\n检查已完成实验:")
    completed = []
    remaining = []
    
    for exp in experiments:
        log_dir = Path(f"logs/baseline/{exp['log_name']}")
        # 检查实际文件
        z_data_files = list(log_dir.glob("**/z_data.torch"))
        
        # 如果文件存在，或者checkpoint中标记为完成
        if z_data_files or exp['log_name'] in checkpoint_completed:
            completed.append(exp['log_name'])
            if z_data_files:
                print(f"  ✅ {exp['log_name']} (已完成)")
            else:
                print(f"  ⚠️  {exp['log_name']} (checkpoint标记为完成，但文件不存在，将重新运行)")
                remaining.append(exp)
        else:
            remaining.append(exp)
    
    if completed:
        print(f"\n已完成: {len(completed)}/{len(experiments)}")
    
    if not remaining:
        print("\n✅ 所有实验已完成！")
        # 清理checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"[INFO] 已清理checkpoint文件")
        return
    
    print(f"\n剩余: {len(remaining)}/{len(experiments)}")
    print("\n开始运行剩余实验...")
    
    # 运行剩余实验，每完成一个就保存checkpoint
    for i, exp in enumerate(remaining, 1):
        print(f"\n进度: {i}/{len(remaining)}")
        success = run_vpr_experiment(
            exp["method"],
            exp["backbone"],
            exp["dim"],
            exp["image_size"],
            exp["log_name"]
        )
        
        if success:
            # 验证文件确实存在
            log_dir = Path(f"logs/baseline/{exp['log_name']}")
            z_data_files = list(log_dir.glob("**/z_data.torch"))
            if z_data_files:
                completed.append(exp['log_name'])
                # 保存checkpoint
                save_checkpoint(completed, checkpoint_path)
                print(f"[CHECKPOINT] 实验 {exp['log_name']} 完成，已保存checkpoint")
        else:
            print(f"\n⚠️  实验失败，但继续运行下一个...")
    
    print("\n" + "="*80)
    print("所有实验完成！")
    print("="*80)
    
    # 清理checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[INFO] 已清理checkpoint文件（所有实验已完成）")
    
    print("\n下一步：运行SF-XS val的Image Matching实验")
    print("命令: python run_sfxs_val_image_matching.py")

if __name__ == "__main__":
    main()
