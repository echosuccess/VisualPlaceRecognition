#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行运行SF-XS val的Image Matching实验（Extension 6.1验证集）
用于在Colab中并行执行多个实验
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def run_im_experiment(matcher, vpr_method, dataset, distance, device='cuda'):
    """运行单个Image Matching实验"""
    
    cmd = [
        "python", "run_image_matching_baseline.py",
        "--matcher", matcher,
        "--vpr_method", vpr_method,
        "--dataset", dataset,
        "--distance", distance,
        "--device", device
    ]
    
    exp_key = f"{matcher}_{vpr_method}"
    print(f"\n{'='*80}")
    print(f"[开始] {exp_key}")
    print(f"{'='*80}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {exp_key} 完成！")
        return True, exp_key
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {exp_key} 失败: {e}")
        return False, exp_key

def main():
    """主函数"""
    
    print("="*80)
    print("SF-XS val Image Matching实验 - 并行执行（Extension 6.1验证集）")
    print("="*80)
    
    # 检查VPR结果是否存在
    vpr_methods = ['cosplace', 'netvlad', 'mixvpr', 'megaloc']
    missing_vpr = []
    
    for method in vpr_methods:
        vpr_dir = Path(f"logs/baseline/{method}_l2_sfxs_val")
        if not vpr_dir.exists():
            missing_vpr.append(method)
    
    if missing_vpr:
        print(f"\n❌ 错误: 以下VPR实验尚未完成:")
        for m in missing_vpr:
            print(f"  - {m}_l2_sfxs_val")
        print("\n请先运行: python run_sfxs_val_vpr.py")
        return
    
    # 定义所有实验
    experiments = []
    matchers = ['superpoint-lg', 'loftr', 'superglue']
    
    for vpr_method in vpr_methods:
        for matcher in matchers:
            experiments.append({
                'matcher': matcher,
                'vpr_method': vpr_method,
                'dataset': 'sfxs_val',
                'distance': 'l2'
            })
    
    print(f"\n总共 {len(experiments)} 个Image Matching实验")
    
    # Checkpoint路径
    checkpoint_path = Path("checkpoints/sfxs_val_image_matching.json")
    
    # 加载checkpoint
    checkpoint_completed = []
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            checkpoint_completed = checkpoint_data.get('completed', [])
            if checkpoint_completed:
                print(f"\n[CHECKPOINT] 从checkpoint恢复: {len(checkpoint_completed)} 个已完成实验")
        except Exception as e:
            print(f"[WARN] 无法加载checkpoint: {e}")
    
    # 检查哪些已经完成（包括checkpoint和实际文件）
    print("\n检查已完成实验:")
    completed = []
    remaining = []
    
    for exp in experiments:
        exp_key = f"{exp['matcher']}_{exp['vpr_method']}"
        result_file = Path(f"results/image_matching/{exp['matcher']}_{exp['vpr_method']}_l2_{exp['dataset']}.json")
        result_file_pkl = Path(f"results/image_matching/{exp['matcher']}_{exp['vpr_method']}_l2_{exp['dataset']}.pkl")
        
        # 检查实际文件或checkpoint
        if result_file.exists() or result_file_pkl.exists() or exp_key in checkpoint_completed:
            completed.append(exp_key)
            if result_file.exists() or result_file_pkl.exists():
                print(f"  ✅ {exp['matcher']} + {exp['vpr_method']} (已完成)")
            else:
                print(f"  ⚠️  {exp['matcher']} + {exp['vpr_method']} (checkpoint标记为完成，但文件不存在，将重新运行)")
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
    print("\n开始并行运行剩余实验...")
    
    # 保存checkpoint的函数
    def save_checkpoint(completed_list, checkpoint_path):
        checkpoint_data = {
            'completed': completed_list,
            'timestamp': datetime.now().isoformat()
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        print(f"[CHECKPOINT] 已保存: {checkpoint_path}")
    
    # 并行执行（使用多进程）
    # 注意：在Colab中，由于GPU限制，建议使用较小的并行数
    # 如果只有一个GPU，建议max_workers=1（顺序执行）
    # 如果有多个GPU或想利用CPU，可以设置max_workers=2-4
    
    max_workers = 1  # 默认顺序执行（避免GPU冲突）
    print(f"\n[INFO] 使用 {max_workers} 个并行进程（建议在Colab中保持为1以避免GPU冲突）")
    print(f"[INFO] 如果想并行执行，请在多个Colab cell中分别运行不同的实验")
    
    # 如果只有一个worker，顺序执行（更安全）
    if max_workers == 1:
        for i, exp in enumerate(remaining, 1):
            print(f"\n进度: {i}/{len(remaining)}")
            success, exp_key = run_im_experiment(
                exp['matcher'],
                exp['vpr_method'],
                exp['dataset'],
                exp['distance']
            )
            
            if success:
                # 验证文件确实存在
                result_file = Path(f"results/image_matching/{exp['matcher']}_{exp['vpr_method']}_l2_{exp['dataset']}.json")
                result_file_pkl = Path(f"results/image_matching/{exp['matcher']}_{exp['vpr_method']}_l2_{exp['dataset']}.pkl")
                if result_file.exists() or result_file_pkl.exists():
                    completed.append(exp_key)
                    # 保存checkpoint
                    save_checkpoint(completed, checkpoint_path)
                    print(f"[CHECKPOINT] 实验 {exp_key} 完成，已保存checkpoint")
            else:
                print(f"\n⚠️  实验失败，但继续运行下一个...")
    else:
        # 多进程并行执行（注意：在Colab中可能不适用，因为GPU限制）
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for exp in remaining:
                future = executor.submit(
                    run_im_experiment,
                    exp['matcher'],
                    exp['vpr_method'],
                    exp['dataset'],
                    exp['distance']
                )
                futures[future] = exp
            
            for i, future in enumerate(as_completed(futures), 1):
                exp = futures[future]
                exp_key = f"{exp['matcher']}_{exp['vpr_method']}"
                try:
                    success, _ = future.result()
                    if success:
                        # 验证文件确实存在
                        result_file = Path(f"results/image_matching/{exp['matcher']}_{exp['vpr_method']}_l2_{exp['dataset']}.json")
                        result_file_pkl = Path(f"results/image_matching/{exp['matcher']}_{exp['vpr_method']}_l2_{exp['dataset']}.pkl")
                        if result_file.exists() or result_file_pkl.exists():
                            completed.append(exp_key)
                            save_checkpoint(completed, checkpoint_path)
                            print(f"[CHECKPOINT] 实验 {exp_key} 完成，已保存checkpoint")
                except Exception as e:
                    print(f"\n⚠️  实验 {exp_key} 异常: {e}")
    
    print("\n" + "="*80)
    print("所有实验完成！")
    print("="*80)
    
    # 清理checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[INFO] 已清理checkpoint文件（所有实验已完成）")
    
    print("\n下一步：运行Extension 6.1")
    print("命令: python run_extension_6_1.py")

if __name__ == "__main__":
    main()
