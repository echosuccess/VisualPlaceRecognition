#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行SF-XS val的Image Matching实验（Extension 6.1验证集）
"""

import subprocess
import sys
from pathlib import Path

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
    
    print(f"\n{'='*80}")
    print(f"运行: {matcher} + {vpr_method} + {dataset}")
    print(f"{'='*80}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ {matcher} + {vpr_method} + {dataset} 完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {matcher} + {vpr_method} + {dataset} 失败: {e}")
        return False

def main():
    """主函数"""
    
    print("="*80)
    print("SF-XS val Image Matching实验（Extension 6.1验证集）")
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
    
    print(f"\n将运行 {len(experiments)} 个Image Matching实验")
    
    # 检查哪些已经完成
    print("\n检查已完成实验:")
    completed = []
    remaining = []
    
    for exp in experiments:
        result_file = Path(f"results/image_matching/{exp['matcher']}_{exp['vpr_method']}_l2_{exp['dataset']}.json")
        result_file_pkl = Path(f"results/image_matching/{exp['matcher']}_{exp['vpr_method']}_l2_{exp['dataset']}.pkl")
        
        if result_file.exists() or result_file_pkl.exists():
            completed.append(f"{exp['matcher']}_{exp['vpr_method']}")
            print(f"  ✅ {exp['matcher']} + {exp['vpr_method']}")
        else:
            remaining.append(exp)
    
    if completed:
        print(f"\n已完成: {len(completed)}/{len(experiments)}")
    
    if not remaining:
        print("\n✅ 所有实验已完成！")
        return
    
    print(f"\n剩余: {len(remaining)}/{len(experiments)}")
    print("\n开始运行剩余实验...")
    
    # 运行剩余实验
    for i, exp in enumerate(remaining, 1):
        print(f"\n进度: {i}/{len(remaining)}")
        success = run_im_experiment(
            exp['matcher'],
            exp['vpr_method'],
            exp['dataset'],
            exp['distance']
        )
        
        if not success:
            print(f"\n⚠️  实验失败，但继续运行下一个...")
    
    print("\n" + "="*80)
    print("所有实验完成！")
    print("="*80)
    print("\n下一步：运行Extension 6.1")
    print("命令: python run_extension_6_1.py")

if __name__ == "__main__":
    main()
