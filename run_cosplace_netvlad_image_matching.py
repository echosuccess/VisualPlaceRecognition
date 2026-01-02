#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行 CosPlace 和 NetVLAD 的 Image Matching 实验
"""

import subprocess
import time
from pathlib import Path
import json

# 只运行 CosPlace 和 NetVLAD
VPR_METHODS = ['cosplace', 'netvlad']
DATASETS = ['sf_xs_test', 'tokyo_xs_test', 'svox_sun_test', 'svox_night_test']
DISTANCES = ['dot_product']
IMAGE_MATCHERS = ['superglue', 'loftr', 'superpoint-lg']


def check_vpr_log_exists(vpr_method, distance, dataset):
    """检查VPR实验日志是否存在"""
    vpr_exp_name = f"{vpr_method}_{distance}_{dataset}"
    vpr_log_base = Path("logs/baseline") / vpr_exp_name
    
    if not vpr_log_base.exists():
        return False
    
    timestamp_dirs = sorted([d for d in vpr_log_base.iterdir() if d.is_dir()])
    if not timestamp_dirs:
        return False
    
    latest_dir = timestamp_dirs[-1]
    data_file = latest_dir / "z_data.torch"
    return data_file.exists()


def run_image_matching_experiment(matcher, vpr_method, dataset, distance='dot_product', top_k=20, device='cuda'):
    """运行单个Image Matching实验"""
    
    if not check_vpr_log_exists(vpr_method, distance, dataset):
        print(f"  [SKIP] VPR log not found: {vpr_method}_{distance}_{dataset}")
        return False
    
    output_path = Path(f"results/image_matching/{matcher}_{vpr_method}_{distance}_{dataset}.json")
    if output_path.exists():
        print(f"  [SKIP] Results already exist: {output_path.name}")
        return True
    
    cmd = [
        'python', 'run_image_matching_baseline.py',
        '--matcher', matcher,
        '--vpr_method', vpr_method,
        '--dataset', dataset,
        '--distance', distance,
        '--top_k', str(top_k),
        '--device', device
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {matcher} on {vpr_method} {dataset}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"  [OK] Completed: {matcher} on {vpr_method} {dataset}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] Failed: {matcher} on {vpr_method} {dataset}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run Image Matching for CosPlace and NetVLAD')
    parser.add_argument('--top_k', type=int, default=20, help='Top-K predictions to match')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--dataset', type=str, default=None, help='Specific dataset (optional)')
    parser.add_argument('--matcher', type=str, default=None, help='Specific matcher (optional)')
    
    args = parser.parse_args()
    
    # 生成实验列表
    experiments = []
    for vpr_method in VPR_METHODS:
        for dataset in DATASETS:
            if args.dataset and dataset != args.dataset:
                continue
            for distance in DISTANCES:
                for matcher in IMAGE_MATCHERS:
                    if args.matcher and matcher != args.matcher:
                        continue
                    if check_vpr_log_exists(vpr_method, distance, dataset):
                        experiments.append({
                            'matcher': matcher,
                            'vpr_method': vpr_method,
                            'dataset': dataset,
                            'distance': distance
                        })
    
    print(f"\n{'='*80}")
    print(f"CosPlace & NetVLAD Image Matching Experiments")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiments)}")
    print(f"VPR methods: {VPR_METHODS}")
    print(f"Datasets: {DATASETS}")
    print(f"Matchers: {IMAGE_MATCHERS}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n[Progress] Experiment {i}/{len(experiments)}")
        
        output_path = Path(f"results/image_matching/{exp['matcher']}_{exp['vpr_method']}_{exp['distance']}_{exp['dataset']}.json")
        if output_path.exists():
            print(f"  [SKIP] Already exists: {output_path.name}")
            skip_count += 1
            continue
        
        success = run_image_matching_experiment(
            matcher=exp['matcher'],
            vpr_method=exp['vpr_method'],
            dataset=exp['dataset'],
            distance=exp['distance'],
            top_k=args.top_k,
            device=args.device
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("Experiments Completed!")
    print(f"{'='*80}")
    print(f"Total: {len(experiments)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Skipped: {skip_count}")
    print(f"Time: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
