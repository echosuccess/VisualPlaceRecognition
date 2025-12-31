#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量运行所有Image Matching实验
对每个VPR方法的结果运行3种Image Matching方法
"""

import subprocess
import time
from pathlib import Path
import json


# 实验配置
VPR_METHODS = ['cosplace', 'netvlad', 'mixvpr', 'megaloc']
DATASETS = ['sf_xs_test', 'tokyo_xs_test', 'svox_sun_test', 'svox_night_test']
DISTANCES = ['dot_product']  # 因为L2和Dot Product结果相同，只需要测试一个
IMAGE_MATCHERS = ['superglue', 'loftr', 'superpoint-lg']

# 可以根据需要选择子集
# 例如，只测试最好的VPR方法：
# VPR_METHODS = ['megaloc', 'mixvpr']


def check_vpr_log_exists(vpr_method, distance, dataset):
    """检查VPR实验日志是否存在"""
    vpr_exp_name = f"{vpr_method}_{distance}_{dataset}"
    vpr_log_base = Path("logs/baseline") / vpr_exp_name
    
    if not vpr_log_base.exists():
        return False
    
    timestamp_dirs = sorted([d for d in vpr_log_base.iterdir() if d.is_dir()])
    if not timestamp_dirs:
        return False
    
    # 检查z_data.torch是否存在
    latest_dir = timestamp_dirs[-1]
    data_file = latest_dir / "z_data.torch"
    
    return data_file.exists()


def run_image_matching_experiment(matcher, vpr_method, dataset, distance='dot_product', top_k=20, device='cuda'):
    """运行单个Image Matching实验"""
    
    # 检查VPR日志是否存在
    if not check_vpr_log_exists(vpr_method, distance, dataset):
        print(f"  [SKIP] VPR log not found: {vpr_method}_{distance}_{dataset}")
        return False
    
    # 检查结果是否已存在
    output_path = Path(f"results/image_matching/{matcher}_{vpr_method}_{distance}_{dataset}.json")
    if output_path.exists():
        print(f"  [SKIP] Results already exist: {output_path.name}")
        return True
    
    # 运行实验
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
        print(f"  Error: {e}")
        return False


def generate_experiment_plan():
    """生成实验计划"""
    experiments = []
    
    for vpr_method in VPR_METHODS:
        for dataset in DATASETS:
            for distance in DISTANCES:
                for matcher in IMAGE_MATCHERS:
                    # 检查VPR日志是否存在
                    if check_vpr_log_exists(vpr_method, distance, dataset):
                        experiments.append({
                            'matcher': matcher,
                            'vpr_method': vpr_method,
                            'dataset': dataset,
                            'distance': distance
                        })
    
    return experiments


def print_experiment_plan(experiments):
    """打印实验计划"""
    print(f"\n{'='*80}")
    print("Experiment Plan")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiments)}")
    print(f"\nBreakdown:")
    print(f"  VPR methods: {len(VPR_METHODS)}")
    print(f"  Datasets: {len(DATASETS)}")
    print(f"  Image matchers: {len(IMAGE_MATCHERS)}")
    print(f"  Expected total: {len(VPR_METHODS)} x {len(DATASETS)} x {len(IMAGE_MATCHERS)} = {len(VPR_METHODS) * len(DATASETS) * len(IMAGE_MATCHERS)}")
    print(f"  Actual (with existing VPR logs): {len(experiments)}")
    
    # 按matcher分组统计
    matcher_counts = {}
    for exp in experiments:
        matcher = exp['matcher']
        matcher_counts[matcher] = matcher_counts.get(matcher, 0) + 1
    
    print(f"\nExperiments per matcher:")
    for matcher, count in matcher_counts.items():
        print(f"  {matcher}: {count} experiments")
    
    print(f"{'='*80}\n")


def estimate_time(num_experiments):
    """估计所需时间"""
    # 假设每个实验平均需要的时间
    # - SuperGlue: ~30秒/query, 对于100个query + top20预测 = ~1小时
    # - LoFTR: ~1分钟/query = ~2小时
    # - SuperPoint-LG: ~20秒/query = ~40分钟
    
    avg_time_per_exp = {
        'superglue': 60,      # 1小时 = 60分钟
        'loftr': 120,         # 2小时
        'superpoint-lg': 40   # 40分钟
    }
    
    total_minutes = 0
    for exp in generate_experiment_plan():
        matcher = exp['matcher']
        total_minutes += avg_time_per_exp.get(matcher, 60)
    
    hours = total_minutes / 60
    print(f"\n[ESTIMATE] Total time: ~{total_minutes} minutes (~{hours:.1f} hours)")
    print(f"[TIP] You can run matchers in parallel to reduce time!")
    print(f"[TIP] Or test on a subset first (e.g., only MegaLoc)\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run all Image Matching experiments')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--top_k', type=int, default=20, help='Match top-K predictions per query')
    parser.add_argument('--quick_test', action='store_true', 
                        help='Quick test: only run on MegaLoc + SF-XS')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("Batch Image Matching Experiments")
    print(f"{'='*80}\n")
    
    # 快速测试模式
    if args.quick_test:
        print("[MODE] Quick test mode: only MegaLoc + SF-XS")
        global VPR_METHODS, DATASETS
        VPR_METHODS = ['megaloc']
        DATASETS = ['sf_xs_test']
    
    # 生成实验计划
    experiments = generate_experiment_plan()
    
    if not experiments:
        print("[ERROR] No experiments to run! Check if VPR logs exist.")
        return
    
    # 打印计划
    print_experiment_plan(experiments)
    
    # 估计时间
    estimate_time(len(experiments))
    
    # 确认
    response = input("Start experiments? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # 运行实验
    start_time = time.time()
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n[Progress] Experiment {i}/{len(experiments)}")
        
        # 检查是否已完成
        output_path = Path(f"results/image_matching/{exp['matcher']}_{exp['vpr_method']}_{exp['distance']}_{exp['dataset']}.json")
        if output_path.exists():
            print(f"  [SKIP] Already exists: {output_path.name}")
            skip_count += 1
            continue
        
        # 运行
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
    
    # 总结
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("Batch Experiments Completed!")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Skipped: {skip_count}")
    print(f"Time elapsed: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)")
    print(f"{'='*80}\n")
    
    # 保存实验摘要
    summary = {
        'total': len(experiments),
        'success': success_count,
        'fail': fail_count,
        'skip': skip_count,
        'time_minutes': elapsed_time / 60,
        'experiments': experiments
    }
    
    summary_path = Path("results/image_matching/batch_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[SAVED] Summary saved to: {summary_path}\n")


if __name__ == "__main__":
    main()
