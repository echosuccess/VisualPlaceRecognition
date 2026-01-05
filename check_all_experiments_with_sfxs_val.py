#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计所有Image Matching实验（包含sfxs_val）
在Colab中运行，检查最新状态
"""

from pathlib import Path
import json
from collections import defaultdict
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 定义所有应该运行的实验
MATCHERS = ['superglue', 'loftr', 'superpoint-lg']
VPR_METHODS = ['cosplace', 'netvlad', 'mixvpr', 'megaloc']

# Test数据集（dot_product）
TEST_DATASETS = ['sf_xs_test', 'tokyo_xs_test', 'svox_sun_test', 'svox_night_test']
TEST_DISTANCE = 'dot_product'

# Val数据集（l2，用于Extension 6.1）
VAL_DATASETS = ['sfxs_val']
VAL_DISTANCE = 'l2'

def check_result_exists(matcher, vpr_method, dataset, distance):
    """检查结果文件是否存在且完整"""
    result_file = Path(f"results/image_matching/{matcher}_{vpr_method}_{distance}_{dataset}.json")
    if not result_file.exists():
        return False, 'NOT_EXISTS', 0
    
    size_mb = result_file.stat().st_size / (1024*1024)
    filename = result_file.name
    
    # 检查文件大小是否合理
    expected_sizes = {
        'svox_sun_test': 60,  # MB
        'svox_night_test': 3.5,
        'sf_xs_test': 5.0,
        'tokyo_xs_test': 1.5,
        'sfxs_val': 40,  # 大约和sf_xs_test类似
    }
    
    expected_size = expected_sizes.get(dataset, 1.0)
    if size_mb < expected_size * 0.1:  # 小于预期的10%认为不完整
        return False, 'INCOMPLETE', size_mb
    
    return True, 'OK', size_mb

def check_vpr_log_exists(vpr_method, dataset, distance):
    """检查VPR日志是否存在"""
    vpr_exp_name = f"{vpr_method}_{distance}_{dataset}"
    vpr_log_base = Path(f"logs/baseline/{vpr_exp_name}")
    
    if not vpr_log_base.exists():
        return False
    
    timestamp_dirs = sorted([d for d in vpr_log_base.iterdir() if d.is_dir()])
    if not timestamp_dirs:
        return False
    
    latest_dir = timestamp_dirs[-1]
    z_data_path = latest_dir / "z_data.torch"
    return z_data_path.exists()

def main():
    print("="*80)
    print("统计所有Image Matching实验（包含sfxs_val）")
    print("="*80)
    
    results_dir = Path("results/image_matching")
    if not results_dir.exists():
        print("\n[ERROR] results/image_matching 目录不存在")
        print("请确保在Colab中运行，并且结果文件在正确的位置")
        return
    
    # 统计所有实验
    all_experiments = []
    completed = []
    missing = []
    missing_vpr = []
    
    # Test数据集
    for matcher in MATCHERS:
        for vpr_method in VPR_METHODS:
            for dataset in TEST_DATASETS:
                exp_key = f"{matcher}_{vpr_method}_{TEST_DISTANCE}_{dataset}"
                all_experiments.append({
                    'matcher': matcher,
                    'vpr_method': vpr_method,
                    'dataset': dataset,
                    'distance': TEST_DISTANCE,
                    'exp_key': exp_key
                })
                
                # 检查VPR日志是否存在
                if not check_vpr_log_exists(vpr_method, dataset, TEST_DISTANCE):
                    missing_vpr.append(exp_key)
                    continue
                
                # 检查结果文件
                exists, status, size_mb = check_result_exists(matcher, vpr_method, dataset, TEST_DISTANCE)
                
                if exists:
                    completed.append(exp_key)
                else:
                    missing.append({
                        'exp_key': exp_key,
                        'matcher': matcher,
                        'vpr_method': vpr_method,
                        'dataset': dataset,
                        'distance': TEST_DISTANCE,
                        'status': status,
                        'size_mb': size_mb
                    })
    
    # Val数据集（sfxs_val）
    for matcher in MATCHERS:
        for vpr_method in VPR_METHODS:
            for dataset in VAL_DATASETS:
                exp_key = f"{matcher}_{vpr_method}_{VAL_DISTANCE}_{dataset}"
                all_experiments.append({
                    'matcher': matcher,
                    'vpr_method': vpr_method,
                    'dataset': dataset,
                    'distance': VAL_DISTANCE,
                    'exp_key': exp_key
                })
                
                # 检查VPR日志是否存在
                if not check_vpr_log_exists(vpr_method, dataset, VAL_DISTANCE):
                    missing_vpr.append(exp_key)
                    continue
                
                # 检查结果文件
                exists, status, size_mb = check_result_exists(matcher, vpr_method, dataset, VAL_DISTANCE)
                
                if exists:
                    completed.append(exp_key)
                else:
                    missing.append({
                        'exp_key': exp_key,
                        'matcher': matcher,
                        'vpr_method': vpr_method,
                        'dataset': dataset,
                        'distance': VAL_DISTANCE,
                        'status': status,
                        'size_mb': size_mb
                    })
    
    # 输出统计
    print(f"\n总计实验数: {len(all_experiments)}")
    print(f"  已完成: {len(completed)}")
    print(f"  未完成: {len(missing)}")
    print(f"  缺少VPR日志: {len(missing_vpr)}")
    
    # 按数据集详细统计
    print("\n" + "="*80)
    print("按数据集统计")
    print("="*80)
    
    # Test数据集
    for dataset in TEST_DATASETS:
        dataset_completed = [k for k in completed if k.endswith(f"_{dataset}")]
        dataset_missing = [exp for exp in missing if exp['dataset'] == dataset]
        dataset_missing_vpr = [k for k in missing_vpr if k.endswith(f"_{dataset}")]
        
        total = len(MATCHERS) * len(VPR_METHODS)
        print(f"\n[{dataset}] (dot_product):")
        print(f"  已完成: {len(dataset_completed)}/{total}")
        print(f"  未完成: {len(dataset_missing)}")
        print(f"  缺少VPR: {len(dataset_missing_vpr)}")
        
        if dataset_missing:
            print(f"  未完成的实验:")
            for exp in dataset_missing:
                print(f"    - {exp['matcher']} + {exp['vpr_method']} ({exp['status']})")
    
    # Val数据集
    for dataset in VAL_DATASETS:
        dataset_completed = [k for k in completed if k.endswith(f"_{dataset}")]
        dataset_missing = [exp for exp in missing if exp['dataset'] == dataset]
        dataset_missing_vpr = [k for k in missing_vpr if k.endswith(f"_{dataset}")]
        
        total = len(MATCHERS) * len(VPR_METHODS)
        print(f"\n[{dataset}] (l2, Extension 6.1验证集):")
        print(f"  已完成: {len(dataset_completed)}/{total}")
        print(f"  未完成: {len(dataset_missing)}")
        print(f"  缺少VPR: {len(dataset_missing_vpr)}")
        
        if dataset_missing:
            print(f"  未完成的实验:")
            for exp in dataset_missing:
                print(f"    - {exp['matcher']} + {exp['vpr_method']} ({exp['status']})")
        
        if dataset_missing_vpr:
            print(f"  缺少VPR的实验:")
            for k in dataset_missing_vpr:
                parts = k.split('_')
                if len(parts) >= 3:
                    vpr_method = parts[1]
                    print(f"    - {vpr_method}")
    
    # 详细列出未完成的实验
    if missing:
        print("\n" + "="*80)
        print("未完成的实验（需要运行）")
        print("="*80)
        
        # 按数据集分组
        by_dataset = defaultdict(list)
        for exp in missing:
            by_dataset[exp['dataset']].append(exp)
        
        for dataset, exps in sorted(by_dataset.items()):
            print(f"\n[{dataset}]: {len(exps)} 个实验")
            for exp in exps:
                print(f"  - {exp['matcher']} + {exp['vpr_method']} ({exp['status']})")
    
    # 生成运行命令
    if missing:
        print("\n" + "="*80)
        print("生成的运行命令")
        print("="*80)
        
        # 按数据集分组
        by_dataset_run = defaultdict(list)
        for exp in missing:
            dataset = exp['dataset']
            by_dataset_run[dataset].append(exp)
        
        cell_num = 1
        for dataset, exps in sorted(by_dataset_run.items()):
            print(f"\n# ===== {dataset} ({len(exps)} 个实验) =====")
            for exp in exps:
                cmd = f"!python run_image_matching_baseline.py --matcher {exp['matcher']} --vpr_method {exp['vpr_method']} --dataset {exp['dataset']} --distance {exp['distance']} --device cuda"
                print(f"\n# Cell {cell_num}: {exp['matcher']} + {exp['vpr_method']}")
                print(cmd)
                cell_num += 1

if __name__ == "__main__":
    main()
