#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计所有Image Matching实验，找出未完成的
"""

from pathlib import Path
import json
from collections import defaultdict

# 定义所有应该运行的实验
MATCHERS = ['superglue', 'loftr', 'superpoint-lg']
VPR_METHODS = ['cosplace', 'netvlad', 'mixvpr', 'megaloc']
DATASETS = ['sf_xs_test', 'tokyo_xs_test', 'svox_sun_test', 'svox_night_test']
DISTANCES = ['dot_product']  # 只运行dot_product（根据output.txt）

# 异常文件列表（需要重新运行）
ABNORMAL_FILES = [
    'superpoint-lg_mixvpr_dot_product_tokyo_xs_test.json',  # 必须重新运行（不完整）
    'loftr_megaloc_dot_product_tokyo_xs_test.json',  # 可选重新运行
    'superglue_cosplace_dot_product_tokyo_xs_test.json',  # 可选重新运行
    'superglue_megaloc_dot_product_tokyo_xs_test.json',  # 可选重新运行
    'superglue_mixvpr_dot_product_tokyo_xs_test.json',  # 可选重新运行
    'superpoint-lg_cosplace_dot_product_tokyo_xs_test.json',  # 可选重新运行
    'superpoint-lg_megaloc_dot_product_tokyo_xs_test.json',  # 可选重新运行
]

def check_result_exists(matcher, vpr_method, distance, dataset):
    """检查结果文件是否存在"""
    result_file = Path(f"results/image_matching/{matcher}_{vpr_method}_{distance}_{dataset}.json")
    return result_file.exists()

def check_vpr_log_exists(vpr_method, distance, dataset):
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
    print("统计所有Image Matching实验")
    print("="*80)
    
    results_dir = Path("results/image_matching")
    
    # 统计所有实验
    all_experiments = []
    completed = []
    missing = []
    missing_vpr = []
    abnormal = []
    
    for matcher in MATCHERS:
        for vpr_method in VPR_METHODS:
            for dataset in DATASETS:
                for distance in DISTANCES:
                    exp_key = f"{matcher}_{vpr_method}_{distance}_{dataset}"
                    all_experiments.append({
                        'matcher': matcher,
                        'vpr_method': vpr_method,
                        'dataset': dataset,
                        'distance': distance,
                        'exp_key': exp_key
                    })
                    
                    # 检查VPR日志是否存在
                    if not check_vpr_log_exists(vpr_method, distance, dataset):
                        missing_vpr.append(exp_key)
                        continue
                    
                    # 检查结果文件是否存在
                    if check_result_exists(matcher, vpr_method, distance, dataset):
                        result_file = Path(f"results/image_matching/{matcher}_{vpr_method}_{distance}_{dataset}.json")
                        filename = result_file.name
                        
                        if filename in ABNORMAL_FILES:
                            abnormal.append({
                                'exp_key': exp_key,
                                'matcher': matcher,
                                'vpr_method': vpr_method,
                                'dataset': dataset,
                                'distance': distance,
                                'filename': filename,
                                'must_rerun': filename == 'superpoint-lg_mixvpr_dot_product_tokyo_xs_test.json'
                            })
                        else:
                            completed.append(exp_key)
                    else:
                        missing.append({
                            'exp_key': exp_key,
                            'matcher': matcher,
                            'vpr_method': vpr_method,
                            'dataset': dataset,
                            'distance': distance
                        })
    
    # 输出统计
    print(f"\n总计实验数: {len(all_experiments)}")
    print(f"  已完成: {len(completed)}")
    print(f"  未完成: {len(missing)}")
    print(f"  异常（需重新运行）: {len(abnormal)}")
    print(f"  缺少VPR日志: {len(missing_vpr)}")
    
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
                print(f"  - {exp['matcher']} + {exp['vpr_method']}")
    
    # 列出异常文件
    must_rerun = []
    optional_rerun = []
    
    if abnormal:
        print("\n" + "="*80)
        print("异常文件（需要重新运行）")
        print("="*80)
        
        must_rerun = [e for e in abnormal if e['must_rerun']]
        optional_rerun = [e for e in abnormal if not e['must_rerun']]
        
        if must_rerun:
            print(f"\n[必须重新运行] ({len(must_rerun)} 个):")
            for exp in must_rerun:
                print(f"  - {exp['matcher']} + {exp['vpr_method']} + {exp['dataset']}")
                print(f"    原因: 文件不完整")
        
        if optional_rerun:
            print(f"\n[可选重新运行] ({len(optional_rerun)} 个):")
            for exp in optional_rerun:
                print(f"  - {exp['matcher']} + {exp['vpr_method']} + {exp['dataset']}")
                print(f"    原因: 比率异常（可能是数据集本身的问题）")
    
    # 列出缺少VPR日志的实验
    if missing_vpr:
        print("\n" + "="*80)
        print("缺少VPR日志的实验（需要先运行VPR）")
        print("="*80)
        
        by_vpr = defaultdict(list)
        for exp_key in missing_vpr:
            parts = exp_key.split('_')
            if len(parts) >= 3:
                vpr_method = parts[1]
                by_vpr[vpr_method].append(exp_key)
        
        for vpr_method, exp_keys in sorted(by_vpr.items()):
            print(f"\n[{vpr_method}]: {len(exp_keys)} 个实验缺少VPR日志")
            # 只显示前5个
            for exp_key in exp_keys[:5]:
                print(f"  - {exp_key}")
            if len(exp_keys) > 5:
                print(f"  ... 还有 {len(exp_keys) - 5} 个")
    
    # 生成运行命令
    print("\n" + "="*80)
    print("生成的运行命令")
    print("="*80)
    
    # 必须重新运行的
    if must_rerun:
        print("\n[必须重新运行]")
        for exp in must_rerun:
            cmd = f"!python run_image_matching_baseline.py --matcher {exp['matcher']} --vpr_method {exp['vpr_method']} --dataset {exp['dataset']} --distance {exp['distance']} --device cuda"
            print(cmd)
    
    # 未完成的实验
    if missing:
        print("\n[未完成的实验]")
        # 按数据集分组，方便并行
        by_dataset = defaultdict(list)
        for exp in missing:
            by_dataset[exp['dataset']].append(exp)
        
        for dataset, exps in sorted(by_dataset.items()):
            print(f"\n# {dataset} ({len(exps)} 个实验)")
            for exp in exps:
                cmd = f"!python run_image_matching_baseline.py --matcher {exp['matcher']} --vpr_method {exp['vpr_method']} --dataset {exp['dataset']} --distance {exp['distance']} --device cuda"
                print(cmd)
    
    # 可选重新运行的
    if optional_rerun:
        print("\n[可选重新运行]")
        for exp in optional_rerun:
            cmd = f"!python run_image_matching_baseline.py --matcher {exp['matcher']} --vpr_method {exp['vpr_method']} --dataset {exp['dataset']} --distance {exp['distance']} --device cuda"
            print(cmd)
    
    # 并行运行建议
    print("\n" + "="*80)
    print("并行运行建议")
    print("="*80)
    
    total_to_run = len(missing) + len(abnormal)
    print(f"\n总计需要运行: {total_to_run} 个实验")
    
    if total_to_run > 0:
        print("\n并行策略:")
        print("1. 使用多个Colab Cell，每个Cell运行一个命令")
        print("2. Colab免费版：可能无法真正并行，会排队执行")
        print("3. Colab Pro：可以尝试同时运行2-3个Cell")
        print("4. 建议：按数据集分组，同一数据集的不同matcher可以并行")
        
        # 按数据集统计
        all_to_run = missing + abnormal
        by_dataset_run = defaultdict(list)
        for exp in all_to_run:
            dataset = exp['dataset']
            by_dataset_run[dataset].append(exp)
        
        print("\n按数据集分组（可以并行运行）:")
        for dataset, exps in sorted(by_dataset_run.items()):
            print(f"\n  {dataset}: {len(exps)} 个实验")
            print(f"    可以创建 {len(exps)} 个Cell并行运行")

if __name__ == "__main__":
    main()
