#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在Colab中统计所有Image Matching实验，找出未完成的
"""

from pathlib import Path
import json

# 定义所有应该运行的实验
MATCHERS = ['superglue', 'loftr', 'superpoint-lg']
VPR_METHODS = ['cosplace', 'netvlad', 'mixvpr', 'megaloc']
DATASETS = ['sf_xs_test', 'tokyo_xs_test', 'svox_sun_test', 'svox_night_test']
DISTANCE = 'dot_product'

# 异常文件列表（需要重新运行）
ABNORMAL_FILES = {
    'superpoint-lg_mixvpr_dot_product_tokyo_xs_test.json': 'MUST',  # 必须重新运行（不完整）
    'loftr_megaloc_dot_product_tokyo_xs_test.json': 'OPTIONAL',
    'superglue_cosplace_dot_product_tokyo_xs_test.json': 'OPTIONAL',
    'superglue_megaloc_dot_product_tokyo_xs_test.json': 'OPTIONAL',
    'superglue_mixvpr_dot_product_tokyo_xs_test.json': 'OPTIONAL',
    'superpoint-lg_cosplace_dot_product_tokyo_xs_test.json': 'OPTIONAL',
    'superpoint-lg_megaloc_dot_product_tokyo_xs_test.json': 'OPTIONAL',
}

def check_result_exists(matcher, vpr_method, dataset):
    """检查结果文件是否存在且完整"""
    result_file = Path(f"results/image_matching/{matcher}_{vpr_method}_{DISTANCE}_{dataset}.json")
    if not result_file.exists():
        return False, 'NOT_EXISTS'
    
    size_mb = result_file.stat().st_size / (1024*1024)
    filename = result_file.name
    
    # 检查是否是异常文件
    if filename in ABNORMAL_FILES:
        if ABNORMAL_FILES[filename] == 'MUST':
            return False, 'ABNORMAL_INCOMPLETE'
        else:
            return True, 'ABNORMAL_RATIO'  # 文件存在但比率异常
    
    # 检查文件大小是否合理
    expected_sizes = {
        'svox_sun_test': 60,  # MB
        'svox_night_test': 3.5,
        'sf_xs_test': 5.0,
        'tokyo_xs_test': 1.5,
    }
    
    expected_size = expected_sizes.get(dataset, 1.0)
    if size_mb < expected_size * 0.1:  # 小于预期的10%认为不完整
        return False, 'INCOMPLETE'
    
    return True, 'OK'

def check_vpr_log_exists(vpr_method, dataset):
    """检查VPR日志是否存在"""
    vpr_exp_name = f"{vpr_method}_{DISTANCE}_{dataset}"
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
    if not results_dir.exists():
        print("\n[ERROR] results/image_matching 目录不存在")
        print("请确保在Colab中运行，并且结果文件在正确的位置")
        return
    
    # 统计所有实验
    all_experiments = []
    completed = []
    missing = []
    missing_vpr = []
    abnormal_must = []
    abnormal_optional = []
    
    for matcher in MATCHERS:
        for vpr_method in VPR_METHODS:
            for dataset in DATASETS:
                exp_key = f"{matcher}_{vpr_method}_{DISTANCE}_{dataset}"
                all_experiments.append({
                    'matcher': matcher,
                    'vpr_method': vpr_method,
                    'dataset': dataset,
                    'exp_key': exp_key
                })
                
                # 检查VPR日志是否存在
                if not check_vpr_log_exists(vpr_method, dataset):
                    missing_vpr.append(exp_key)
                    continue
                
                # 检查结果文件
                exists, status = check_result_exists(matcher, vpr_method, dataset)
                
                if status == 'ABNORMAL_INCOMPLETE':
                    abnormal_must.append({
                        'exp_key': exp_key,
                        'matcher': matcher,
                        'vpr_method': vpr_method,
                        'dataset': dataset
                    })
                elif status == 'ABNORMAL_RATIO':
                    abnormal_optional.append({
                        'exp_key': exp_key,
                        'matcher': matcher,
                        'vpr_method': vpr_method,
                        'dataset': dataset
                    })
                elif exists:
                    completed.append(exp_key)
                else:
                    missing.append({
                        'exp_key': exp_key,
                        'matcher': matcher,
                        'vpr_method': vpr_method,
                        'dataset': dataset,
                        'status': status
                    })
    
    # 输出统计
    print(f"\n总计实验数: {len(all_experiments)}")
    print(f"  已完成: {len(completed)}")
    print(f"  未完成: {len(missing)}")
    print(f"  必须重新运行（异常）: {len(abnormal_must)}")
    print(f"  可选重新运行（异常）: {len(abnormal_optional)}")
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
    if abnormal_must or abnormal_optional:
        print("\n" + "="*80)
        print("异常文件（需要重新运行）")
        print("="*80)
        
        if abnormal_must:
            print(f"\n[必须重新运行] ({len(abnormal_must)} 个):")
            for exp in abnormal_must:
                print(f"  - {exp['matcher']} + {exp['vpr_method']} + {exp['dataset']}")
                print(f"    原因: 文件不完整")
        
        if abnormal_optional:
            print(f"\n[可选重新运行] ({len(abnormal_optional)} 个):")
            for exp in abnormal_optional:
                print(f"  - {exp['matcher']} + {exp['vpr_method']} + {exp['dataset']}")
                print(f"    原因: 比率异常（可能是数据集本身的问题）")
    
    # 生成运行命令
    print("\n" + "="*80)
    print("生成的运行命令（按数据集分组，便于并行）")
    print("="*80)
    
    # 合并所有需要运行的实验
    all_to_run = missing + abnormal_must + abnormal_optional
    
    if all_to_run:
        # 按数据集分组
        by_dataset_run = defaultdict(list)
        for exp in all_to_run:
            dataset = exp['dataset']
            by_dataset_run[dataset].append(exp)
        
        cell_num = 1
        for dataset, exps in sorted(by_dataset_run.items()):
            print(f"\n# ===== {dataset} ({len(exps)} 个实验) =====")
            for exp in exps:
                cmd = f"!python run_image_matching_baseline.py --matcher {exp['matcher']} --vpr_method {exp['vpr_method']} --dataset {exp['dataset']} --distance {DISTANCE} --device cuda"
                print(f"\n# Cell {cell_num}: {exp['matcher']} + {exp['vpr_method']}")
                print(cmd)
                cell_num += 1
    
    # 并行运行建议
    print("\n" + "="*80)
    print("并行运行建议")
    print("="*80)
    
    total_to_run = len(all_to_run)
    print(f"\n总计需要运行: {total_to_run} 个实验")
    
    if total_to_run > 0:
        print("\n并行策略:")
        print("1. 使用多个Colab Cell，每个Cell运行一个命令")
        print("2. 同一数据集的不同matcher可以并行运行（新代码有独立checkpoint）")
        print("3. Colab免费版：可能无法真正并行，会排队执行")
        print("4. Colab Pro：可以尝试同时运行2-3个Cell")
        
        # 按数据集统计
        by_dataset_run = defaultdict(list)
        for exp in all_to_run:
            dataset = exp['dataset']
            by_dataset_run[dataset].append(exp)
        
        print("\n按数据集分组（可以并行运行）:")
        for dataset, exps in sorted(by_dataset_run.items()):
            print(f"\n  {dataset}: {len(exps)} 个实验")
            print(f"    建议：创建 {len(exps)} 个Cell，可以同时运行（注意GPU资源）")

if __name__ == "__main__":
    from collections import defaultdict
    main()
