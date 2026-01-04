#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整统计所有Image Matching实验（包括sfxs_val）
"""

from pathlib import Path
from collections import defaultdict

# 定义所有应该运行的实验
MATCHERS = ['superglue', 'loftr', 'superpoint-lg']
VPR_METHODS = ['cosplace', 'netvlad', 'mixvpr', 'megaloc']

# Test数据集（dot_product）
TEST_DATASETS = ['sf_xs_test', 'tokyo_xs_test', 'svox_sun_test', 'svox_night_test']
TEST_DISTANCE = 'dot_product'

# Val数据集（l2，用于Extension 6.1）
VAL_DATASETS = ['sfxs_val']
VAL_DISTANCE = 'l2'

# 异常文件列表
ABNORMAL_FILES = {
    'superpoint-lg_mixvpr_dot_product_tokyo_xs_test.json': 'MUST',
    'loftr_megaloc_dot_product_tokyo_xs_test.json': 'OPTIONAL',
    'superglue_cosplace_dot_product_tokyo_xs_test.json': 'OPTIONAL',
    'superglue_megaloc_dot_product_tokyo_xs_test.json': 'OPTIONAL',
    'superglue_mixvpr_dot_product_tokyo_xs_test.json': 'OPTIONAL',
    'superpoint-lg_cosplace_dot_product_tokyo_xs_test.json': 'OPTIONAL',
    'superpoint-lg_megaloc_dot_product_tokyo_xs_test.json': 'OPTIONAL',
}

def check_result_exists(matcher, vpr_method, distance, dataset):
    """检查结果文件是否存在且完整"""
    result_file = Path(f"results/image_matching/{matcher}_{vpr_method}_{distance}_{dataset}.json")
    if not result_file.exists():
        return False, 'NOT_EXISTS'
    
    size_mb = result_file.stat().st_size / (1024*1024)
    filename = result_file.name
    
    # 检查是否是异常文件
    if filename in ABNORMAL_FILES:
        if ABNORMAL_FILES[filename] == 'MUST':
            return False, 'ABNORMAL_INCOMPLETE'
        else:
            return True, 'ABNORMAL_RATIO'
    
    # 检查文件大小是否合理
    expected_sizes = {
        'svox_sun_test': 60,
        'svox_night_test': 3.5,
        'sf_xs_test': 5.0,
        'tokyo_xs_test': 1.5,
        'sfxs_val': 40,  # 根据已有文件推断
    }
    
    expected_size = expected_sizes.get(dataset, 1.0)
    if size_mb < expected_size * 0.1:
        return False, 'INCOMPLETE'
    
    return True, 'OK'

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
    print("完整统计所有Image Matching实验（包括sfxs_val）")
    print("="*80)
    
    results_dir = Path("results/image_matching")
    if not results_dir.exists():
        print("\n[ERROR] results/image_matching 目录不存在")
        return
    
    # 统计所有实验
    all_experiments = []
    completed = []
    missing = []
    missing_vpr = []
    abnormal_must = []
    abnormal_optional = []
    
    # Test数据集实验
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
                
                if not check_vpr_log_exists(vpr_method, TEST_DISTANCE, dataset):
                    missing_vpr.append(exp_key)
                    continue
                
                exists, status = check_result_exists(matcher, vpr_method, TEST_DISTANCE, dataset)
                
                if status == 'ABNORMAL_INCOMPLETE':
                    abnormal_must.append({
                        'exp_key': exp_key,
                        'matcher': matcher,
                        'vpr_method': vpr_method,
                        'dataset': dataset,
                        'distance': TEST_DISTANCE
                    })
                elif status == 'ABNORMAL_RATIO':
                    abnormal_optional.append({
                        'exp_key': exp_key,
                        'matcher': matcher,
                        'vpr_method': vpr_method,
                        'dataset': dataset,
                        'distance': TEST_DISTANCE
                    })
                elif exists:
                    completed.append(exp_key)
                else:
                    missing.append({
                        'exp_key': exp_key,
                        'matcher': matcher,
                        'vpr_method': vpr_method,
                        'dataset': dataset,
                        'distance': TEST_DISTANCE,
                        'status': status
                    })
    
    # Val数据集实验（sfxs_val）
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
                
                if not check_vpr_log_exists(vpr_method, VAL_DISTANCE, dataset):
                    missing_vpr.append(exp_key)
                    continue
                
                exists, status = check_result_exists(matcher, vpr_method, VAL_DISTANCE, dataset)
                
                if exists:
                    completed.append(exp_key)
                else:
                    missing.append({
                        'exp_key': exp_key,
                        'matcher': matcher,
                        'vpr_method': vpr_method,
                        'dataset': dataset,
                        'distance': VAL_DISTANCE,
                        'status': status
                    })
    
    # 输出统计
    print(f"\n总计实验数: {len(all_experiments)}")
    print(f"  已完成: {len(completed)}")
    print(f"  未完成: {len(missing)}")
    print(f"  必须重新运行（异常）: {len(abnormal_must)}")
    print(f"  可选重新运行（异常）: {len(abnormal_optional)}")
    print(f"  缺少VPR日志: {len(missing_vpr)}")
    
    # 按数据集分组统计
    print("\n" + "="*80)
    print("按数据集统计")
    print("="*80)
    
    by_dataset = defaultdict(lambda: {'completed': 0, 'missing': 0, 'abnormal': 0})
    
    for exp_key in completed:
        parts = exp_key.split('_')
        if len(parts) >= 4:
            dataset = '_'.join(parts[3:])
            by_dataset[dataset]['completed'] += 1
    
    for exp in missing:
        by_dataset[exp['dataset']]['missing'] += 1
    
    for exp in abnormal_must + abnormal_optional:
        by_dataset[exp['dataset']]['abnormal'] += 1
    
    for dataset, stats in sorted(by_dataset.items()):
        total = stats['completed'] + stats['missing'] + stats['abnormal']
        print(f"\n{dataset}:")
        print(f"  已完成: {stats['completed']}/{total}")
        print(f"  未完成: {stats['missing']}/{total}")
        print(f"  异常: {stats['abnormal']}/{total}")
    
    # 详细列出未完成的实验
    if missing:
        print("\n" + "="*80)
        print("未完成的实验（需要运行）")
        print("="*80)
        
        by_dataset_missing = defaultdict(list)
        for exp in missing:
            by_dataset_missing[exp['dataset']].append(exp)
        
        for dataset, exps in sorted(by_dataset_missing.items()):
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
        
        if abnormal_optional:
            print(f"\n[可选重新运行] ({len(abnormal_optional)} 个):")
            for exp in abnormal_optional:
                print(f"  - {exp['matcher']} + {exp['vpr_method']} + {exp['dataset']}")
    
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
            # 检查是否是sfxs_val
            sfxs_val_count = sum(1 for k in exp_keys if 'sfxs_val' in k)
            if sfxs_val_count > 0:
                print(f"\n[{vpr_method}]: {sfxs_val_count} 个sfxs_val实验缺少VPR日志")
                print(f"  需要先运行: python run_sfxs_val_vpr.py")
    
    # 生成运行命令
    print("\n" + "="*80)
    print("生成的运行命令（按数据集分组，便于并行）")
    print("="*80)
    
    all_to_run = missing + abnormal_must + abnormal_optional
    
    if all_to_run:
        by_dataset_run = defaultdict(list)
        for exp in all_to_run:
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
        
        by_dataset_run = defaultdict(list)
        for exp in all_to_run:
            dataset = exp['dataset']
            by_dataset_run[dataset].append(exp)
        
        print("\n按数据集分组（可以并行运行）:")
        for dataset, exps in sorted(by_dataset_run.items()):
            print(f"\n  {dataset}: {len(exps)} 个实验")
            print(f"    建议：创建 {len(exps)} 个Cell，可以同时运行")

if __name__ == "__main__":
    main()
