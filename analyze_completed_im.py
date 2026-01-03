#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析已完成的Image Matching结果"""

import sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

folder = Path('mydoc/drive-download-20260103T202954Z-3-001')
json_files = list(folder.glob('*.json'))
pkl_files = list(folder.glob('*.pkl'))

print("="*80)
print("已完成的Image Matching结果分析")
print("="*80)
print(f"\nJSON文件: {len(json_files)} 个")
print(f"PKL文件: {len(pkl_files)} 个")

# 解析文件名
experiments = defaultdict(lambda: defaultdict(set))

for f in json_files:
    parts = f.stem.split('_')
    if len(parts) >= 4:
        matcher = parts[0]
        vpr_method = parts[1]
        distance = parts[2]
        dataset = '_'.join(parts[3:])
        
        key = f"{vpr_method}_{distance}_{dataset}"
        experiments[key][matcher].add('json')

for f in pkl_files:
    parts = f.stem.split('_')
    if len(parts) >= 4:
        matcher = parts[0]
        vpr_method = parts[1]
        distance = parts[2]
        dataset = '_'.join(parts[3:])
        
        key = f"{vpr_method}_{distance}_{dataset}"
        experiments[key][matcher].add('pkl')

print("\n" + "="*80)
print("按数据集统计")
print("="*80)

datasets = defaultdict(lambda: defaultdict(set))
for key, matchers in experiments.items():
    parts = key.split('_')
    if len(parts) >= 3:
        vpr_method = parts[0]
        distance = parts[1]
        dataset = '_'.join(parts[2:])
        
        datasets[dataset][vpr_method].update(matchers.keys())

for dataset in sorted(datasets.keys()):
    print(f"\n📊 {dataset}:")
    for vpr_method in sorted(datasets[dataset].keys()):
        matchers = sorted(datasets[dataset][vpr_method])
        print(f"  {vpr_method}: {len(matchers)}/3 matchers - {', '.join(matchers)}")

print("\n" + "="*80)
print("按VPR方法统计")
print("="*80)

vpr_methods = defaultdict(lambda: defaultdict(set))
for key, matchers in experiments.items():
    parts = key.split('_')
    if len(parts) >= 3:
        vpr_method = parts[0]
        distance = parts[1]
        dataset = '_'.join(parts[2:])
        
        vpr_methods[vpr_method][dataset].update(matchers.keys())

for vpr_method in sorted(vpr_methods.keys()):
    print(f"\n📊 {vpr_method}:")
    for dataset in sorted(vpr_methods[vpr_method].keys()):
        matchers = sorted(vpr_methods[vpr_method][dataset])
        print(f"  {dataset}: {len(matchers)}/3 matchers - {', '.join(matchers)}")

print("\n" + "="*80)
print("完成度统计")
print("="*80)

# 理论上应该有：4 VPR × 4 datasets × 3 matchers = 48个实验
# 但实际可能只用了dot_product
expected = 4 * 4 * 3  # 4 VPR × 4 datasets × 3 matchers
actual = sum(len(matchers) for matchers in experiments.values())

print(f"\n理论总数: {expected} 个实验 (4 VPR × 4 datasets × 3 matchers)")
print(f"实际完成: {actual} 个实验")
print(f"完成度: {actual/expected*100:.1f}%")

# 检查哪些组合缺失
print("\n缺失的实验:")
all_vpr = ['cosplace', 'netvlad', 'mixvpr', 'megaloc']
all_datasets = ['sf_xs_test', 'tokyo_xs_test', 'svox_sun_test', 'svox_night_test']
all_matchers = ['superpoint-lg', 'loftr', 'superglue']

missing = []
for vpr in all_vpr:
    for dataset in all_datasets:
        for matcher in all_matchers:
            key = f"{vpr}_dot_product_{dataset}"
            if matcher not in experiments.get(key, {}):
                missing.append(f"{vpr} + {matcher} + {dataset}")

if missing:
    print(f"  共缺失 {len(missing)} 个实验")
    print(f"  示例（前10个）:")
    for m in missing[:10]:
        print(f"    - {m}")
    if len(missing) > 10:
        print(f"    ... 还有 {len(missing)-10} 个")
else:
    print("  ✅ 所有实验都已完成！")

print("="*80)
