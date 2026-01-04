#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新运行Tokyo-XS异常文件
"""

import subprocess
import sys
from pathlib import Path

# 必须重新运行（文件不完整）
must_rerun = [
    {
        'matcher': 'superpoint-lg',
        'vpr_method': 'mixvpr',
        'dataset': 'tokyo_xs_test',
        'distance': 'dot_product',
        'reason': '文件不完整（只有13KB）'
    }
]

# 可选重新运行（比率异常）
optional_rerun = [
    {
        'matcher': 'loftr',
        'vpr_method': 'megaloc',
        'dataset': 'tokyo_xs_test',
        'distance': 'dot_product',
        'reason': '比率 1.14x (异常)'
    },
    {
        'matcher': 'superglue',
        'vpr_method': 'cosplace',
        'dataset': 'tokyo_xs_test',
        'distance': 'dot_product',
        'reason': '比率 1.48x (异常)'
    },
    {
        'matcher': 'superglue',
        'vpr_method': 'megaloc',
        'dataset': 'tokyo_xs_test',
        'distance': 'dot_product',
        'reason': '比率 1.02x (异常)'
    },
    {
        'matcher': 'superglue',
        'vpr_method': 'mixvpr',
        'dataset': 'tokyo_xs_test',
        'distance': 'dot_product',
        'reason': '比率 1.45x (异常)'
    },
    {
        'matcher': 'superpoint-lg',
        'vpr_method': 'cosplace',
        'dataset': 'tokyo_xs_test',
        'distance': 'dot_product',
        'reason': '比率 1.36x (异常)'
    },
    {
        'matcher': 'superpoint-lg',
        'vpr_method': 'megaloc',
        'dataset': 'tokyo_xs_test',
        'distance': 'dot_product',
        'reason': '比率 1.23x (异常)'
    },
]

def build_command(exp):
    """构建运行命令"""
    cmd = [
        'python', 'run_image_matching_baseline.py',
        '--matcher', exp['matcher'],
        '--vpr_method', exp['vpr_method'],
        '--dataset', exp['dataset'],
        '--distance', exp['distance'],
        '--device', 'cuda'
    ]
    return cmd

def check_result_exists(exp):
    """检查结果文件是否已存在"""
    result_file = Path(f"results/image_matching/{exp['matcher']}_{exp['vpr_method']}_{exp['distance']}_{exp['dataset']}.json")
    return result_file.exists()

def main():
    print("="*80)
    print("Tokyo-XS异常文件重新运行脚本")
    print("="*80)
    
    # 检查必须重新运行的文件
    print("\n[1] 必须重新运行的文件（文件不完整）:")
    must_run = []
    for exp in must_rerun:
        if check_result_exists(exp):
            size_mb = Path(f"results/image_matching/{exp['matcher']}_{exp['vpr_method']}_{exp['distance']}_{exp['dataset']}.json").stat().st_size / (1024*1024)
            if size_mb < 0.1:  # 小于100KB认为不完整
                print(f"  [MUST] {exp['matcher']} + {exp['vpr_method']}: {exp['reason']} (当前大小: {size_mb:.2f}MB)")
                must_run.append(exp)
            else:
                print(f"  [OK] {exp['matcher']} + {exp['vpr_method']}: 文件已存在且完整 ({size_mb:.2f}MB)")
        else:
            print(f"  [MUST] {exp['matcher']} + {exp['vpr_method']}: {exp['reason']} (文件不存在)")
            must_run.append(exp)
    
    # 检查可选重新运行的文件
    print("\n[2] 可选重新运行的文件（比率异常）:")
    optional_run = []
    for exp in optional_rerun:
        if check_result_exists(exp):
            print(f"  [OPTIONAL] {exp['matcher']} + {exp['vpr_method']}: {exp['reason']} (文件已存在)")
            optional_run.append(exp)
        else:
            print(f"  [OPTIONAL] {exp['matcher']} + {exp['vpr_method']}: {exp['reason']} (文件不存在)")
            optional_run.append(exp)
    
    # 询问用户
    print("\n" + "="*80)
    print("运行选项")
    print("="*80)
    
    if must_run:
        print(f"\n必须重新运行: {len(must_run)} 个")
        for exp in must_run:
            print(f"  - {exp['matcher']} + {exp['vpr_method']}: {exp['reason']}")
    
    if optional_run:
        print(f"\n可选重新运行: {len(optional_run)} 个")
        print("  这些文件的比率 < 1.5x，但可能是Tokyo-XS数据集本身的问题")
        print("  如果时间允许，建议重新运行以确保准确性")
    
    if not must_run and not optional_run:
        print("\n[OK] 所有文件都已存在，不需要重新运行")
        return
    
    # 生成命令
    print("\n" + "="*80)
    print("生成的运行命令")
    print("="*80)
    
    all_commands = []
    
    if must_run:
        print("\n[必须运行]")
        for exp in must_run:
            cmd = build_command(exp)
            cmd_str = ' '.join(cmd)
            print(f"\n{cmd_str}")
            all_commands.append(cmd_str)
    
    if optional_run:
        print("\n[可选运行]")
        for exp in optional_run:
            cmd = build_command(exp)
            cmd_str = ' '.join(cmd)
            print(f"\n{cmd_str}")
            all_commands.append(cmd_str)
    
    # 生成Colab并行命令
    print("\n" + "="*80)
    print("Colab并行运行命令（复制到不同的Cell）")
    print("="*80)
    
    for i, cmd_str in enumerate(all_commands, 1):
        print(f"\n# Cell {i}")
        print(f"!{cmd_str}")
    
    print("\n" + "="*80)
    print("说明")
    print("="*80)
    print("\n1. 将上面的命令复制到不同的Colab Cell中")
    print("2. 可以同时运行多个Cell（但注意GPU资源限制）")
    print("3. 使用新代码运行，每个matcher会有独立的checkpoint，不会冲突")

if __name__ == "__main__":
    main()
