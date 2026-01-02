#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动重新运行损坏的Image Matching实验
检测损坏的JSON文件并重新运行对应的实验
"""

import json
import subprocess
from pathlib import Path
import sys


def check_json_valid(json_file):
    """检查JSON文件是否有效"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'results' in data and 'analysis' in data:
            return True, None
        else:
            return False, "Missing required keys"
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, str(e)


def parse_filename(filename):
    """解析文件名获取实验参数"""
    # 格式: matcher_vpr_method_distance_dataset.json
    # 例如: superglue_megaloc_dot_product_sf_xs_test.json
    parts = filename.stem.split('_')
    
    if len(parts) < 4:
        return None
    
    matcher = parts[0]
    vpr_method = parts[1]
    
    # distance可能是dot_product (2个词) 或 l2 (1个词)
    if parts[2] == 'dot':
        distance = 'dot_product'
        dataset = '_'.join(parts[4:])
    else:
        distance = parts[2]
        dataset = '_'.join(parts[3:])
    
    return {
        'matcher': matcher,
        'vpr_method': vpr_method,
        'distance': distance,
        'dataset': dataset,
        'filename': filename.name
    }


def find_corrupted_experiments(results_dir="results/image_matching"):
    """查找所有损坏的实验"""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"❌ Directory not found: {results_dir}")
        return []
    
    json_files = list(results_dir.glob("*.json"))
    json_files = [f for f in json_files if 'summary' not in f.name and not f.name.endswith('.bak')]
    
    corrupted = []
    
    for json_file in json_files:
        is_valid, error = check_json_valid(json_file)
        if not is_valid:
            exp_info = parse_filename(json_file)
            if exp_info:
                exp_info['error'] = error
                exp_info['json_file'] = json_file
                corrupted.append(exp_info)
    
    return corrupted


def run_experiment(exp_info, top_k=20, device='cuda', delete_corrupted=True):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"Running: {exp_info['matcher']} on {exp_info['vpr_method']} ({exp_info['dataset']})")
    print(f"{'='*80}")
    
    # 删除损坏的JSON文件（如果存在）
    if delete_corrupted and exp_info['json_file'].exists():
        print(f"[INFO] Removing corrupted file: {exp_info['json_file'].name}")
        exp_info['json_file'].unlink()
    
    # 构建命令
    cmd = [
        'python', 'run_image_matching_baseline.py',
        '--matcher', exp_info['matcher'],
        '--vpr_method', exp_info['vpr_method'],
        '--dataset', exp_info['dataset'],
        '--distance', exp_info['distance'],
        '--top_k', str(top_k),
        '--device', device
    ]
    
    print(f"[INFO] Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ [OK] Completed: {exp_info['filename']}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ [FAIL] Failed: {exp_info['filename']}")
        print(f"   Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  [INTERRUPTED] Experiment interrupted by user")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Re-run corrupted Image Matching experiments')
    parser.add_argument('--top_k', type=int, default=20, help='Top-K predictions to match')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--dir', type=str, default='results/image_matching', 
                       help='Results directory')
    parser.add_argument('--keep-corrupted', action='store_true',
                       help='Keep corrupted JSON files (don\'t delete)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only show what would be run, don\'t actually run')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("Finding Corrupted Image Matching Experiments")
    print(f"{'='*80}\n")
    
    # 查找损坏的实验
    corrupted = find_corrupted_experiments(args.dir)
    
    if not corrupted:
        print("✅ No corrupted experiments found! All JSON files are valid.")
        return
    
    print(f"Found {len(corrupted)} corrupted experiment(s):\n")
    
    for i, exp in enumerate(corrupted, 1):
        print(f"{i}. {exp['filename']}")
        print(f"   Matcher: {exp['matcher']}")
        print(f"   VPR Method: {exp['vpr_method']}")
        print(f"   Dataset: {exp['dataset']}")
        print(f"   Error: {exp['error']}")
        print()
    
    if args.dry_run:
        print("🔍 [DRY RUN] Would run the above experiments")
        print("   Remove --dry-run to actually run them")
        return
    
    # 确认
    print(f"{'='*80}")
    response = input(f"Re-run {len(corrupted)} experiment(s)? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # 运行实验
    print(f"\n{'='*80}")
    print("Re-running Corrupted Experiments")
    print(f"{'='*80}\n")
    
    success_count = 0
    fail_count = 0
    
    for i, exp in enumerate(corrupted, 1):
        print(f"\n[{i}/{len(corrupted)}]")
        if run_experiment(exp, top_k=args.top_k, device=args.device, 
                         delete_corrupted=not args.keep_corrupted):
            success_count += 1
        else:
            fail_count += 1
    
    # 总结
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Total: {len(corrupted)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
