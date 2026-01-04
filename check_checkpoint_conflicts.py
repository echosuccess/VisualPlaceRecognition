#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查checkpoint冲突并识别对应的实验
"""

import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def analyze_checkpoint_file(ckpt_path):
    """分析单个checkpoint文件"""
    ckpt_path = Path(ckpt_path)
    
    if not ckpt_path.exists():
        return None
    
    try:
        with open(ckpt_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        last_query_idx = checkpoint.get('last_query_idx', -1)
        num_queries = checkpoint.get('num_queries', -1)
        results = checkpoint.get('results', {})
        K = checkpoint.get('K', 20)
        
        num_matches = len(results.get('query_ids', []))
        progress = (last_query_idx + 1) / num_queries * 100 if num_queries > 0 else 0
        
        # 从文件名提取VPR时间戳
        filename = ckpt_path.name
        if '_checkpoint.pkl' in filename:
            # 旧格式：YYYY-MM-DD_HH-MM-SS_checkpoint.pkl
            # 新格式：matcher_YYYY-MM-DD_HH-MM-SS_checkpoint.pkl
            if filename.startswith(('superglue_', 'loftr_', 'superpoint-lg_')):
                # 新格式
                parts = filename.replace('_checkpoint.pkl', '').split('_', 1)
                matcher_name = parts[0]
                vpr_timestamp = parts[1] if len(parts) > 1 else 'unknown'
            else:
                # 旧格式
                matcher_name = 'unknown'
                vpr_timestamp = filename.replace('_checkpoint.pkl', '')
        else:
            matcher_name = 'unknown'
            vpr_timestamp = 'unknown'
        
        mtime = datetime.fromtimestamp(ckpt_path.stat().st_mtime)
        
        return {
            'file': filename,
            'matcher': matcher_name,
            'vpr_timestamp': vpr_timestamp,
            'last_query_idx': last_query_idx,
            'num_queries': num_queries,
            'num_matches': num_matches,
            'progress': progress,
            'modified_time': mtime,
            'file_size_mb': ckpt_path.stat().st_size / (1024 * 1024),
            'is_old_format': matcher_name == 'unknown'
        }
    except Exception as e:
        return {
            'file': ckpt_path.name,
            'error': str(e)
        }

def find_vpr_experiment_for_timestamp(timestamp):
    """根据时间戳查找对应的VPR实验"""
    baseline_dir = Path("logs/baseline")
    if not baseline_dir.exists():
        return []
    
    matching_experiments = []
    
    # 遍历所有VPR实验目录
    for exp_dir in baseline_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # 查找时间戳目录
        for timestamp_dir in exp_dir.iterdir():
            if timestamp_dir.is_dir() and timestamp_dir.name == timestamp:
                matching_experiments.append({
                    'vpr_exp': exp_dir.name,
                    'timestamp_dir': timestamp_dir.name
                })
    
    return matching_experiments

def main():
    """主函数"""
    checkpoint_dir = Path("checkpoints/image_matching")
    
    if not checkpoint_dir.exists():
        print("Checkpoint目录不存在")
        return
    
    checkpoint_files = sorted(checkpoint_dir.glob("*checkpoint*.pkl"))
    
    print("="*80)
    print("Checkpoint冲突分析")
    print("="*80)
    print(f"\n找到 {len(checkpoint_files)} 个checkpoint文件\n")
    
    analyses = []
    for ckpt_file in checkpoint_files:
        analysis = analyze_checkpoint_file(ckpt_file)
        if analysis:
            analyses.append(analysis)
    
    # 按修改时间排序
    analyses.sort(key=lambda x: x.get('modified_time', datetime.min))
    
    # 显示详细信息
    print("Checkpoint文件详情:")
    print("-"*80)
    for i, analysis in enumerate(analyses, 1):
        if 'error' in analysis:
            print(f"{i}. {analysis['file']}")
            print(f"   ❌ 错误: {analysis['error']}")
        else:
            print(f"{i}. {analysis['file']}")
            if analysis['is_old_format']:
                print(f"   ⚠️  旧格式（无matcher名称）")
            else:
                print(f"   ✅ 新格式（matcher: {analysis['matcher']}）")
            print(f"   VPR时间戳: {analysis['vpr_timestamp']}")
            print(f"   进度: {analysis['progress']:.1f}% ({analysis['last_query_idx']+1}/{analysis['num_queries']})")
            print(f"   已匹配: {analysis['num_matches']} 个")
            print(f"   文件大小: {analysis['file_size_mb']:.2f} MB")
            print(f"   修改时间: {analysis['modified_time']}")
            
            # 查找对应的VPR实验
            vpr_exps = find_vpr_experiment_for_timestamp(analysis['vpr_timestamp'])
            if vpr_exps:
                print(f"   对应VPR实验: {', '.join([e['vpr_exp'] for e in vpr_exps])}")
        print()
    
    # 检查冲突
    print("="*80)
    print("冲突检测")
    print("="*80)
    
    # 按VPR时间戳分组
    by_timestamp = defaultdict(list)
    for analysis in analyses:
        if 'error' not in analysis:
            by_timestamp[analysis['vpr_timestamp']].append(analysis)
    
    conflicts = []
    for timestamp, checkpoints in by_timestamp.items():
        if len(checkpoints) > 1:
            conflicts.append((timestamp, checkpoints))
    
    if conflicts:
        print(f"\n❌ 发现 {len(conflicts)} 个冲突！")
        for timestamp, checkpoints in conflicts:
            print(f"\n  VPR时间戳: {timestamp}")
            print(f"  有 {len(checkpoints)} 个checkpoint文件:")
            for ckpt in checkpoints:
                matcher_info = f"matcher: {ckpt['matcher']}" if not ckpt['is_old_format'] else "matcher: 未知（旧格式）"
                print(f"    - {ckpt['file']}")
                print(f"      {matcher_info}, 进度: {ckpt['progress']:.1f}%")
            print(f"  ⚠️  这些checkpoint会互相覆盖！")
            
            # 查找对应的VPR实验
            vpr_exps = find_vpr_experiment_for_timestamp(timestamp)
            if vpr_exps:
                print(f"  对应VPR实验: {', '.join([e['vpr_exp'] for e in vpr_exps])}")
                print(f"  ⚠️  如果有多个matcher同时运行这个VPR实验，会冲突！")
    else:
        print("\n✅ 没有发现冲突（每个VPR时间戳只有一个checkpoint）")
    
    # 检查旧格式checkpoint
    old_format = [a for a in analyses if a.get('is_old_format', False) and 'error' not in a]
    if old_format:
        print(f"\n⚠️  发现 {len(old_format)} 个旧格式checkpoint（无matcher名称）:")
        for ckpt in old_format:
            print(f"  - {ckpt['file']} (VPR时间戳: {ckpt['vpr_timestamp']})")
        print("\n  这些checkpoint来自修复前的代码，如果多个matcher同时运行会冲突")
    
    # 建议
    print("\n" + "="*80)
    print("建议")
    print("="*80)
    
    if conflicts:
        print("\n❌ 发现冲突！必须处理：")
        print("  1. 立即中断所有正在运行的实验")
        print("  2. 更新代码（git pull 或手动更新 run_image_matching_baseline.py）")
        print("  3. 备份或删除冲突的checkpoint文件")
        print("  4. 使用修复后的代码重新运行")
    elif old_format:
        print("\n⚠️  有旧格式checkpoint，建议：")
        print("  1. 确保使用修复后的代码（checkpoint文件名包含matcher名称）")
        print("  2. 如果后续要并行运行多个matcher，先更新代码")
        print("  3. 当前正在运行的实验可以继续，但不要同时运行其他matcher")
    else:
        print("\n✅ 所有checkpoint都是新格式，可以安全并行运行多个matcher")

if __name__ == "__main__":
    main()
