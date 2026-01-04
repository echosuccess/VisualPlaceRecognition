#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析checkpoint文件，判断是否有冲突风险
"""

import pickle
from pathlib import Path
from datetime import datetime

def analyze_checkpoint(checkpoint_path):
    """分析单个checkpoint文件"""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        return None
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # 提取信息
        last_query_idx = checkpoint.get('last_query_idx', -1)
        num_queries = checkpoint.get('num_queries', -1)
        results = checkpoint.get('results', {})
        
        # 从results推断matcher（通过检查数据特征）
        # 注意：旧checkpoint没有保存matcher信息
        num_matches = len(results.get('query_ids', []))
        progress = (last_query_idx + 1) / num_queries * 100 if num_queries > 0 else 0
        
        # 检查文件修改时间
        mtime = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)
        
        return {
            'file': checkpoint_path.name,
            'last_query_idx': last_query_idx,
            'num_queries': num_queries,
            'num_matches': num_matches,
            'progress': progress,
            'modified_time': mtime,
            'file_size_mb': checkpoint_path.stat().st_size / (1024 * 1024)
        }
    except Exception as e:
        return {
            'file': checkpoint_path.name,
            'error': str(e)
        }

def main():
    """主函数"""
    checkpoint_dir = Path("checkpoints/image_matching")
    
    if not checkpoint_dir.exists():
        print("Checkpoint目录不存在")
        return
    
    # 查找所有checkpoint文件
    checkpoint_files = sorted(checkpoint_dir.glob("*checkpoint*.pkl"))
    
    print("="*80)
    print("Checkpoint文件分析")
    print("="*80)
    print(f"\n找到 {len(checkpoint_files)} 个checkpoint文件\n")
    
    analyses = []
    for ckpt_file in checkpoint_files:
        analysis = analyze_checkpoint(ckpt_file)
        if analysis:
            analyses.append(analysis)
    
    # 按修改时间排序
    analyses.sort(key=lambda x: x.get('modified_time', datetime.min))
    
    # 显示分析结果
    for i, analysis in enumerate(analyses, 1):
        if 'error' in analysis:
            print(f"{i}. {analysis['file']}")
            print(f"   ❌ 错误: {analysis['error']}")
        else:
            print(f"{i}. {analysis['file']}")
            print(f"   进度: {analysis['progress']:.1f}% ({analysis['last_query_idx']+1}/{analysis['num_queries']})")
            print(f"   已匹配: {analysis['num_matches']} 个")
            print(f"   文件大小: {analysis['file_size_mb']:.2f} MB")
            print(f"   修改时间: {analysis['modified_time']}")
        print()
    
    # 检查冲突风险
    print("="*80)
    print("冲突风险分析")
    print("="*80)
    
    # 按VPR日志时间戳分组
    vpr_timestamps = {}
    for analysis in analyses:
        if 'error' in analysis:
            continue
        # 从文件名提取VPR时间戳（旧格式：YYYY-MM-DD_HH-MM-SS_checkpoint.pkl）
        filename = analysis['file']
        if '_checkpoint.pkl' in filename:
            timestamp = filename.replace('_checkpoint.pkl', '')
            if timestamp not in vpr_timestamps:
                vpr_timestamps[timestamp] = []
            vpr_timestamps[timestamp].append(analysis)
    
    # 检查是否有多个checkpoint使用同一个VPR时间戳
    conflicts = []
    for timestamp, checkpoints in vpr_timestamps.items():
        if len(checkpoints) > 1:
            conflicts.append((timestamp, checkpoints))
    
    if conflicts:
        print(f"\n⚠️  发现 {len(conflicts)} 个潜在的checkpoint冲突:")
        for timestamp, checkpoints in conflicts:
            print(f"\n  VPR时间戳: {timestamp}")
            print(f"  有 {len(checkpoints)} 个checkpoint文件:")
            for ckpt in checkpoints:
                print(f"    - {ckpt['file']} (进度: {ckpt['progress']:.1f}%)")
            print(f"  ⚠️  这些checkpoint可能来自不同的matcher，会互相覆盖！")
    else:
        print("\n✅ 没有发现checkpoint冲突（每个VPR时间戳只有一个checkpoint）")
    
    # 建议
    print("\n" + "="*80)
    print("建议")
    print("="*80)
    
    if conflicts:
        print("\n❌ 发现冲突！建议：")
        print("  1. 中断所有正在运行的实验")
        print("  2. 更新代码（使用修复后的版本，checkpoint文件名包含matcher名称）")
        print("  3. 删除旧的checkpoint文件（或重命名保存）")
        print("  4. 重新运行实验")
    else:
        print("\n✅ 当前没有冲突，但建议：")
        print("  1. 确保使用修复后的代码（checkpoint文件名包含matcher名称）")
        print("  2. 如果后续要并行运行多个matcher，先更新代码")
    
    print("\n注意：")
    print("  - 旧格式checkpoint文件名：YYYY-MM-DD_HH-MM-SS_checkpoint.pkl")
    print("  - 新格式checkpoint文件名：matcher_YYYY-MM-DD_HH-MM-SS_checkpoint.pkl")
    print("  - 新格式可以避免多个matcher同时运行时的冲突")

if __name__ == "__main__":
    main()
