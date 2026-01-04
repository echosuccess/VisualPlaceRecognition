#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比sfxs_val数据集上不同matcher的结果
"""

import json
from pathlib import Path
import sys

def analyze_result_file(result_file):
    """分析单个结果文件"""
    result_file = Path(result_file)
    
    if not result_file.exists():
        return None
    
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data.get('results', {})
        query_ids = results.get('query_ids', [])
        is_correct = results.get('is_correct', [])
        num_inliers = results.get('num_inliers', [])
        pred_ranks = results.get('pred_ranks', [])
        
        # 统计
        total_matches = len(is_correct)
        num_correct = sum(1 for x in is_correct if x)
        correct_ratio = num_correct / total_matches if total_matches > 0 else 0
        
        # 只统计top-1（rank=0）的正确率
        top1_matches = [i for i, rank in enumerate(pred_ranks) if rank == 0]
        top1_correct = sum(1 for i in top1_matches if is_correct[i])
        top1_ratio = top1_correct / len(top1_matches) if len(top1_matches) > 0 else 0
        
        # 统计唯一queries
        unique_queries = len(set(query_ids))
        
        # Inliers统计
        if num_inliers:
            correct_inliers = [num_inliers[i] for i in range(len(num_inliers)) if is_correct[i]]
            incorrect_inliers = [num_inliers[i] for i in range(len(num_inliers)) if not is_correct[i]]
            
            if correct_inliers and incorrect_inliers:
                mean_correct_inliers = sum(correct_inliers) / len(correct_inliers)
                mean_incorrect_inliers = sum(incorrect_inliers) / len(incorrect_inliers)
                ratio = mean_correct_inliers / max(mean_incorrect_inliers, 1)
            else:
                mean_correct_inliers = 0
                mean_incorrect_inliers = 0
                ratio = 0
        else:
            mean_correct_inliers = 0
            mean_incorrect_inliers = 0
            ratio = 0
        
        return {
            'file': result_file.name,
            'total_matches': total_matches,
            'unique_queries': unique_queries,
            'num_correct': num_correct,
            'correct_ratio': correct_ratio,
            'top1_correct': top1_correct,
            'top1_ratio': top1_ratio,
            'mean_correct_inliers': mean_correct_inliers,
            'mean_incorrect_inliers': mean_incorrect_inliers,
            'inlier_ratio': ratio
        }
    except Exception as e:
        print(f"❌ 无法读取 {result_file.name}: {e}")
        return None

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("="*80)
    print("对比 sfxs_val 数据集上的 Image Matching 结果")
    print("="*80)
    
    results_dir = Path("results/image_matching")
    
    # 查找所有sfxs_val的结果文件
    sfxs_val_files = sorted(results_dir.glob("*_*_l2_sfxs_val.json"))
    
    if not sfxs_val_files:
        print("\n⚠️  没有找到 sfxs_val 的结果文件")
        print(f"   检查目录: {results_dir}")
        return
    
    print(f"\n找到 {len(sfxs_val_files)} 个 sfxs_val 结果文件:\n")
    
    all_results = []
    for f in sfxs_val_files:
        result = analyze_result_file(f)
        if result:
            all_results.append(result)
    
    if not all_results:
        print("❌ 无法读取任何结果文件")
        return
    
    # 打印对比表格
    print(f"{'Matcher':<20} {'Queries':<10} {'总匹配':<10} {'正确数':<10} {'正确率':<12} {'Top-1率':<12} {'Inlier比率':<12}")
    print("-" * 100)
    
    for r in all_results:
        # 从文件名提取matcher和vpr_method
        parts = r['file'].replace('.json', '').split('_')
        matcher = parts[0]
        vpr_method = parts[1] if len(parts) > 1 else 'unknown'
        exp_name = f"{matcher} + {vpr_method}"
        
        print(f"{exp_name:<20} {r['unique_queries']:<10} {r['total_matches']:<10} "
              f"{r['num_correct']:<10} {r['correct_ratio']*100:>10.2f}% "
              f"{r['top1_ratio']*100:>10.2f}% {r['inlier_ratio']:>10.2f}x")
    
    # 分析
    print("\n" + "="*80)
    print("分析")
    print("="*80)
    
    # 检查loftr的结果
    loftr_results = [r for r in all_results if 'loftr' in r['file']]
    if loftr_results:
        print(f"\nLoFTR 结果:")
        for r in loftr_results:
            print(f"  {r['file']}:")
            print(f"    Top-20 正确率: {r['correct_ratio']*100:.2f}%")
            print(f"    Top-1 正确率: {r['top1_ratio']*100:.2f}%")
            print(f"    Inlier比率: {r['inlier_ratio']:.2f}x")
            
            if r['correct_ratio'] < 0.05:
                print(f"    ⚠️  Top-20正确率 < 5%，可能异常")
            elif r['correct_ratio'] < 0.10:
                print(f"    ⚠️  Top-20正确率 < 10%，较低但可能正常")
            else:
                print(f"    ✅ Top-20正确率正常")
            
            if r['inlier_ratio'] < 1.5:
                print(f"    ⚠️  Inlier比率 < 1.5x，可能异常")
            else:
                print(f"    ✅ Inlier比率正常（正确预测的inliers明显更多）")
    
    # 对比其他matcher
    other_results = [r for r in all_results if 'loftr' not in r['file']]
    if other_results:
        print(f"\n其他 Matcher 结果（用于对比）:")
        for r in other_results:
            print(f"  {r['file']}:")
            print(f"    Top-20 正确率: {r['correct_ratio']*100:.2f}%")
            print(f"    Top-1 正确率: {r['top1_ratio']*100:.2f}%")
            print(f"    Inlier比率: {r['inlier_ratio']:.2f}x")
    
    # 总结
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    
    if loftr_results:
        loftr = loftr_results[0]
        print(f"\nLoFTR + CosPlace + sfxs_val:")
        print(f"  - Top-20正确率: {loftr['correct_ratio']*100:.2f}%")
        print(f"  - Top-1正确率: {loftr['top1_ratio']*100:.2f}%")
        print(f"  - Inlier比率: {loftr['inlier_ratio']:.2f}x")
        
        if other_results:
            other = other_results[0]
            print(f"\n对比 ({other['file']}):")
            print(f"  - Top-20正确率: {other['correct_ratio']*100:.2f}%")
            print(f"  - Top-1正确率: {other['top1_ratio']*100:.2f}%")
            print(f"  - Inlier比率: {other['inlier_ratio']:.2f}x")
            
            # 判断是否异常
            if loftr['correct_ratio'] < other['correct_ratio'] * 0.5:
                print(f"\n⚠️  警告: LoFTR的正确率明显低于其他matcher")
                print(f"   可能原因:")
                print(f"     1. LoFTR在这个数据集上表现较差")
                print(f"     2. 数据有问题")
            elif loftr['inlier_ratio'] < 1.5:
                print(f"\n⚠️  警告: LoFTR的Inlier比率异常低")
                print(f"   说明正确和错误预测的inliers差异不大")
                print(f"   可能数据有问题或LoFTR不适合这个数据集")
            else:
                print(f"\n✅ LoFTR结果看起来正常（与其他matcher相比）")

if __name__ == "__main__":
    main()
