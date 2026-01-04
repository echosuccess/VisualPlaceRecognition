#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证Image Matching结果文件的完整性
检查是否有数据丢失或checkpoint冲突导致的问题
"""

import json
import pickle
from pathlib import Path
import sys

def verify_result_file(result_path):
    """验证单个结果文件"""
    result_path = Path(result_path)
    
    if not result_path.exists():
        return {
            'status': 'MISSING',
            'file': str(result_path),
            'issues': ['文件不存在']
        }
    
    # 尝试加载
    try:
        if result_path.suffix == '.json':
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif result_path.suffix == '.pkl':
            with open(result_path, 'rb') as f:
                data = pickle.load(f)
        else:
            return {
                'status': 'UNKNOWN_FORMAT',
                'file': str(result_path),
                'issues': [f'未知格式: {result_path.suffix}']
            }
        
        # 检查数据结构
        if 'results' not in data:
            return {
                'status': 'INVALID',
                'file': str(result_path),
                'issues': ['缺少results字段']
            }
        
        results = data['results']
        
        # 检查关键字段
        required_fields = ['query_ids', 'pred_ranks', 'is_correct', 'num_inliers']
        missing_fields = [f for f in required_fields if f not in results]
        if missing_fields:
            return {
                'status': 'INCOMPLETE',
                'file': str(result_path),
                'issues': [f'缺少字段: {missing_fields}']
            }
        
        # 检查数据完整性
        query_ids = results['query_ids']
        num_queries = len(set(query_ids))  # 唯一query数量
        num_matches = len(query_ids)
        
        # 检查是否有重复的query+rank组合（不应该有）
        match_keys = [(qid, rank) for qid, rank in zip(query_ids, results['pred_ranks'])]
        unique_matches = len(set(match_keys))
        
        issues = []
        
        # 检查数据一致性
        if len(query_ids) != len(results['pred_ranks']):
            issues.append(f'数据长度不一致: query_ids={len(query_ids)}, pred_ranks={len(results["pred_ranks"])}')
        
        if len(query_ids) != len(results['is_correct']):
            issues.append(f'数据长度不一致: query_ids={len(query_ids)}, is_correct={len(results["is_correct"])}')
        
        if len(query_ids) != len(results['num_inliers']):
            issues.append(f'数据长度不一致: query_ids={len(query_ids)}, num_inliers={len(results["num_inliers"])}')
        
        # 检查是否有重复（可能表示checkpoint冲突）
        if unique_matches != num_matches:
            issues.append(f'发现重复的query+rank组合: 唯一={unique_matches}, 总计={num_matches}')
        
        # 检查每个query是否有完整的top-K预测
        # 假设K=20
        expected_matches_per_query = 20
        expected_total = num_queries * expected_matches_per_query
        
        if num_matches < expected_total * 0.95:  # 允许5%的容差（可能有些预测被跳过）
            issues.append(f'匹配数量可能不足: 期望至少{expected_total * 0.95:.0f}, 实际={num_matches}')
        
        # 检查analysis字段
        if 'analysis' not in data:
            issues.append('缺少analysis字段')
        
        status = 'OK' if not issues else 'WARNING'
        
        return {
            'status': status,
            'file': str(result_path),
            'num_queries': num_queries,
            'num_matches': num_matches,
            'expected_matches': expected_total,
            'unique_matches': unique_matches,
            'file_size_mb': result_path.stat().st_size / (1024 * 1024),
            'issues': issues
        }
        
    except json.JSONDecodeError as e:
        return {
            'status': 'CORRUPTED',
            'file': str(result_path),
            'issues': [f'JSON解析错误: {e}']
        }
    except Exception as e:
        return {
            'status': 'ERROR',
            'file': str(result_path),
            'issues': [f'加载错误: {e}']
        }

def main():
    """主函数"""
    results_dir = Path("results/image_matching")
    
    if not results_dir.exists():
        print(f"[ERROR] 结果目录不存在: {results_dir}")
        return
    
    # 查找所有结果文件
    json_files = list(results_dir.glob("*.json"))
    pkl_files = list(results_dir.glob("*.pkl"))
    
    # 优先使用JSON，如果没有则使用PKL
    result_files = []
    for json_file in json_files:
        result_files.append(json_file)
    for pkl_file in pkl_files:
        # 如果对应的JSON不存在，才使用PKL
        json_file = pkl_file.with_suffix('.json')
        if json_file not in json_files:
            result_files.append(pkl_file)
    
    print("="*80)
    print("Image Matching 结果文件完整性验证")
    print("="*80)
    print(f"\n找到 {len(result_files)} 个结果文件\n")
    
    results = []
    for result_file in sorted(result_files):
        print(f"检查: {result_file.name}...")
        verification = verify_result_file(result_file)
        results.append(verification)
        
        status_icon = {
            'OK': '✅',
            'WARNING': '⚠️',
            'INCOMPLETE': '❌',
            'CORRUPTED': '💥',
            'ERROR': '❌',
            'MISSING': '❌',
            'INVALID': '❌'
        }.get(verification['status'], '❓')
        
        print(f"  {status_icon} {verification['status']}")
        if verification.get('num_queries'):
            print(f"    Queries: {verification['num_queries']}, Matches: {verification['num_matches']}")
        if verification.get('issues'):
            for issue in verification['issues']:
                print(f"    ⚠️  {issue}")
        print()
    
    # 总结
    print("="*80)
    print("验证总结")
    print("="*80)
    
    status_counts = {}
    for r in results:
        status = r['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")
    
    # 列出有问题的文件
    problematic = [r for r in results if r['status'] not in ['OK', 'WARNING']]
    if problematic:
        print(f"\n⚠️  发现 {len(problematic)} 个有问题的文件:")
        for r in problematic:
            print(f"  - {Path(r['file']).name}: {r['status']}")
            for issue in r.get('issues', []):
                print(f"    {issue}")
    
    # 列出有警告的文件
    warnings = [r for r in results if r['status'] == 'WARNING']
    if warnings:
        print(f"\n⚠️  发现 {len(warnings)} 个有警告的文件:")
        for r in warnings:
            print(f"  - {Path(r['file']).name}")
            for issue in r.get('issues', []):
                print(f"    {issue}")
    
    print("\n" + "="*80)
    print("建议")
    print("="*80)
    
    if problematic:
        print("\n❌ 有问题的文件需要重新运行")
    elif warnings:
        print("\n⚠️  有警告的文件建议检查，但可能可以继续使用")
    else:
        print("\n✅ 所有文件看起来都是完整的！")
    
    print("\n注意：")
    print("  - 如果发现重复的query+rank组合，可能是checkpoint冲突导致的")
    print("  - 如果匹配数量明显不足，可能是实验被中断或checkpoint被覆盖")
    print("  - 建议使用修复后的代码重新运行有问题的实验")

if __name__ == "__main__":
    main()
