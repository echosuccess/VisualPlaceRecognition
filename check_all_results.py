#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量检查所有Image Matching结果文件的数据分布
"""

from pathlib import Path
import json
from collections import defaultdict

def check_single_file(result_file):
    """检查单个结果文件"""
    try:
        with open(str(result_file), 'r') as f:
            data = json.load(f)
        
        # 检查关键字段
        if 'results' not in data:
            return {
                'status': 'ERROR',
                'error': '缺少results字段'
            }
        
        results = data['results']
        required_fields = ['num_inliers', 'is_correct', 'query_ids']
        missing_fields = [f for f in required_fields if f not in results]
        
        if missing_fields:
            return {
                'status': 'ERROR',
                'error': f'缺少字段: {missing_fields}'
            }
        
        inliers = results['num_inliers']
        is_correct = results['is_correct']
        query_ids = results['query_ids']
        
        # 检查数据长度一致性
        if len(inliers) != len(is_correct) or len(inliers) != len(query_ids):
            return {
                'status': 'ERROR',
                'error': f'数据长度不一致: inliers={len(inliers)}, is_correct={len(is_correct)}, query_ids={len(query_ids)}'
            }
        
        # 检查数据分布
        correct_inliers = [inliers[i] for i in range(len(inliers)) if is_correct[i]]
        incorrect_inliers = [inliers[i] for i in range(len(inliers)) if not is_correct[i]]
        
        if len(correct_inliers) == 0 or len(incorrect_inliers) == 0:
            return {
                'status': 'WARN',
                'error': '数据不足（正确或错误预测为空）',
                'num_queries': len(query_ids),
                'num_matches': len(inliers),
                'num_correct': len(correct_inliers),
                'num_incorrect': len(incorrect_inliers)
            }
        
        correct_mean = sum(correct_inliers) / len(correct_inliers)
        incorrect_mean = sum(incorrect_inliers) / len(incorrect_inliers)
        ratio = correct_mean / max(incorrect_mean, 1)
        
        # 判断数据是否合理
        is_reasonable = ratio > 1.5
        
        return {
            'status': 'OK' if is_reasonable else 'WARN',
            'num_queries': len(query_ids),
            'num_matches': len(inliers),
            'num_correct': len(correct_inliers),
            'num_incorrect': len(incorrect_inliers),
            'correct_mean_inliers': correct_mean,
            'incorrect_mean_inliers': incorrect_mean,
            'ratio': ratio,
            'is_reasonable': is_reasonable
        }
        
    except json.JSONDecodeError as e:
        return {
            'status': 'ERROR',
            'error': f'JSON解析错误: {e}'
        }
    except Exception as e:
        return {
            'status': 'ERROR',
            'error': f'读取错误: {e}'
        }

def main():
    print("="*80)
    print("批量检查所有Image Matching结果文件")
    print("="*80)
    
    results_dir = Path("results/image_matching")
    if not results_dir.exists():
        print(f"\n[ERROR] {results_dir} 目录不存在")
        return
    
    # 查找所有JSON文件
    json_files = sorted([f for f in results_dir.glob("*.json") if f.name != 'batch_summary.json'])
    
    if not json_files:
        print(f"\n[INFO] 未找到结果文件")
        return
    
    print(f"\n找到 {len(json_files)} 个结果文件\n")
    
    # 检查每个文件
    results_summary = {
        'OK': [],
        'WARN': [],
        'ERROR': []
    }
    
    for json_file in json_files:
        size_mb = json_file.stat().st_size / (1024*1024)
        result = check_single_file(json_file)
        result['file'] = json_file.name
        result['size_mb'] = size_mb
        
        status = result['status']
        results_summary[status].append(result)
        
        # 显示结果
        if status == 'OK':
            print(f"✅ {json_file.name}")
            print(f"   大小: {size_mb:.1f}MB")
            print(f"   查询数: {result['num_queries']}, 匹配数: {result['num_matches']}")
            print(f"   正确预测平均inliers: {result['correct_mean_inliers']:.1f}")
            print(f"   错误预测平均inliers: {result['incorrect_mean_inliers']:.1f}")
            print(f"   比率: {result['ratio']:.2f}x (合理)")
        elif status == 'WARN':
            print(f"⚠️  {json_file.name}")
            print(f"   大小: {size_mb:.1f}MB")
            if 'error' in result:
                print(f"   {result['error']}")
            else:
                print(f"   查询数: {result['num_queries']}, 匹配数: {result['num_matches']}")
                print(f"   正确预测平均inliers: {result['correct_mean_inliers']:.1f}")
                print(f"   错误预测平均inliers: {result['incorrect_mean_inliers']:.1f}")
                print(f"   比率: {result['ratio']:.2f}x (异常，可能存在数据混乱)")
        else:  # ERROR
            print(f"❌ {json_file.name}")
            print(f"   大小: {size_mb:.1f}MB")
            print(f"   错误: {result['error']}")
        print()
    
    # 总结
    print("="*80)
    print("总结")
    print("="*80)
    
    total = len(json_files)
    ok_count = len(results_summary['OK'])
    warn_count = len(results_summary['WARN'])
    error_count = len(results_summary['ERROR'])
    
    print(f"\n总计: {total} 个文件")
    print(f"  ✅ 正常: {ok_count}")
    print(f"  ⚠️  警告: {warn_count}")
    print(f"  ❌ 错误: {error_count}")
    
    if warn_count > 0:
        print(f"\n⚠️  发现 {warn_count} 个有警告的文件:")
        for r in results_summary['WARN']:
            if 'error' in r:
                print(f"    - {r['file']}: {r['error']}")
            else:
                print(f"    - {r['file']}: 比率 {r['ratio']:.2f}x (异常)")
        print("\n建议：检查这些文件的数据分布，如果异常可能需要重新运行")
    
    if error_count > 0:
        print(f"\n❌ 发现 {error_count} 个有错误的文件:")
        for r in results_summary['ERROR']:
            print(f"    - {r['file']}: {r['error']}")
        print("\n建议：这些文件需要修复或重新运行")
    
    if ok_count == total:
        print("\n✅ 所有文件检查通过！数据分布都合理。")
    elif ok_count + warn_count == total and error_count == 0:
        print("\n⚠️  所有文件都可以读取，但部分文件的数据分布异常，建议检查。")
    else:
        print("\n❌ 发现错误文件，需要处理。")
    
    # 按VPR实验分组统计
    print("\n" + "="*80)
    print("按VPR实验分组统计")
    print("="*80)
    
    vpr_groups = defaultdict(list)
    for r in results_summary['OK'] + results_summary['WARN']:
        # 解析文件名: {matcher}_{vpr_method}_{distance}_{dataset}.json
        parts = r['file'].replace('.json', '').split('_')
        if len(parts) >= 4:
            matcher = parts[0]
            vpr_method = parts[1]
            distance = parts[2]
            dataset = '_'.join(parts[3:])
            vpr_exp_key = f"{vpr_method}_{distance}_{dataset}"
            vpr_groups[vpr_exp_key].append({
                'matcher': matcher,
                'result': r
            })
    
    for vpr_exp_key, exps in sorted(vpr_groups.items()):
        if len(exps) > 1:
            print(f"\n  {vpr_exp_key}: {len(exps)} 个matcher")
            for exp in exps:
                r = exp['result']
                if r['status'] == 'OK':
                    print(f"    ✅ {exp['matcher']}: 比率 {r['ratio']:.2f}x")
                else:
                    print(f"    ⚠️  {exp['matcher']}: 比率 {r['ratio']:.2f}x (异常)")

if __name__ == "__main__":
    main()
