#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细分析异常文件，检查是否存在checkpoint冲突
"""

from pathlib import Path
import json
from collections import defaultdict

# 异常文件列表
abnormal_files = [
    'loftr_megaloc_dot_product_tokyo_xs_test.json',
    'superglue_cosplace_dot_product_tokyo_xs_test.json',
    'superglue_megaloc_dot_product_tokyo_xs_test.json',
    'superglue_mixvpr_dot_product_tokyo_xs_test.json',
    'superpoint-lg_cosplace_dot_product_tokyo_xs_test.json',
    'superpoint-lg_megaloc_dot_product_tokyo_xs_test.json',
    'superpoint-lg_mixvpr_dot_product_tokyo_xs_test.json',
]

print("="*80)
print("详细分析异常文件")
print("="*80)

results_dir = Path("results/image_matching")
checkpoint_dir = Path("checkpoints/image_matching")

# 按VPR实验分组
vpr_groups = defaultdict(list)

for filename in abnormal_files:
    result_file = results_dir / filename
    if not result_file.exists():
        print(f"\n❌ 文件不存在: {filename}")
        continue
    
    # 解析文件名
    parts = filename.replace('.json', '').split('_')
    if len(parts) >= 4:
        matcher = parts[0]
        vpr_method = parts[1]
        distance = parts[2]
        dataset = '_'.join(parts[3:])
        
        vpr_exp_key = f"{vpr_method}_{distance}_{dataset}"
        
        # 查找VPR日志时间戳
        vpr_log_base = Path(f"logs/baseline/{vpr_exp_key}")
        timestamp = None
        if vpr_log_base.exists():
            timestamp_dirs = sorted([d for d in vpr_log_base.iterdir() if d.is_dir()])
            if timestamp_dirs:
                timestamp = timestamp_dirs[-1].name
        
        vpr_groups[vpr_exp_key].append({
            'matcher': matcher,
            'filename': filename,
            'timestamp': timestamp,
            'result_file': result_file
        })

# 分析每个VPR实验组
print(f"\n发现 {len(vpr_groups)} 个VPR实验组有异常文件\n")

for vpr_exp_key, exps in vpr_groups.items():
    print("="*80)
    print(f"VPR实验: {vpr_exp_key}")
    print("="*80)
    
    timestamp = exps[0]['timestamp']
    
    # 检查checkpoint
    if timestamp:
        old_checkpoint = checkpoint_dir / f"{timestamp}_checkpoint.pkl"
        has_old_checkpoint = old_checkpoint.exists()
        
        if has_old_checkpoint:
            print(f"\n⚠️  [HIGH RISK] 发现旧格式checkpoint: {old_checkpoint.name}")
            print(f"   这些matcher可能共享了checkpoint，存在冲突风险！")
        else:
            print(f"\n✅ 没有旧格式checkpoint")
    
    # 分析每个matcher的结果
    print(f"\n有 {len(exps)} 个matcher的结果异常:")
    
    for exp in exps:
        result_file = exp['result_file']
        size_mb = result_file.stat().st_size / (1024*1024)
        
        try:
            with open(str(result_file), 'r') as f:
                data = json.load(f)
            
            results = data['results']
            inliers = results['num_inliers']
            is_correct = results['is_correct']
            query_ids = results['query_ids']
            
            correct_inliers = [inliers[i] for i in range(len(inliers)) if is_correct[i]]
            incorrect_inliers = [inliers[i] for i in range(len(inliers)) if not is_correct[i]]
            
            if len(correct_inliers) == 0 or len(incorrect_inliers) == 0:
                print(f"\n  ❌ {exp['matcher']}: {exp['filename']}")
                print(f"     大小: {size_mb:.1f}MB")
                print(f"     查询数: {len(query_ids)}")
                print(f"     匹配数: {len(inliers)}")
                print(f"     正确预测: {len(correct_inliers)}, 错误预测: {len(incorrect_inliers)}")
                print(f"     ⚠️  数据不足，无法判断数据分布")
            else:
                correct_mean = sum(correct_inliers) / len(correct_inliers)
                incorrect_mean = sum(incorrect_inliers) / len(incorrect_inliers)
                ratio = correct_mean / max(incorrect_mean, 1)
                
                print(f"\n  ⚠️  {exp['matcher']}: {exp['filename']}")
                print(f"     大小: {size_mb:.1f}MB")
                print(f"     查询数: {len(query_ids)}, 匹配数: {len(inliers)}")
                print(f"     正确预测: {len(correct_inliers)}, 平均inliers: {correct_mean:.1f}")
                print(f"     错误预测: {len(incorrect_inliers)}, 平均inliers: {incorrect_mean:.1f}")
                print(f"     比率: {ratio:.2f}x (异常，< 1.5x)")
                
                # 检查是否所有matcher都异常
                if ratio < 1.0:
                    print(f"     ❌ 比率 < 1.0，说明错误预测的inliers反而更高！")
                    print(f"     ❌ 这强烈暗示数据可能混乱或有问题")
        
        except Exception as e:
            print(f"\n  ❌ {exp['matcher']}: 无法读取 - {e}")
    
    # 检查是否有其他matcher的结果（正常情况）
    all_matchers = ['superglue', 'loftr', 'superpoint-lg']
    missing_matchers = [m for m in all_matchers if m not in [e['matcher'] for e in exps]]
    
    if missing_matchers:
        print(f"\n  其他matcher的结果:")
        for matcher in missing_matchers:
            other_file = results_dir / f"{matcher}_{vpr_exp_key}.json"
            if other_file.exists():
                try:
                    with open(str(other_file), 'r') as f:
                        data = json.load(f)
                    results = data['results']
                    inliers = results['num_inliers']
                    is_correct = results['is_correct']
                    
                    correct_inliers = [inliers[i] for i in range(len(inliers)) if is_correct[i]]
                    incorrect_inliers = [inliers[i] for i in range(len(inliers)) if not is_correct[i]]
                    
                    if len(correct_inliers) > 0 and len(incorrect_inliers) > 0:
                        correct_mean = sum(correct_inliers) / len(correct_inliers)
                        incorrect_mean = sum(incorrect_inliers) / len(incorrect_inliers)
                        ratio = correct_mean / max(incorrect_mean, 1)
                        
                        if ratio > 1.5:
                            print(f"      ✅ {matcher}: 比率 {ratio:.2f}x (正常)")
                        else:
                            print(f"      ⚠️  {matcher}: 比率 {ratio:.2f}x (也异常)")
                except:
                    print(f"      ❌ {matcher}: 无法读取")
            else:
                print(f"      ❌ {matcher}: 文件不存在")

# 总结和建议
print("\n" + "="*80)
print("总结和建议")
print("="*80)

print("\n发现的问题:")
print("1. 所有异常文件都来自 Tokyo-XS 数据集")
print("2. 这些文件的正确/错误预测inliers比率 < 1.5x")
print("3. 部分文件比率甚至 < 1.0（错误预测的inliers反而更高）")

print("\n可能的原因:")
print("1. Tokyo-XS数据集本身比较困难，导致正确和错误预测差异不大")
print("2. 存在checkpoint冲突，导致数据混乱")
print("3. 数据文件本身有问题")

print("\n建议:")
print("1. 检查这些实验是否使用了旧格式checkpoint（见上面的分析）")
print("2. 如果使用了旧格式checkpoint，建议重新运行这些实验")
print("3. 如果所有Tokyo-XS的结果都异常，可能是数据集本身的问题")
print("4. 可以对比其他数据集（如SF-XS, SVOX）的结果，看是否只有Tokyo-XS异常")
