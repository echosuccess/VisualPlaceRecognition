#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版结果分析工具（无emoji，适配Windows终端）
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import json

def parse_info_log(log_path):
    """解析info.log文件，提取Recall指标"""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取Recall@N指标
        recalls = {}
        for n in [1, 5, 10, 20]:
            pattern = rf'R@{n}:\s*([\d.]+)'
            match = re.search(pattern, content)
            if match:
                recalls[f'R@{n}'] = float(match.group(1))
        
        return recalls
    except Exception as e:
        return {}


def collect_all_results(baseline_dir="logs/baseline"):
    """收集所有实验结果"""
    baseline_path = Path(baseline_dir)
    
    if not baseline_path.exists():
        print(f"[ERROR] Directory not found: {baseline_dir}")
        return {}
    
    results = defaultdict(dict)
    
    print("\n" + "="*80)
    print("Collecting Experiment Results...")
    print("="*80 + "\n")
    
    # 遍历所有实验目录
    exp_dirs = sorted([d for d in baseline_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(exp_dirs)} experiment directories\n")
    
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        
        # 解析实验名称：method_distance_dataset
        # 例如：cosplace_l2_sf_xs_test
        # 已知的方法和距离
        methods_list = ['cosplace', 'netvlad', 'mixvpr', 'megaloc']
        distances_list = ['l2', 'dot_product']
        
        # 找出method
        method = None
        for m in methods_list:
            if exp_name.startswith(m + '_'):
                method = m
                break
        
        if not method:
            print(f"[WARN] Cannot find method in: {exp_name}")
            continue
        
        # 移除method部分
        remaining = exp_name[len(method)+1:]
        
        # 找出distance
        distance = None
        for d in distances_list:
            if remaining.startswith(d + '_'):
                distance = d
                break
        
        if not distance:
            print(f"[WARN] Cannot find distance in: {exp_name}")
            continue
        
        # 剩余的就是dataset
        dataset = remaining[len(distance)+1:]
        
        # 找到最新的时间戳子目录
        timestamp_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
        if not timestamp_dirs:
            print(f"[WARN] No timestamp directory found: {exp_name}")
            continue
        
        latest_dir = timestamp_dirs[-1]  # 最新的
        info_log = latest_dir / "info.log"
        
        if not info_log.exists():
            print(f"[WARN] info.log not found: {exp_name}")
            continue
        
        # 解析结果
        recalls = parse_info_log(info_log)
        
        if recalls:
            # 存储结果
            key = (method, dataset)
            results[key][distance] = {
                'recalls': recalls,
                'path': str(latest_dir)
            }
            
            # 打印
            r1 = recalls.get('R@1', 0)
            r5 = recalls.get('R@5', 0)
            r10 = recalls.get('R@10', 0)
            print(f"[OK] {method:10s} | {distance:12s} | {dataset:20s} | R@1={r1:6.2f} R@5={r5:6.2f} R@10={r10:6.2f}")
        else:
            print(f"[FAIL] {method:10s} | {distance:12s} | {dataset:20s} | Parse failed")
    
    print(f"\n" + "="*80)
    print(f"Collected {len(results)} method-dataset combinations")
    print("="*80 + "\n")
    
    return results


def print_comparison_table(results):
    """打印对比表格"""
    
    print("\n" + "="*80)
    print("L2 vs Dot Product Comparison")
    print("="*80 + "\n")
    
    methods = ['cosplace', 'netvlad', 'mixvpr', 'megaloc']
    datasets = ['sf_xs_test', 'tokyo_xs_test', 'svox_sun_test', 'svox_night_test']
    
    # 统计
    overall_l2_wins = 0
    overall_dot_wins = 0
    overall_ties = 0
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Method: {method.upper()}")
        print('='*80)
        
        print(f"\n{'Dataset':<25} {'Metric':<8} {'L2':>10} {'DotProd':>10} {'Winner':>12}")
        print('-'*80)
        
        for dataset in datasets:
            key = (method, dataset)
            if key not in results:
                continue
            
            data = results[key]
            
            # 获取L2和Dot product的结果
            l2_data = data.get('l2', {})
            dot_data = data.get('dot_product', {})
            
            l2_recalls = l2_data.get('recalls', {})
            dot_recalls = dot_data.get('recalls', {})
            
            # 比较R@1（最重要的指标）
            l2_r1 = l2_recalls.get('R@1', 0)
            dot_r1 = dot_recalls.get('R@1', 0)
            
            if abs(l2_r1 - dot_r1) < 0.5:
                r1_winner = 'Tie'
                overall_ties += 1
            elif l2_r1 > dot_r1:
                r1_winner = 'L2'
                overall_l2_wins += 1
            else:
                r1_winner = 'DotProduct'
                overall_dot_wins += 1
            
            # 打印R@1, R@5, R@10
            for metric in ['R@1', 'R@5', 'R@10']:
                l2_val = l2_recalls.get(metric, 0)
                dot_val = dot_recalls.get(metric, 0)
                
                if metric == 'R@1':
                    winner = r1_winner
                else:
                    winner = ''
                
                print(f"{dataset:<25} {metric:<8} {l2_val:9.2f}% {dot_val:9.2f}% {winner:>12}")
    
    # 总体统计
    total = overall_l2_wins + overall_dot_wins + overall_ties
    print(f"\n" + "="*80)
    print("OVERALL SUMMARY (Based on R@1)")
    print("="*80)
    print(f"Total comparisons: {total}")
    
    if total > 0:
        print(f"L2 wins:           {overall_l2_wins} ({overall_l2_wins/total*100:.1f}%)")
        print(f"Dot Product wins:  {overall_dot_wins} ({overall_dot_wins/total*100:.1f}%)")
        print(f"Ties:              {overall_ties} ({overall_ties/total*100:.1f}%)")
    
        if overall_dot_wins > overall_l2_wins:
            print(f"\nRECOMMENDATION: Use Dot Product (cosine similarity)")
        elif overall_l2_wins > overall_dot_wins:
            print(f"\nRECOMMENDATION: Use L2 distance")
        else:
            print(f"\nRECOMMENDATION: Both metrics perform similarly")
    else:
        print("\n[WARN] No comparisons found!")
    
    print("="*80 + "\n")


def print_method_ranking(results):
    """打印方法排名"""
    
    print("\n" + "="*80)
    print("Method Ranking (by average R@1)")
    print("="*80 + "\n")
    
    method_scores = defaultdict(list)
    
    for key, data in results.items():
        method, dataset = key
        for distance, dist_data in data.items():
            r1 = dist_data['recalls'].get('R@1', 0)
            method_scores[f"{method}_{distance}"].append(r1)
    
    # 计算平均值并排序
    method_avg = {k: sum(v)/len(v) for k, v in method_scores.items()}
    sorted_methods = sorted(method_avg.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Rank':<6} {'Method':<30} {'Avg R@1':<12} {'Datasets':<10}")
    print('-'*80)
    
    for i, (method_dist, avg_r1) in enumerate(sorted_methods, 1):
        num_datasets = len(method_scores[method_dist])
        print(f"{i:<6} {method_dist:<30} {avg_r1:10.2f}% {num_datasets:<10}")
    
    print("="*80 + "\n")


def save_results_json(results, output_file="results/baseline_summary.json"):
    """保存结果为JSON"""
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    
    # 转换为可序列化的格式
    serializable = {}
    for (method, dataset), data in results.items():
        key = f"{method}_{dataset}"
        serializable[key] = data
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    print(f"[SAVED] Results saved to: {output_file}")


def save_csv_table(results, output_file="results/baseline_table.csv"):
    """保存为CSV表格"""
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    
    methods = ['cosplace', 'netvlad', 'mixvpr', 'megaloc']
    datasets = ['sf_xs_test', 'tokyo_xs_test', 'svox_sun_test', 'svox_night_test']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write("Method,Dataset,Distance,R@1,R@5,R@10,R@20\n")
        
        # 写入数据
        for method in methods:
            for dataset in datasets:
                key = (method, dataset)
                if key not in results:
                    continue
                
                data = results[key]
                
                for distance in ['l2', 'dot_product']:
                    if distance not in data:
                        continue
                    
                    recalls = data[distance]['recalls']
                    r1 = recalls.get('R@1', 0)
                    r5 = recalls.get('R@5', 0)
                    r10 = recalls.get('R@10', 0)
                    r20 = recalls.get('R@20', 0)
                    
                    f.write(f"{method},{dataset},{distance},{r1:.2f},{r5:.2f},{r10:.2f},{r20:.2f}\n")
    
    print(f"[SAVED] CSV table saved to: {output_file}")


def main():
    print("\n" + "="*80)
    print("Baseline Results Analysis")
    print("="*80 + "\n")
    
    # 1. 收集结果
    results = collect_all_results()
    
    if not results:
        print("[ERROR] No experiment results found!")
        return
    
    # 2. 打印对比表格
    print_comparison_table(results)
    
    # 3. 打印方法排名
    print_method_ranking(results)
    
    # 4. 保存结果
    save_results_json(results)
    save_csv_table(results)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Review the comparison table and choose the best distance metric")
    print("  2. Document your choice and reasoning")
    print("  3. Run Image Matching methods (Superglue, LoFTR, etc.)")
    print("  4. Start Extension 6.1 (if required)")
    print("\nFiles generated:")
    print("  - results/baseline_summary.json")
    print("  - results/baseline_table.csv")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
