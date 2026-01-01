#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析所有Baseline实验结果
收集Recall数据，生成对比表格
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
        
        # 提取处理时间（如果有）
        time_pattern = r'Processing time:\s*([\d.]+)\s*s'
        time_match = re.search(time_pattern, content)
        processing_time = float(time_match.group(1)) if time_match else None
        
        return recalls, processing_time
    except Exception as e:
        print(f"  ⚠️ 解析失败: {e}")
        return {}, None


def collect_all_results(baseline_dir="logs/baseline"):
    """收集所有实验结果"""
    baseline_path = Path(baseline_dir)
    
    if not baseline_path.exists():
        print(f"[X] Directory not found: {baseline_dir}")
        return {}
    
    results = defaultdict(dict)
    
    print("\n" + "="*70)
    print("[*] Collecting experiment results...")
    print("="*70 + "\n")
    
    # 遍历所有实验目录
    exp_dirs = sorted([d for d in baseline_path.iterdir() if d.is_dir()])
    
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        
        # 解析实验名称：method_distance_dataset
        # 例如：cosplace_l2_sf_xs_test
        parts = exp_name.rsplit('_', 3)  # 从右边分割，最多3次
        if len(parts) >= 4:
            method = parts[0]
            distance = parts[1]
            dataset = '_'.join(parts[2:])  # sf_xs_test 或 svox_night_test
        else:
            print(f"  ⚠️ 无法解析实验名: {exp_name}")
            continue
        
        # 找到最新的时间戳子目录
        timestamp_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
        if not timestamp_dirs:
            print(f"  ⚠️ 没有找到时间戳目录: {exp_name}")
            continue
        
        latest_dir = timestamp_dirs[-1]  # 最新的
        info_log = latest_dir / "info.log"
        
        if not info_log.exists():
            print(f"  ⚠️ info.log 不存在: {exp_name}")
            continue
        
        # 解析结果
        recalls, proc_time = parse_info_log(info_log)
        
        if recalls:
            # 存储结果
            key = (method, dataset)
            results[key][distance] = {
                'recalls': recalls,
                'time': proc_time,
                'path': str(latest_dir)
            }
            
            # 打印
            r1 = recalls.get('R@1', 0)
            r5 = recalls.get('R@5', 0)
            print(f"✅ {method:10s} | {distance:12s} | {dataset:20s} | R@1={r1:5.2f}% R@5={r5:5.2f}%")
        else:
            print(f"❌ {method:10s} | {distance:12s} | {dataset:20s} | 解析失败")
    
    return results


def generate_comparison_table(results):
    """生成对比表格"""
    
    print("\n" + "="*70)
    print("📋 L2 vs Dot Product 对比表")
    print("="*70 + "\n")
    
    # 按方法分组
    methods = ['cosplace', 'netvlad', 'mixvpr', 'megaloc']
    datasets = ['sf_xs_test', 'tokyo_xs_test', 'svox_sun_test', 'svox_night_test']
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"🔍 {method.upper()}")
        print('='*70)
        
        print(f"\n{'Dataset':<20} | {'Metric':<6} | {'L2':>8} | {'Dot Prod':>8} | {'Winner':>10}")
        print('-'*70)
        
        method_wins = {'l2': 0, 'dot_product': 0}
        
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
            
            # 比较每个指标
            for metric in ['R@1', 'R@5', 'R@10', 'R@20']:
                l2_val = l2_recalls.get(metric, 0)
                dot_val = dot_recalls.get(metric, 0)
                
                # 判断winner
                if abs(l2_val - dot_val) < 0.1:
                    winner = '🟡 Tie'
                elif l2_val > dot_val:
                    winner = '🔵 L2'
                    if metric == 'R@1':  # R@1最重要
                        method_wins['l2'] += 1
                else:
                    winner = '🟢 Dot'
                    if metric == 'R@1':
                        method_wins['dot_product'] += 1
                
                print(f"{dataset:<20} | {metric:<6} | {l2_val:7.2f}% | {dot_val:7.2f}% | {winner:>10}")
        
        # 方法总结
        print(f"\n{'='*70}")
        print(f"📊 {method.upper()} 总结: L2赢了{method_wins['l2']}个数据集, Dot Product赢了{method_wins['dot_product']}个数据集")
        print('='*70)


def generate_latex_table(results):
    """生成LaTeX表格"""
    
    methods = ['cosplace', 'netvlad', 'mixvpr', 'megaloc']
    datasets = ['sf_xs_test', 'tokyo_xs_test', 'svox_sun_test', 'svox_night_test']
    dataset_names = {
        'sf_xs_test': 'SF-XS',
        'tokyo_xs_test': 'Tokyo-XS',
        'svox_sun_test': 'SVOX-Sun',
        'svox_night_test': 'SVOX-Night'
    }
    
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{VPR Methods Comparison: L2 vs Dot Product}")
    latex.append("\\label{tab:vpr_comparison}")
    latex.append("\\begin{tabular}{llcccc}")
    latex.append("\\hline")
    latex.append("Method & Dataset & Distance & R@1 & R@5 & R@10 \\\\")
    latex.append("\\hline")
    
    for method in methods:
        method_name = method.replace('_', ' ').title()
        first_row = True
        
        for dataset in datasets:
            key = (method, dataset)
            if key not in results:
                continue
            
            data = results[key]
            ds_name = dataset_names.get(dataset, dataset)
            
            for distance in ['l2', 'dot_product']:
                if distance not in data:
                    continue
                
                recalls = data[distance]['recalls']
                r1 = recalls.get('R@1', 0)
                r5 = recalls.get('R@5', 0)
                r10 = recalls.get('R@10', 0)
                
                dist_name = 'L2' if distance == 'l2' else 'Dot'
                
                if first_row:
                    latex.append(f"{method_name} & {ds_name} & {dist_name} & {r1:.2f} & {r5:.2f} & {r10:.2f} \\\\")
                    first_row = False
                else:
                    latex.append(f" & {ds_name} & {dist_name} & {r1:.2f} & {r5:.2f} & {r10:.2f} \\\\")
        
        latex.append("\\hline")
    
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return '\n'.join(latex)


def generate_summary_report(results):
    """生成总结报告"""
    
    print("\n" + "="*70)
    print("📈 总体分析")
    print("="*70 + "\n")
    
    # 统计L2 vs Dot Product胜负
    overall_wins = {'l2': 0, 'dot_product': 0, 'tie': 0}
    
    for key, data in results.items():
        method, dataset = key
        
        l2_recalls = data.get('l2', {}).get('recalls', {})
        dot_recalls = data.get('dot_product', {}).get('recalls', {})
        
        l2_r1 = l2_recalls.get('R@1', 0)
        dot_r1 = dot_recalls.get('R@1', 0)
        
        if abs(l2_r1 - dot_r1) < 0.5:
            overall_wins['tie'] += 1
        elif l2_r1 > dot_r1:
            overall_wins['l2'] += 1
        else:
            overall_wins['dot_product'] += 1
    
    total = sum(overall_wins.values())
    print(f"📊 R@1 胜负统计（共{total}个数据集）：")
    print(f"  🔵 L2 更好: {overall_wins['l2']} ({overall_wins['l2']/total*100:.1f}%)")
    print(f"  🟢 Dot Product 更好: {overall_wins['dot_product']} ({overall_wins['dot_product']/total*100:.1f}%)")
    print(f"  🟡 平局: {overall_wins['tie']} ({overall_wins['tie']/total*100:.1f}%)")
    
    # 找出最佳方法
    print(f"\n📊 最佳方法排名（按平均R@1）：")
    method_scores = defaultdict(list)
    
    for key, data in results.items():
        method, dataset = key
        for distance, dist_data in data.items():
            r1 = dist_data['recalls'].get('R@1', 0)
            method_scores[f"{method}_{distance}"].append(r1)
    
    # 计算平均值并排序
    method_avg = {k: sum(v)/len(v) for k, v in method_scores.items()}
    sorted_methods = sorted(method_avg.items(), key=lambda x: x[1], reverse=True)
    
    for i, (method_dist, avg_r1) in enumerate(sorted_methods, 1):
        print(f"  {i}. {method_dist:<25} : {avg_r1:5.2f}%")
    
    # 推荐
    print(f"\n💡 推荐：")
    best_method = sorted_methods[0][0]
    print(f"  最佳方法: {best_method}")
    
    if overall_wins['dot_product'] > overall_wins['l2']:
        print(f"  推荐距离度量: Dot Product（在{overall_wins['dot_product']}/{total}个数据集上表现更好）")
    elif overall_wins['l2'] > overall_wins['dot_product']:
        print(f"  推荐距离度量: L2（在{overall_wins['l2']}/{total}个数据集上表现更好）")
    else:
        print(f"  两种距离度量性能相近，可根据具体场景选择")


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
    
    print(f"\n💾 结果已保存: {output_file}")


def main():
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*18 + "Baseline 结果分析工具" + " "*25 + "║")
    print("╚" + "═"*68 + "╝")
    
    # 1. 收集结果
    results = collect_all_results()
    
    if not results:
        print("\n❌ 没有找到任何实验结果！")
        return
    
    # 2. 生成对比表格
    generate_comparison_table(results)
    
    # 3. 生成LaTeX表格
    latex_table = generate_latex_table(results)
    latex_file = "results/baseline_table.tex"
    Path(latex_file).parent.mkdir(exist_ok=True, parents=True)
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    print(f"\n📄 LaTeX表格已保存: {latex_file}")
    
    # 4. 生成总结报告
    generate_summary_report(results)
    
    # 5. 保存JSON
    save_results_json(results)
    
    print("\n" + "="*70)
    print("✅ 分析完成！")
    print("="*70)
    print("\n📋 接下来要做的事：")
    print("  1. 查看对比表格，选择最佳距离度量")
    print("  2. 记录你的选择理由（为什么选择L2或Dot Product？）")
    print("  3. 运行 Image Matching 方法（Superglue, LoFTR等）")
    print("  4. 开始 Extension 6.1（如果需要）")
    print("\n💡 提示：查看 results/baseline_summary.json 获取完整数据\n")


if __name__ == "__main__":
    main()
