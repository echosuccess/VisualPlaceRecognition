#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析Image Matching结果
- 汇总所有实验的inliers统计
- 分析inliers与VPR预测正确性的关系
- 生成可视化和表格
"""

import json
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def load_all_results(results_dir="results/image_matching"):
    """加载所有Image Matching结果"""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        return {}
    
    all_results = {}
    result_files = list(results_dir.glob("*.json"))
    
    # 过滤掉summary文件
    result_files = [f for f in result_files if 'summary' not in f.name]
    
    print(f"Found {len(result_files)} result files")
    
    for result_file in result_files:
        # 解析文件名: matcher_vpr_distance_dataset.json
        # 例如: superglue_megaloc_dot_product_sf_xs_test.json
        parts = result_file.stem.split('_')
        
        if len(parts) >= 4:
            matcher = parts[0]
            vpr_method = parts[1]
            # distance可能是dot_product (2个词) 或 l2 (1个词)
            if parts[2] == 'dot':
                distance = 'dot_product'
                dataset = '_'.join(parts[4:])
            else:
                distance = parts[2]
                dataset = '_'.join(parts[3:])
            
            # 尝试加载JSON文件
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 验证数据结构
                if 'results' not in data or 'analysis' not in data:
                    raise ValueError("Missing required keys: 'results' or 'analysis'")
                
                key = (matcher, vpr_method, dataset)
                all_results[key] = data
                print(f"  [OK] Loaded: {result_file.name}")
                
            except json.JSONDecodeError as e:
                print(f"  [ERROR] JSON decode error in {result_file.name}: {e}")
                print(f"  [INFO] Attempting to recover from pickle file...")
                
                # 尝试从pickle文件恢复
                pkl_file = result_file.with_suffix('.pkl')
                if pkl_file.exists():
                    try:
                        with open(pkl_file, 'rb') as f:
                            pkl_data = pickle.load(f)
                        
                        # 转换为JSON格式
                        data = {
                            'results': pkl_data['results'],
                            'analysis': pkl_data['analysis']
                        }
                        
                        # 尝试重新保存为JSON
                        try:
                            with open(result_file, 'w', encoding='utf-8') as f:
                                json.dump(data, f, indent=2, ensure_ascii=False)
                            print(f"  [OK] Recovered and saved: {result_file.name}")
                            
                            key = (matcher, vpr_method, dataset)
                            all_results[key] = data
                        except Exception as save_error:
                            print(f"  [WARN] Could not save recovered JSON: {save_error}")
                            print(f"  [INFO] Using pickle data directly...")
                            # 使用pickle数据，但需要转换格式
                            key = (matcher, vpr_method, dataset)
                            all_results[key] = data
                    except Exception as pkl_error:
                        print(f"  [ERROR] Failed to load pickle: {pkl_error}")
                        print(f"  [SKIP] Skipping {result_file.name}")
                else:
                    print(f"  [SKIP] No pickle file found, skipping {result_file.name}")
                    
            except Exception as e:
                print(f"  [ERROR] Failed to load {result_file.name}: {e}")
                print(f"  [SKIP] Skipping this file")
        else:
            print(f"  [WARN] Cannot parse filename: {result_file.name}")
    
    print(f"Loaded {len(all_results)} experiments\n")
    return all_results


def print_inliers_summary(all_results):
    """打印inliers统计摘要"""
    print(f"\n{'='*100}")
    print("Inliers vs Prediction Correctness - Summary")
    print(f"{'='*100}\n")
    
    # 表头
    print(f"{'Matcher':<15} {'VPR Method':<12} {'Dataset':<20} {'Correct (mean)':<15} {'Incorrect (mean)':<17} {'Ratio':<10}")
    print('-'*100)
    
    # 按matcher分组
    for (matcher, vpr_method, dataset), data in sorted(all_results.items()):
        analysis = data['analysis']
        
        correct_mean = analysis['correct_mean_inliers']
        incorrect_mean = analysis['incorrect_mean_inliers']
        ratio = analysis.get('ratio', 0)
        
        print(f"{matcher:<15} {vpr_method:<12} {dataset:<20} {correct_mean:>13.2f}  {incorrect_mean:>15.2f}  {ratio:>8.2f}x")
    
    print('='*100 + '\n')


def analyze_by_matcher(all_results):
    """按Image Matching方法分析"""
    print(f"\n{'='*80}")
    print("Analysis by Image Matching Method")
    print(f"{'='*80}\n")
    
    matcher_stats = defaultdict(lambda: {'correct': [], 'incorrect': [], 'ratios': []})
    
    for (matcher, vpr_method, dataset), data in all_results.items():
        analysis = data['analysis']
        
        matcher_stats[matcher]['correct'].append(analysis['correct_mean_inliers'])
        matcher_stats[matcher]['incorrect'].append(analysis['incorrect_mean_inliers'])
        if 'ratio' in analysis:
            matcher_stats[matcher]['ratios'].append(analysis['ratio'])
    
    for matcher, stats in sorted(matcher_stats.items()):
        avg_correct = np.mean(stats['correct'])
        avg_incorrect = np.mean(stats['incorrect'])
        avg_ratio = np.mean(stats['ratios']) if stats['ratios'] else 0
        
        print(f"\n{matcher.upper()}")
        print(f"  Avg inliers (correct):   {avg_correct:.2f}")
        print(f"  Avg inliers (incorrect): {avg_incorrect:.2f}")
        print(f"  Avg ratio (correct/incorrect): {avg_ratio:.2f}x")
        
        if avg_ratio > 1.5:
            print(f"  [GOOD] Can distinguish correct from incorrect predictions!")
        elif avg_ratio > 1.2:
            print(f"  [OK] Some distinction capability")
        else:
            print(f"  [POOR] Limited distinction capability")
    
    print(f"\n{'='*80}\n")


def analyze_by_vpr_method(all_results):
    """按VPR方法分析"""
    print(f"\n{'='*80}")
    print("Analysis by VPR Method")
    print(f"{'='*80}\n")
    
    vpr_stats = defaultdict(lambda: {'correct': [], 'incorrect': [], 'ratios': []})
    
    for (matcher, vpr_method, dataset), data in all_results.items():
        analysis = data['analysis']
        
        vpr_stats[vpr_method]['correct'].append(analysis['correct_mean_inliers'])
        vpr_stats[vpr_method]['incorrect'].append(analysis['incorrect_mean_inliers'])
        if 'ratio' in analysis:
            vpr_stats[vpr_method]['ratios'].append(analysis['ratio'])
    
    for vpr_method, stats in sorted(vpr_stats.items()):
        avg_correct = np.mean(stats['correct'])
        avg_incorrect = np.mean(stats['incorrect'])
        avg_ratio = np.mean(stats['ratios']) if stats['ratios'] else 0
        
        print(f"\n{vpr_method.upper()}")
        print(f"  Avg inliers (correct):   {avg_correct:.2f}")
        print(f"  Avg inliers (incorrect): {avg_incorrect:.2f}")
        print(f"  Avg ratio: {avg_ratio:.2f}x")


def create_visualization(all_results, output_dir="results/image_matching"):
    """创建可视化图表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Creating Visualizations")
    print(f"{'='*80}\n")
    
    # 1. Inliers分布对比图（正确 vs 错误预测）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Inliers Distribution: Correct vs Incorrect Predictions', fontsize=16)
    
    matchers = ['superglue', 'loftr', 'superpoint-lg']
    
    for idx, matcher in enumerate(matchers):
        if idx >= len(axes.flat):
            break
        
        ax = axes.flat[idx]
        
        # 收集该matcher的所有数据
        correct_inliers = []
        incorrect_inliers = []
        
        for (m, vpr, dataset), data in all_results.items():
            if m == matcher:
                results = data['results']
                is_correct = np.array(results['is_correct'])
                num_inliers = np.array(results['num_inliers'])
                
                correct_inliers.extend(num_inliers[is_correct])
                incorrect_inliers.extend(num_inliers[~is_correct])
        
        # 绘制直方图
        if correct_inliers and incorrect_inliers:
            ax.hist(correct_inliers, bins=50, alpha=0.6, label='Correct', color='green', density=True)
            ax.hist(incorrect_inliers, bins=50, alpha=0.6, label='Incorrect', color='red', density=True)
            ax.set_xlabel('Number of Inliers')
            ax.set_ylabel('Density')
            ax.set_title(f'{matcher.upper()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_path = output_dir / 'inliers_distribution.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] Visualization: {viz_path}")
    plt.close()
    
    # 2. Ratio对比图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    matchers_list = []
    vpr_methods_list = []
    ratios_list = []
    
    for (matcher, vpr_method, dataset), data in all_results.items():
        analysis = data['analysis']
        if 'ratio' in analysis:
            matchers_list.append(matcher)
            vpr_methods_list.append(f"{vpr_method}")
            ratios_list.append(analysis['ratio'])
    
    # 按matcher分组
    unique_matchers = list(set(matchers_list))
    x = np.arange(len(unique_matchers))
    
    avg_ratios = []
    for matcher in unique_matchers:
        indices = [i for i, m in enumerate(matchers_list) if m == matcher]
        avg_ratio = np.mean([ratios_list[i] for i in indices])
        avg_ratios.append(avg_ratio)
    
    bars = ax.bar(x, avg_ratios, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_xlabel('Image Matching Method')
    ax.set_ylabel('Average Ratio (Correct/Incorrect Inliers)')
    ax.set_title('Discrimination Power of Image Matching Methods')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in unique_matchers])
    ax.axhline(y=1.0, color='red', linestyle='--', label='No discrimination (ratio=1)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, ratio in zip(bars, avg_ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.2f}x',
                ha='center', va='bottom')
    
    plt.tight_layout()
    ratio_path = output_dir / 'discrimination_ratio.png'
    plt.savefig(ratio_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] Visualization: {ratio_path}")
    plt.close()


def generate_latex_table(all_results, output_path="results/image_matching/table.tex"):
    """生成LaTeX表格"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Image Matching Results: Inliers vs Prediction Correctness}")
    latex.append("\\label{tab:image_matching}")
    latex.append("\\begin{tabular}{llllcc}")
    latex.append("\\hline")
    latex.append("Matcher & VPR Method & Dataset & Correct (mean) & Incorrect (mean) & Ratio \\\\")
    latex.append("\\hline")
    
    for (matcher, vpr_method, dataset), data in sorted(all_results.items()):
        analysis = data['analysis']
        
        correct_mean = analysis['correct_mean_inliers']
        incorrect_mean = analysis['incorrect_mean_inliers']
        ratio = analysis.get('ratio', 0)
        
        latex.append(f"{matcher} & {vpr_method} & {dataset} & {correct_mean:.2f} & {incorrect_mean:.2f} & {ratio:.2f} \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"[SAVED] LaTeX table: {output_path}")


def main():
    print(f"\n{'='*80}")
    print("Image Matching Results Analysis")
    print(f"{'='*80}\n")
    
    # 1. 加载所有结果
    all_results = load_all_results()
    
    if not all_results:
        print("[ERROR] No results found!")
        return
    
    # 2. 打印摘要
    print_inliers_summary(all_results)
    
    # 3. 按方法分析
    analyze_by_matcher(all_results)
    analyze_by_vpr_method(all_results)
    
    # 4. 创建可视化
    create_visualization(all_results)
    
    # 5. 生成LaTeX表格
    generate_latex_table(all_results)
    
    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}\n")
    print("Generated files:")
    print("  - results/image_matching/inliers_distribution.png")
    print("  - results/image_matching/discrimination_ratio.png")
    print("  - results/image_matching/table.tex")
    print("\nNext steps:")
    print("  1. Review the visualizations and tables")
    print("  2. Write conclusions about inliers vs prediction correctness")
    print("  3. Start Extension 6.1 (if required)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
