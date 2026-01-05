#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze all Baseline experiment results
Collect Recall data and generate comparison tables
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import json

def parse_info_log(log_path):
    """Parse info.log file and extract Recall metrics"""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        recalls = {}
        for n in [1, 5, 10, 20]:
            pattern = rf'R@{n}:\s*([\d.]+)'
            match = re.search(pattern, content)
            if match:
                recalls[f'R@{n}'] = float(match.group(1))
        
        time_pattern = r'Processing time:\s*([\d.]+)\s*s'
        time_match = re.search(time_pattern, content)
        processing_time = float(time_match.group(1)) if time_match else None
        
        return recalls, processing_time
    except Exception as e:
        print(f"  [WARN] Parse failed: {e}")
        return {}, None


def collect_all_results(baseline_dir="logs/baseline"):
    """Collect all experiment results"""
    baseline_path = Path(baseline_dir)
    
    if not baseline_path.exists():
        print(f"[ERROR] Directory not found: {baseline_dir}")
        return {}
    
    results = defaultdict(dict)
    
    print("\n" + "="*70)
    print("[*] Collecting experiment results...")
    print("="*70 + "\n")
    
    exp_dirs = sorted([d for d in baseline_path.iterdir() if d.is_dir()])
    
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        
        parts = exp_name.rsplit('_', 3)
        if len(parts) >= 4:
            method = parts[0]
            distance = parts[1]
            dataset = '_'.join(parts[2:])
        else:
            print(f"  [WARN] Cannot parse experiment name: {exp_name}")
            continue
        
        timestamp_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
        if not timestamp_dirs:
            print(f"  [WARN] No timestamp directory found: {exp_name}")
            continue
        
        latest_dir = timestamp_dirs[-1]
        info_log = latest_dir / "info.log"
        
        if not info_log.exists():
            print(f"  [WARN] info.log does not exist: {exp_name}")
            continue
        
        recalls, proc_time = parse_info_log(info_log)
        
        if recalls:
            key = (method, dataset)
            results[key][distance] = {
                'recalls': recalls,
                'time': proc_time,
                'path': str(latest_dir)
            }
            
            r1 = recalls.get('R@1', 0)
            r5 = recalls.get('R@5', 0)
            print(f"[OK] {method:10s} | {distance:12s} | {dataset:20s} | R@1={r1:5.2f}% R@5={r5:5.2f}%")
        else:
            print(f"[ERROR] {method:10s} | {distance:12s} | {dataset:20s} | Parse failed")
    
    return results


def generate_comparison_table(results):
    """Generate comparison table"""
    
    print("\n" + "="*70)
    print("L2 vs Dot Product Comparison Table")
    print("="*70 + "\n")
    
    methods = ['cosplace', 'netvlad', 'mixvpr', 'megaloc']
    datasets = ['sf_xs_test', 'tokyo_xs_test', 'svox_sun_test', 'svox_night_test']
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"{method.upper()}")
        print('='*70)
        
        print(f"\n{'Dataset':<20} | {'Metric':<6} | {'L2':>8} | {'Dot Prod':>8} | {'Winner':>10}")
        print('-'*70)
        
        method_wins = {'l2': 0, 'dot_product': 0}
        
        for dataset in datasets:
            key = (method, dataset)
            if key not in results:
                continue
            
            data = results[key]
            
            l2_data = data.get('l2', {})
            dot_data = data.get('dot_product', {})
            
            l2_recalls = l2_data.get('recalls', {})
            dot_recalls = dot_data.get('recalls', {})
            
            for metric in ['R@1', 'R@5', 'R@10', 'R@20']:
                l2_val = l2_recalls.get(metric, 0)
                dot_val = dot_recalls.get(metric, 0)
                
                if abs(l2_val - dot_val) < 0.1:
                    winner = 'Tie'
                elif l2_val > dot_val:
                    winner = 'L2'
                    if metric == 'R@1':
                        method_wins['l2'] += 1
                else:
                    winner = 'Dot'
                    if metric == 'R@1':
                        method_wins['dot_product'] += 1
                
                print(f"{dataset:<20} | {metric:<6} | {l2_val:7.2f}% | {dot_val:7.2f}% | {winner:>10}")
        
        print(f"\n{'='*70}")
        print(f"{method.upper()} Summary: L2 won {method_wins['l2']} datasets, Dot Product won {method_wins['dot_product']} datasets")
        print('='*70)


def generate_latex_table(results):
    """Generate LaTeX table"""
    
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
    """Generate summary report"""
    
    print("\n" + "="*70)
    print("Overall Analysis")
    print("="*70 + "\n")
    
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
    print(f"R@1 Win Statistics (total {total} datasets):")
    print(f"  L2 better: {overall_wins['l2']} ({overall_wins['l2']/total*100:.1f}%)")
    print(f"  Dot Product better: {overall_wins['dot_product']} ({overall_wins['dot_product']/total*100:.1f}%)")
    print(f"  Tie: {overall_wins['tie']} ({overall_wins['tie']/total*100:.1f}%)")
    
    print(f"\nBest Method Ranking (by average R@1):")
    method_scores = defaultdict(list)
    
    for key, data in results.items():
        method, dataset = key
        for distance, dist_data in data.items():
            r1 = dist_data['recalls'].get('R@1', 0)
            method_scores[f"{method}_{distance}"].append(r1)
    
    method_avg = {k: sum(v)/len(v) for k, v in method_scores.items()}
    sorted_methods = sorted(method_avg.items(), key=lambda x: x[1], reverse=True)
    
    for i, (method_dist, avg_r1) in enumerate(sorted_methods, 1):
        print(f"  {i}. {method_dist:<25} : {avg_r1:5.2f}%")
    
    print(f"\nRecommendation:")
    best_method = sorted_methods[0][0]
    print(f"  Best method: {best_method}")
    
    if overall_wins['dot_product'] > overall_wins['l2']:
        print(f"  Recommended distance metric: Dot Product (performs better on {overall_wins['dot_product']}/{total} datasets)")
    elif overall_wins['l2'] > overall_wins['dot_product']:
        print(f"  Recommended distance metric: L2 (performs better on {overall_wins['l2']}/{total} datasets)")
    else:
        print(f"  Both distance metrics perform similarly, choose based on specific scenario")


def save_results_json(results, output_file="results/baseline_summary.json"):
    """Save results as JSON"""
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    
    serializable = {}
    for (method, dataset), data in results.items():
        key = f"{method}_{dataset}"
        serializable[key] = data
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Results saved: {output_file}")


def main():
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*18 + "Baseline Results Analysis Tool" + " "*20 + "║")
    print("╚" + "═"*68 + "╝")
    
    results = collect_all_results()
    
    if not results:
        print("\n[ERROR] No experiment results found!")
        return
    
    generate_comparison_table(results)
    
    latex_table = generate_latex_table(results)
    latex_file = "results/baseline_table.tex"
    Path(latex_file).parent.mkdir(exist_ok=True, parents=True)
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    print(f"\n[OK] LaTeX table saved: {latex_file}")
    
    generate_summary_report(results)
    
    save_results_json(results)
    
    print("\n" + "="*70)
    print("[OK] Analysis completed!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review comparison table, select best distance metric")
    print("  2. Record your selection rationale (why choose L2 or Dot Product?)")
    print("  3. Run Image Matching methods (SuperGlue, LoFTR, etc.)")
    print("  4. Start Extension 6.1 (if required)")
    print("\n[INFO] Tip: Check results/baseline_summary.json for complete data\n")


if __name__ == "__main__":
    main()
