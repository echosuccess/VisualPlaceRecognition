#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Image Matching results
- Summarize inlier statistics for all experiments
- Analyze correlation between inliers and VPR prediction correctness
- Generate visualizations and tables
Output files:
results/image_matching/inliers_distribution.png - inlier distribution plot
results/image_matching/discrimination_ratio.png - discrimination ratio plot
results/image_matching/table.tex - LaTeX table
Usage:
For Section 5.2 report: use --exclude-val (48 experiments)
For full analysis or Extension 6.1: use --include-all or run directly (60 experiments)
"""

import json
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def load_all_results(results_dir="results/image_matching", exclude_val=False):
    """Load all Image Matching results
    
    Args:
        results_dir: results directory path
        exclude_val: if True, exclude sfxs_val dataset (analyze only test datasets, for Section 5.2)
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        return {}
    
    all_results = {}
    result_files = list(results_dir.glob("*.json"))
    
    # Filter out summary files
    result_files = [f for f in result_files if 'summary' not in f.name]
    
    original_count = len(result_files)
    
    # If exclude_val is True, exclude sfxs_val dataset
    if exclude_val:
        # Check which files contain sfxs_val (debug info)
        val_files = [f for f in result_files if 'sfxs_val' in f.name]
        if val_files:
            print(f"[DEBUG] Found {len(val_files)} files with 'sfxs_val' in name:")
            for vf in val_files[:3]:
                print(f"  - {vf.name}")
            if len(val_files) > 3:
                print(f"  ... and {len(val_files) - 3} more")
        else:
            print(f"[DEBUG] No files with 'sfxs_val' found (all files are test datasets)")
        
        result_files = [f for f in result_files if 'sfxs_val' not in f.name]
        excluded_count = original_count - len(result_files)
        if excluded_count > 0:
            print(f"[INFO] Excluding {excluded_count} sfxs_val files (Section 5.2 analysis)")
            print(f"[INFO] Original files: {original_count}, After filtering: {len(result_files)}")
        else:
            print(f"[INFO] No sfxs_val files to exclude (all {original_count} files are test datasets)")
    
    print(f"Found {len(result_files)} result files")
    print(f"\nChecking each file...")
    print(f"{'='*80}")
    
    for result_file in result_files:
        # Parse filename: matcher_vpr_distance_dataset.json
        # Example: superglue_megaloc_dot_product_sf_xs_test.json
        parts = result_file.stem.split('_')
        
        if len(parts) >= 4:
            matcher = parts[0]
            vpr_method = parts[1]
            # distance could be dot_product (2 words) or l2 (1 word)
            if parts[2] == 'dot':
                distance = 'dot_product'
                dataset = '_'.join(parts[4:])
            else:
                distance = parts[2]
                dataset = '_'.join(parts[3:])
            
            print(f"\nFile: {result_file.name}")
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'results' not in data or 'analysis' not in data:
                    raise ValueError("Missing required keys: 'results' or 'analysis'")
                
                key = (matcher, vpr_method, dataset)
                all_results[key] = data
                print(f"  [OK] Successfully loaded")
                
            except json.JSONDecodeError as e:
                print(f"  [ERROR] JSON decode error!")
                print(f"     Error: {e}")
                print(f"     File: {result_file}")
                print(f"     Line: {e.lineno if hasattr(e, 'lineno') else 'unknown'}, Column: {e.colno if hasattr(e, 'colno') else 'unknown'}")
                print(f"  [INFO] Attempting to recover from pickle file...")
                
                pkl_file = result_file.with_suffix('.pkl')
                if pkl_file.exists():
                    print(f"     Found pickle: {pkl_file.name}")
                    try:
                        with open(pkl_file, 'rb') as f:
                            pkl_data = pickle.load(f)
                        
                        data = {
                            'results': pkl_data['results'],
                            'analysis': pkl_data['analysis']
                        }
                        
                        try:
                            backup_file = result_file.with_suffix('.json.bak')
                            if result_file.exists():
                                import shutil
                                shutil.copy2(result_file, backup_file)
                                print(f"     Backed up corrupted file to: {backup_file.name}")
                            
                            with open(result_file, 'w', encoding='utf-8') as f:
                                json.dump(data, f, indent=2, ensure_ascii=False)
                            print(f"  [OK] Successfully recovered and saved!")
                            
                            key = (matcher, vpr_method, dataset)
                            all_results[key] = data
                        except Exception as save_error:
                            print(f"  [WARN] Could not save recovered JSON: {save_error}")
                            print(f"     Using pickle data directly...")
                            key = (matcher, vpr_method, dataset)
                            all_results[key] = data
                    except Exception as pkl_error:
                        print(f"  [ERROR] Failed to load pickle: {pkl_error}")
                        print(f"  [SKIP] Skipping {result_file.name}")
                else:
                    print(f"  [INFO] Checking for pickle file: {pkl_file.name}")
                    print(f"     Expected path: {pkl_file}")
                    
                    pkl_dir = pkl_file.parent
                    all_pkl_files = list(pkl_dir.glob("*.pkl"))
                    matching_pkl = None
                    
                    json_stem = result_file.stem
                    for pkl in all_pkl_files:
                        if pkl.stem == json_stem:
                            matching_pkl = pkl
                            break
                    
                    if matching_pkl and matching_pkl.exists():
                        print(f"  [OK] Found matching pickle: {matching_pkl.name}")
                        pkl_file = matching_pkl
                        try:
                            with open(pkl_file, 'rb') as f:
                                pkl_data = pickle.load(f)
                            
                            data = {
                                'results': pkl_data['results'],
                                'analysis': pkl_data['analysis']
                            }
                            
                            backup_file = result_file.with_suffix('.json.bak')
                            if result_file.exists():
                                import shutil
                                shutil.copy2(result_file, backup_file)
                                print(f"     Backed up corrupted file to: {backup_file.name}")
                            
                            with open(result_file, 'w', encoding='utf-8') as f:
                                json.dump(data, f, indent=2, ensure_ascii=False)
                            print(f"  [OK] Successfully recovered and saved!")
                            
                            key = (matcher, vpr_method, dataset)
                            all_results[key] = data
                        except Exception as recover_error:
                            print(f"  [ERROR] Failed to recover from {matching_pkl.name}: {recover_error}")
                            print(f"  [SKIP] Skipping {result_file.name}")
                    else:
                        print(f"  [ERROR] No pickle file found")
                        if all_pkl_files:
                            print(f"     Available pickle files in directory:")
                            for pkl in all_pkl_files[:5]:
                                print(f"       - {pkl.name}")
                            if len(all_pkl_files) > 5:
                                print(f"       ... and {len(all_pkl_files) - 5} more")
                        print(f"  [SKIP] Cannot recover, skipping {result_file.name}")
                    
            except Exception as e:
                print(f"  [ERROR] Failed to load {result_file.name}")
                print(f"     Error type: {type(e).__name__}")
                print(f"     Error message: {e}")
                print(f"  [SKIP] Skipping this file")
        else:
            print(f"  [WARN] Cannot parse filename: {result_file.name}")
    
    print(f"\n{'='*80}")
    print(f"Summary: Loaded {len(all_results)}/{len(result_files)} experiments")
    if len(all_results) < len(result_files):
        print(f"[WARN] {len(result_files) - len(all_results)} file(s) were skipped due to errors")
    print(f"{'='*80}\n")
    return all_results


def print_inliers_summary(all_results):
    """Print inlier statistics summary"""
    print(f"\n{'='*100}")
    print("Inliers vs Prediction Correctness - Summary")
    print(f"{'='*100}\n")
    
    # Table header
    print(f"{'Matcher':<15} {'VPR Method':<12} {'Dataset':<20} {'Correct (mean)':<15} {'Incorrect (mean)':<17} {'Ratio':<10}")
    print('-'*100)
    
    # Group by matcher
    for (matcher, vpr_method, dataset), data in sorted(all_results.items()):
        analysis = data['analysis']
        
        correct_mean = analysis['correct_mean_inliers']
        incorrect_mean = analysis['incorrect_mean_inliers']
        ratio = analysis.get('ratio', 0)
        
        print(f"{matcher:<15} {vpr_method:<12} {dataset:<20} {correct_mean:>13.2f}  {incorrect_mean:>15.2f}  {ratio:>8.2f}x")
    
    print('='*100 + '\n')


def analyze_by_matcher(all_results):
    """Analyze by Image Matching method"""
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
    """Analyze by VPR method"""
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
    """Create visualization plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Creating Visualizations")
    print(f"{'='*80}\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Inliers Distribution: Correct vs Incorrect Predictions', fontsize=16)
    
    matchers = ['superglue', 'loftr', 'superpoint-lg']
    
    for idx, matcher in enumerate(matchers):
        if idx >= len(axes.flat):
            break
        
        ax = axes.flat[idx]
        
        correct_inliers = []
        incorrect_inliers = []
        
        for (m, vpr, dataset), data in all_results.items():
            if m == matcher:
                results = data['results']
                is_correct = np.array(results['is_correct'])
                num_inliers = np.array(results['num_inliers'])
                
                correct_inliers.extend(num_inliers[is_correct])
                incorrect_inliers.extend(num_inliers[~is_correct])
        
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
    """Generate LaTeX table"""
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Image Matching results")
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/image_matching',
        help='Results directory path (default: results/image_matching)'
    )
    parser.add_argument(
        '--exclude-val',
        action='store_true',
        help='Exclude sfxs_val dataset, analyze only test datasets (for Section 5.2, 48 experiments)'
    )
    parser.add_argument(
        '--include-all',
        action='store_true',
        help='Analyze all files (including sfxs_val, 60 experiments)'
    )
    args = parser.parse_args()
    
    exclude_val = args.exclude_val
    if args.include_all:
        exclude_val = False
    
    print(f"\n{'='*80}")
    print("Image Matching Results Analysis")
    print(f"Results directory: {args.results_dir}")
    print(f"[DEBUG] args.exclude_val = {args.exclude_val}")
    print(f"[DEBUG] args.include_all = {args.include_all}")
    print(f"[DEBUG] Final exclude_val = {exclude_val}")
    if exclude_val:
        print("Mode: Section 5.2 (Test datasets only, excluding sfxs_val)")
    else:
        print("Mode: All datasets (including sfxs_val)")
    print(f"{'='*80}\n")
    
    all_results = load_all_results(results_dir=args.results_dir, exclude_val=exclude_val)
    
    if not all_results:
        print("[ERROR] No results found!")
        return
    
    print(f"\n[INFO] Loaded {len(all_results)} experiments for analysis")
    
    print_inliers_summary(all_results)
    
    analyze_by_matcher(all_results)
    analyze_by_vpr_method(all_results)
    
    output_dir = Path(args.results_dir)
    create_visualization(all_results, output_dir=str(output_dir))
    
    table_path = output_dir / "table.tex"
    generate_latex_table(all_results, output_path=str(table_path))
    
    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}\n")
    print("Generated files:")
    print(f"  - {output_dir / 'inliers_distribution.png'}")
    print(f"  - {output_dir / 'discrimination_ratio.png'}")
    print(f"  - {table_path}")
    print("\nNext steps:")
    print("  1. Review the visualizations and tables")
    print("  2. Write conclusions about inliers vs prediction correctness")
    print("  3. Start Extension 6.1 (if required)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
