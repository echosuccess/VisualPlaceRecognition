#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline Task: 运行Image Matching方法
- 对VPR的top-K预测运行Image Matching
- 计算inliers数量
- 分析inliers与预测正确性的关系
"""

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import sys

# 添加image-matching-models到路径
sys.path.insert(0, str(Path(__file__).parent / 'image-matching-models'))

from matching import get_matcher


def load_vpr_predictions(log_dir):
    """加载VPR预测结果"""
    data_file = log_dir / "z_data.torch"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    data = torch.load(data_file, map_location='cpu', weights_only=False)
    
    # VPR保存的data包含：
    # - predictions: (num_queries, K) - top-K预测的database索引  
    # - distances: (num_queries, K) - 对应的距离/相似度
    # - positives_per_query: list of lists - 每个query的所有正确匹配
    # - database_utms: database的UTM坐标
    
    # 需要从positives_per_query推断ground_truth（取第一个positive）
    if 'positives_per_query' in data:
        ground_truth = torch.tensor([pos[0] if len(pos) > 0 else -1 
                                     for pos in data['positives_per_query']])
        data['ground_truth'] = ground_truth
    
    return data


def run_image_matching(matcher, query_path, database_path, device='cuda'):
    """
    运行Image Matching并返回inliers数量
    
    Returns:
        num_inliers: int - RANSAC后的inliers数量
    """
    try:
        # 加载图像
        img0 = matcher.load_image(str(query_path), resize=512)
        img1 = matcher.load_image(str(database_path), resize=512)
        
        # 运行匹配
        result = matcher(img0, img1)
        
        # 返回inliers数量
        return result['num_inliers']
    
    except Exception as e:
        print(f"  [ERROR] Matching failed for {query_path.name} <-> {database_path.name}: {e}")
        return 0


def process_vpr_experiment(vpr_log_dir, matcher, database_folder, queries_folder, top_k=20, device='cuda'):
    """
    处理一个VPR实验，对所有query运行Image Matching
    
    Args:
        vpr_log_dir: VPR实验日志目录
        matcher: Image Matching模型
        database_folder: database图像文件夹路径
        queries_folder: queries图像文件夹路径
        top_k: 对前K个预测运行匹配
        device: 设备
    
    Returns:
        results: dict包含所有匹配结果
    """
    print(f"\n{'='*80}")
    print(f"Processing VPR experiment: {vpr_log_dir.name}")
    print(f"{'='*80}")
    
    # 1. 加载VPR预测
    vpr_data = load_vpr_predictions(vpr_log_dir)
    
    predictions = vpr_data['predictions']  # (num_queries, K)
    ground_truth = vpr_data['ground_truth']  # (num_queries,)
    
    # 2. 从文件系统获取图像路径
    database_paths = sorted(Path(database_folder).glob("*.jpg"))
    queries_paths = sorted(Path(queries_folder).glob("*.jpg"))
    
    num_queries = len(queries_paths)
    K = min(top_k, predictions.shape[1])
    
    print(f"Loaded {num_queries} queries, {len(database_paths)} database images")
    print(f"Will match top-{K} predictions per query")
    
    # 2. 对每个query的top-K预测运行匹配
    results = {
        'query_ids': [],
        'pred_ranks': [],  # 预测的rank (0表示top-1)
        'is_correct': [],  # 预测是否正确
        'num_inliers': [],  # inliers数量
        'query_paths': [],
        'database_paths': []
    }
    
    for q_idx in tqdm(range(num_queries), desc="Matching queries"):
        gt_idx = ground_truth[q_idx].item()
        query_path = Path(queries_paths[q_idx])
        
        # 对top-K预测运行匹配
        for rank in range(K):
            pred_idx = predictions[q_idx, rank].item()
            database_path = Path(database_paths[pred_idx])
            
            # 判断预测是否正确
            is_correct = (pred_idx == gt_idx)
            
            # 运行Image Matching
            num_inliers = run_image_matching(
                matcher, 
                query_path, 
                database_path, 
                device=device
            )
            
            # 记录结果
            results['query_ids'].append(q_idx)
            results['pred_ranks'].append(rank)
            results['is_correct'].append(is_correct)
            results['num_inliers'].append(num_inliers)
            results['query_paths'].append(str(query_path))
            results['database_paths'].append(str(database_path))
    
    return results


def analyze_inliers_correlation(results):
    """
    分析inliers数量与预测正确性的关系
    
    Args:
        results: 匹配结果字典
    
    Returns:
        analysis: 分析结果
    """
    inliers = np.array(results['num_inliers'])
    is_correct = np.array(results['is_correct'])
    
    # 分为正确预测和错误预测
    correct_inliers = inliers[is_correct]
    incorrect_inliers = inliers[~is_correct]
    
    analysis = {
        'num_matches': len(inliers),
        'num_correct': int(np.sum(is_correct)),
        'num_incorrect': int(np.sum(~is_correct)),
        
        # 正确预测的统计
        'correct_mean_inliers': float(np.mean(correct_inliers)) if len(correct_inliers) > 0 else 0,
        'correct_median_inliers': float(np.median(correct_inliers)) if len(correct_inliers) > 0 else 0,
        'correct_std_inliers': float(np.std(correct_inliers)) if len(correct_inliers) > 0 else 0,
        
        # 错误预测的统计
        'incorrect_mean_inliers': float(np.mean(incorrect_inliers)) if len(incorrect_inliers) > 0 else 0,
        'incorrect_median_inliers': float(np.median(incorrect_inliers)) if len(incorrect_inliers) > 0 else 0,
        'incorrect_std_inliers': float(np.std(incorrect_inliers)) if len(incorrect_inliers) > 0 else 0,
    }
    
    # 计算差异显著性
    if len(correct_inliers) > 0 and len(incorrect_inliers) > 0:
        analysis['difference'] = analysis['correct_mean_inliers'] - analysis['incorrect_mean_inliers']
        analysis['ratio'] = analysis['correct_mean_inliers'] / max(analysis['incorrect_mean_inliers'], 1)
    
    return analysis


def convert_to_python_types(obj):
    """递归地将NumPy类型转换为Python原生类型，以便JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_python_types(item) for item in obj)
    else:
        return obj


def save_results(results, analysis, output_path):
    """保存结果"""
    output = {
        'results': results,
        'analysis': analysis
    }
    
    # 转换所有NumPy类型为Python原生类型
    output = convert_to_python_types(output)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[SAVED] Results saved to: {output_path}")


def print_analysis(analysis):
    """打印分析结果"""
    print(f"\n{'='*80}")
    print("Analysis: Inliers vs Prediction Correctness")
    print(f"{'='*80}")
    print(f"Total matches: {analysis['num_matches']}")
    print(f"Correct predictions: {analysis['num_correct']}")
    print(f"Incorrect predictions: {analysis['num_incorrect']}")
    print(f"\n--- Correct Predictions ---")
    print(f"Mean inliers:   {analysis['correct_mean_inliers']:.2f}")
    print(f"Median inliers: {analysis['correct_median_inliers']:.2f}")
    print(f"Std inliers:    {analysis['correct_std_inliers']:.2f}")
    print(f"\n--- Incorrect Predictions ---")
    print(f"Mean inliers:   {analysis['incorrect_mean_inliers']:.2f}")
    print(f"Median inliers: {analysis['incorrect_median_inliers']:.2f}")
    print(f"Std inliers:    {analysis['incorrect_std_inliers']:.2f}")
    
    if 'difference' in analysis:
        print(f"\n--- Difference ---")
        print(f"Mean difference: {analysis['difference']:.2f}")
        print(f"Ratio (correct/incorrect): {analysis['ratio']:.2f}x")
        
        if analysis['ratio'] > 1.5:
            print("\n[CONCLUSION] Correct predictions have significantly MORE inliers!")
            print("Inliers can be used as a reliability indicator.")
        elif analysis['ratio'] < 0.67:
            print("\n[CONCLUSION] Incorrect predictions have MORE inliers (unexpected!)") 
        else:
            print("\n[CONCLUSION] Inliers difference is not significant.")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Run Image Matching on VPR predictions')
    parser.add_argument('--matcher', type=str, default='superglue',
                        choices=['superglue', 'loftr', 'superpoint-lg'],
                        help='Image matching method')
    parser.add_argument('--vpr_method', type=str, default='megaloc',
                        choices=['cosplace', 'netvlad', 'mixvpr', 'megaloc'],
                        help='VPR method to analyze')
    parser.add_argument('--dataset', type=str, default='sf_xs_test',
                        help='Dataset name (e.g., sf_xs_test)')
    parser.add_argument('--distance', type=str, default='dot_product',
                        choices=['l2', 'dot_product'],
                        help='Distance metric used in VPR')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Match top-K predictions per query')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--output_dir', type=str, default='results/image_matching',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("Image Matching Baseline Experiment")
    print(f"{'='*80}")
    print(f"Matcher: {args.matcher}")
    print(f"VPR Method: {args.vpr_method}")
    print(f"Dataset: {args.dataset}")
    print(f"Distance Metric: {args.distance}")
    print(f"Top-K: {args.top_k}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")
    
    # 1. 找到VPR实验日志目录
    vpr_exp_name = f"{args.vpr_method}_{args.distance}_{args.dataset}"
    vpr_log_base = Path("logs/baseline") / vpr_exp_name
    
    if not vpr_log_base.exists():
        print(f"[ERROR] VPR log directory not found: {vpr_log_base}")
        return
    
    # 找到最新的时间戳目录
    timestamp_dirs = sorted([d for d in vpr_log_base.iterdir() if d.is_dir()])
    if not timestamp_dirs:
        print(f"[ERROR] No timestamp directory found in {vpr_log_base}")
        return
    
    vpr_log_dir = timestamp_dirs[-1]
    print(f"Using VPR log: {vpr_log_dir}")
    
    # 2. 初始化Image Matching模型
    print(f"\nLoading matcher: {args.matcher}...")
    matcher = get_matcher(args.matcher, device=args.device)
    print(f"[OK] Matcher loaded")
    
    # 3. 构建数据路径（根据VPR方法和数据集）
    # 根据dataset参数推断database和queries路径
    dataset_base = f"data/{args.dataset.replace('_test', '')}"
    if 'svox' in args.dataset:
        if 'night' in args.dataset:
            database_folder = f"{dataset_base}/images/test/gallery"
            queries_folder = f"{dataset_base}/images/test/queries_night"
        else:  # sun
            database_folder = f"{dataset_base}/images/test/gallery"
            queries_folder = f"{dataset_base}/images/test/queries"
    else:
        database_folder = f"{dataset_base}/test/database"
        queries_folder = f"{dataset_base}/test/queries"
    
    print(f"Database folder: {database_folder}")
    print(f"Queries folder: {queries_folder}")
    
    # 运行Image Matching
    results = process_vpr_experiment(
        vpr_log_dir=vpr_log_dir,
        matcher=matcher,
        database_folder=database_folder,
        queries_folder=queries_folder,
        top_k=args.top_k,
        device=args.device
    )
    
    # 4. 分析结果
    analysis = analyze_inliers_correlation(results)
    print_analysis(analysis)
    
    # 5. 保存结果
    output_path = Path(args.output_dir) / f"{args.matcher}_{vpr_exp_name}.json"
    save_results(results, analysis, output_path)
    
    print(f"\n[DONE] Experiment completed!")
    print(f"Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
