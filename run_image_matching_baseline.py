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
import pickle
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
    
    # Checkpoint路径（用于定期保存中间结果）
    checkpoint_dir = Path("checkpoints/image_matching")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{vpr_log_dir.name}_checkpoint.pkl"
    
    # 尝试从checkpoint恢复
    start_idx = 0
    if checkpoint_path.exists():
        try:
            print(f"[INFO] Found checkpoint: {checkpoint_path}")
            print("[INFO] Loading checkpoint...")
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            results = checkpoint['results']
            start_idx = checkpoint['last_query_idx'] + 1
            print(f"[INFO] Resuming from query {start_idx}/{num_queries}")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")
            print("[INFO] Starting from scratch...")
            start_idx = 0
    
    # 定期保存checkpoint的间隔（每处理10%的queries保存一次）
    checkpoint_interval = max(1, num_queries // 10)
    
    for q_idx in tqdm(range(start_idx, num_queries), desc="Matching queries", initial=start_idx, total=num_queries):
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
            
            # 记录结果（确保所有类型都是Python原生类型）
            results['query_ids'].append(int(q_idx))
            results['pred_ranks'].append(int(rank))
            results['is_correct'].append(bool(is_correct))
            results['num_inliers'].append(int(num_inliers))  # 确保转换为Python int
            results['query_paths'].append(str(query_path))
            results['database_paths'].append(str(database_path))
        
        # 定期保存checkpoint
        if (q_idx + 1) % checkpoint_interval == 0 or (q_idx + 1) == num_queries:
            try:
                checkpoint_data = {
                    'results': results,
                    'last_query_idx': q_idx,
                    'num_queries': num_queries,
                    'K': K
                }
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                print(f"\n[CHECKPOINT] Saved at query {q_idx + 1}/{num_queries}")
            except Exception as e:
                print(f"\n[WARN] Failed to save checkpoint: {e}")
    
    # 计算完成后，删除checkpoint（因为已经完成）
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            print(f"[INFO] Removed checkpoint file (computation completed)")
        except:
            pass
    
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
    # 处理NumPy标量类型
    # NumPy 2.0兼容：使用类型检查而不是可能被移除的别名
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
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


def validate_json_serializable(obj, path=""):
    """递归验证对象是否可以JSON序列化"""
    try:
        if isinstance(obj, dict):
            for key, value in obj.items():
                validate_json_serializable(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                validate_json_serializable(item, f"{path}[{i}]")
        elif isinstance(obj, tuple):
            for i, item in enumerate(obj):
                validate_json_serializable(item, f"{path}[{i}]")
        else:
            # 尝试序列化单个值
            json.dumps(obj)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot serialize object at {path}: {type(obj).__name__} - {e}")


def test_json_serialization(data):
    """测试数据是否可以JSON序列化"""
    try:
        # 先尝试序列化到字符串
        json_str = json.dumps(data, indent=2)
        # 再尝试反序列化验证
        json.loads(json_str)
        return True, None
    except Exception as e:
        return False, str(e)


def save_results(results, analysis, output_path):
    """保存结果，带完整的验证和错误处理"""
    print(f"\n[INFO] Preparing to save results to: {output_path}")
    
    # 1. 构建输出数据
    output = {
        'results': results,
        'analysis': analysis
    }
    
    # 2. 转换所有NumPy类型为Python原生类型
    print("[INFO] Converting NumPy types to Python native types...")
    try:
        output = convert_to_python_types(output)
    except Exception as e:
        print(f"[ERROR] Failed to convert types: {e}")
        raise
    
    # 3. 验证数据完整性
    print("[INFO] Validating data integrity...")
    try:
        validate_json_serializable(output)
    except ValueError as e:
        print(f"[ERROR] Data validation failed: {e}")
        raise
    
    # 4. 测试JSON序列化（不实际写入文件）
    print("[INFO] Testing JSON serialization...")
    success, error_msg = test_json_serialization(output)
    if not success:
        print(f"[ERROR] JSON serialization test failed: {error_msg}")
        print("[INFO] Attempting to identify problematic data...")
        # 尝试找出问题数据
        try:
            validate_json_serializable(output)
        except ValueError as e:
            print(f"[ERROR] Problematic data location: {e}")
        raise ValueError(f"Cannot serialize to JSON: {error_msg}")
    
    # 5. 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 6. 实际保存（先写入临时文件，成功后再重命名）
    temp_path = output_path.with_suffix('.json.tmp')
    print(f"[INFO] Writing to temporary file: {temp_path}")
    
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        # 验证文件是否成功写入
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise ValueError("File was not written or is empty")
        
        # 验证文件可以读取
        with open(temp_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        # 验证加载的数据结构
        assert 'results' in loaded, "Missing 'results' key"
        assert 'analysis' in loaded, "Missing 'analysis' key"
        
        # 重命名为正式文件
        temp_path.rename(output_path)
        print(f"[OK] Results saved successfully to: {output_path}")
        print(f"[INFO] File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
    except Exception as e:
        # 如果失败，清理临时文件
        if temp_path.exists():
            print(f"[WARN] Keeping temporary file for recovery: {temp_path}")
        print(f"[ERROR] Failed to save results: {e}")
        raise


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
    
    # 检查是否已有结果文件
    output_path = Path(args.output_dir) / f"{args.matcher}_{vpr_exp_name}.json"
    if output_path.exists():
        print(f"[INFO] Result file already exists: {output_path}")
        print("[INFO] Skipping computation. Delete the file to re-run.")
        return
    
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
    
    # 5. 先保存为pickle格式（更可靠，即使JSON失败也能恢复）
    pickle_path = output_path.with_suffix('.pkl')
    print(f"\n[INFO] Saving results to pickle format first (more reliable): {pickle_path}")
    try:
        pickle_data = {
            'results': results,
            'analysis': analysis,
            'metadata': {
                'matcher': args.matcher,
                'vpr_method': args.vpr_method,
                'dataset': args.dataset,
                'distance': args.distance,
                'top_k': args.top_k
            }
        }
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pickle_path, 'wb') as f:
            pickle.dump(pickle_data, f)
        print(f"[OK] Results saved to pickle: {pickle_path}")
        print(f"[INFO] File size: {pickle_path.stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"[ERROR] Failed to save pickle: {e}")
        raise
    
    # 6. 然后尝试保存为JSON格式（用于分析脚本）
    print(f"\n[INFO] Attempting to save as JSON format: {output_path}")
    try:
        save_results(results, analysis, output_path)
        print(f"[OK] JSON saved successfully!")
    except Exception as e:
        # JSON保存失败，但pickle已保存，所以结果不会丢失
        print(f"\n[WARN] Failed to save JSON format: {e}")
        print(f"[INFO] BUT: Results are safely saved in pickle format: {pickle_path}")
        print(f"[INFO] You can load the pickle file to recover all data:")
        print(f"      import pickle")
        print(f"      with open('{pickle_path}', 'rb') as f:")
        print(f"          data = pickle.load(f)")
        print(f"      results = data['results']")
        print(f"      analysis = data['analysis']")
        # 不抛出异常，因为pickle已成功保存
        return
        # 如果保存失败，尝试保存基本信息以便恢复
        print(f"\n[ERROR] Failed to save results: {e}")
        print("[INFO] Attempting to save minimal recovery data...")
        
        # 保存基本信息（不包含完整results，只保存analysis）
        recovery_path = output_path.with_suffix('.recovery.json')
        try:
            recovery_data = {
                'analysis': convert_to_python_types(analysis),
                'error': str(e),
                'num_results': len(results.get('query_ids', [])),
                'timestamp': str(Path().cwd())
            }
            with open(recovery_path, 'w', encoding='utf-8') as f:
                json.dump(recovery_data, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Recovery data saved to: {recovery_path}")
        except Exception as recovery_error:
            print(f"[WARN] Could not save recovery data: {recovery_error}")
        
        # 检查是否有临时文件
        temp_path = output_path.with_suffix('.json.tmp')
        if temp_path.exists():
            print(f"[INFO] Temporary file exists at: {temp_path}")
            print("[INFO] You can try to manually recover data from the temp file.")
        
        raise
    
    print(f"\n[DONE] Experiment completed!")
    print(f"Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
