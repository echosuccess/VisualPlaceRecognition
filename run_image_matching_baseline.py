#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline Task: Run Image Matching methods
- Run Image Matching on VPR top-K predictions
- Count inliers
- Analyze correlation between inliers and prediction correctness
"""

import torch
import numpy as np
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import argparse
import sys

# Add image-matching-models to path
sys.path.insert(0, str(Path(__file__).parent / 'image-matching-models'))

from matching import get_matcher


def load_vpr_predictions(log_dir):
    """Load VPR prediction results"""
    data_file = log_dir / "z_data.torch"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    data = torch.load(data_file, map_location='cpu', weights_only=False)
    
    # VPR data contains:
    # - predictions: (num_queries, K) - top-K database indices
    # - distances: (num_queries, K) - corresponding distances/similarities
    # - positives_per_query: list of lists - all correct matches per query
    # - database_utms: database UTM coordinates
    
    # Infer ground_truth from positives_per_query (use first positive)
    if 'positives_per_query' in data:
        ground_truth = torch.tensor([pos[0] if len(pos) > 0 else -1 
                                     for pos in data['positives_per_query']])
        data['ground_truth'] = ground_truth
    
    return data


def run_image_matching(matcher, query_path, database_path, device='cuda'):
    """
    Run Image Matching and return inlier count
    
    Returns:
        num_inliers: int - number of inliers after RANSAC
    """
    try:
        img0 = matcher.load_image(str(query_path), resize=512)
        img1 = matcher.load_image(str(database_path), resize=512)
        
        result = matcher(img0, img1)
        
        return result['num_inliers']
    
    except Exception as e:
        print(f"  [ERROR] Matching failed for {query_path.name} <-> {database_path.name}: {e}")
        return 0


def process_vpr_experiment(vpr_log_dir, matcher, database_folder, queries_folder, top_k=20, device='cuda', matcher_name=None):
    """
    Process a VPR experiment, run Image Matching for all queries
    
    Args:
        vpr_log_dir: VPR experiment log directory
        matcher: Image Matching model
        database_folder: database image folder path
        queries_folder: queries image folder path
        top_k: match top-K predictions
        device: device
    
    Returns:
        results: dict containing all matching results
    """
    print(f"\n{'='*80}")
    print(f"Processing VPR experiment: {vpr_log_dir.name}")
    print(f"{'='*80}")
    
    # Load VPR predictions
    vpr_data = load_vpr_predictions(vpr_log_dir)
    
    predictions = vpr_data['predictions']  # (num_queries, K)
    ground_truth = vpr_data['ground_truth']  # (num_queries,)
    
    # Get image paths from file system
    database_folder_path = Path(database_folder)
    queries_folder_path = Path(queries_folder)
    
    # Check if folders exist
    if not database_folder_path.exists():
        raise FileNotFoundError(f"Database folder not found: {database_folder_path}")
    if not queries_folder_path.exists():
        raise FileNotFoundError(f"Queries folder not found: {queries_folder_path}")
    
    # Try multiple image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    database_paths = []
    queries_paths = []
    
    print(f"[DEBUG] Searching for images in:")
    print(f"  Database: {database_folder_path.absolute()}")
    print(f"  Queries: {queries_folder_path.absolute()}")
    print(f"  Database exists: {database_folder_path.exists()}")
    print(f"  Queries exists: {queries_folder_path.exists()}")
    
    # Check what files are actually in the directory
    if queries_folder_path.exists():
        all_items = list(queries_folder_path.iterdir())
        files_only = [f for f in all_items if f.is_file()]
        print(f"[DEBUG] Queries directory contains {len(files_only)} files (total items: {len(all_items)})")
        if files_only:
            sample_files = files_only[:3]
            print(f"[DEBUG] Sample files in queries:")
            for f in sample_files:
                print(f"    {f.name} (suffix: '{f.suffix}')")
    
    # Try direct glob search
    for ext in image_extensions:
        db_files = list(database_folder_path.glob(ext))
        q_files = list(queries_folder_path.glob(ext))
        if db_files:
            print(f"[DEBUG] Found {len(db_files)} files with extension {ext} in database")
        if q_files:
            print(f"[DEBUG] Found {len(q_files)} files with extension {ext} in queries")
        database_paths.extend(sorted(db_files))
        queries_paths.extend(sorted(q_files))
    
    # If direct search found nothing, try listing all files and filtering
    if len(queries_paths) == 0 and queries_folder_path.exists():
        print(f"[DEBUG] Direct glob search found 0 queries, trying alternative method...")
        all_files = list(queries_folder_path.iterdir())
        # Filter out files (not directories)
        files = [f for f in all_files if f.is_file()]
        # Check extensions (including case variants)
        valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = [f for f in files if f.suffix in valid_extensions]
        if image_files:
            queries_paths = sorted(set(image_files))
            print(f"[DEBUG] Found {len(queries_paths)} query files using alternative method")
            if queries_paths:
                print(f"[DEBUG] First query file: {queries_paths[0]}")
    
    if len(database_paths) == 0 and database_folder_path.exists():
        print(f"[DEBUG] Direct glob search found 0 database images, trying alternative method...")
        all_files = list(database_folder_path.iterdir())
        files = [f for f in all_files if f.is_file()]
        valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = [f for f in files if f.suffix in valid_extensions]
        if image_files:
            database_paths = sorted(set(image_files))
            print(f"[DEBUG] Found {len(database_paths)} database files using alternative method")
    
    # Remove duplicates and sort
    database_paths = sorted(set(database_paths))
    queries_paths = sorted(set(queries_paths))
    
    num_queries = len(queries_paths)
    num_database = len(database_paths)
    K = min(top_k, predictions.shape[1])
    
    print(f"Loaded {num_queries} queries, {num_database} database images")
    print(f"Database folder: {database_folder_path} (exists: {database_folder_path.exists()})")
    print(f"Queries folder: {queries_folder_path} (exists: {queries_folder_path.exists()})")
    
    # Check index range in VPR predictions
    print(f"\n[DEBUG] VPR predictions shape: {predictions.shape}")
    print(f"[DEBUG] Ground truth shape: {ground_truth.shape}")
    if len(predictions) > 0:
        max_pred_idx = int(predictions.max().item())
        min_pred_idx = int(predictions.min().item())
        print(f"[DEBUG] Prediction indices range: {min_pred_idx} to {max_pred_idx}")
        print(f"[DEBUG] Database paths available: 0 to {num_database - 1}")
        
        if max_pred_idx >= num_database:
            print(f"\n[WARN] VPR predictions contain indices up to {max_pred_idx}, but only {num_database} database images loaded!")
            print(f"[WARN] This indicates a mismatch between VPR experiment and current database files.")
            
            invalid_count = 0
            total_checked = 0
            sample_invalid = []
            
            check_sample = min(100, len(predictions))
            for q_idx in range(check_sample):
                for rank in range(predictions.shape[1]):
                    total_checked += 1
                    pred_idx = int(predictions[q_idx, rank].item())
                    if pred_idx >= num_database:
                        invalid_count += 1
                        if len(sample_invalid) < 5:
                            sample_invalid.append((q_idx, rank, pred_idx))
            
            if invalid_count > 0:
                invalid_ratio = invalid_count / total_checked
                print(f"[WARN] Found {invalid_count}/{total_checked} ({invalid_ratio*100:.1f}%) invalid indices in sample")
                print(f"[WARN] Example invalid indices: {sample_invalid[:3]}")
                
                if invalid_ratio > 0.1:
                    print(f"\n[ERROR] Too many invalid prediction indices ({invalid_ratio*100:.1f}%)!")
                    print(f"[ERROR] This suggests the database files don't match the VPR experiment.")
                    print(f"[ERROR] Options:")
                    print(f"  1. Re-run VPR experiment with the current database files")
                    print(f"  2. Check if database folder path is correct")
                    print(f"  3. Continue anyway (will skip invalid predictions, but results may be incomplete)")
                    print(f"\n[INFO] Continuing with warnings - invalid predictions will be skipped...")
                else:
                    print(f"[WARN] Some predictions will be skipped, but most are valid.")
    
    if num_queries == 0:
        print(f"[DEBUG] Trying recursive search...")
        all_files = list(queries_folder_path.rglob("*"))
        image_files = [f for f in all_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        print(f"[DEBUG] Found {len(image_files)} image files recursively")
        if image_files:
            print(f"[DEBUG] First few files: {image_files[:5]}")
            queries_paths = sorted(set(image_files))
            num_queries = len(queries_paths)
            print(f"[INFO] Using {num_queries} query images found via recursive search")
        
        if len(database_paths) == 0:
            print(f"[DEBUG] Trying recursive search for database images...")
            all_db_files = list(database_folder_path.rglob("*"))
            db_image_files = [f for f in all_db_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            print(f"[DEBUG] Found {len(db_image_files)} database image files recursively")
            if db_image_files:
                database_paths = sorted(set(db_image_files))
                print(f"[INFO] Using {len(database_paths)} database images found via recursive search")
        
        if num_queries == 0:
            raise ValueError(
                f"No query images found in {queries_folder_path}!\n"
                f"Absolute path: {queries_folder_path.absolute()}\n"
                f"Please check:\n"
                f"  1. The folder path is correct\n"
                f"  2. The folder contains image files (.jpg, .jpeg, .png)\n"
                f"  3. The dataset name matches the actual folder structure\n"
                f"  4. Recursive search found {len(image_files)} image files"
            )
    
    if len(database_paths) == 0:
        raise ValueError(
            f"No database images found in {database_folder_path}!\n"
            f"Please check the folder path and contents."
        )
    
    print(f"Will match top-{K} predictions per query")
    
    # Run matching for top-K predictions of each query
    results = {
        'query_ids': [],
        'pred_ranks': [],  # prediction rank (0 means top-1)
        'is_correct': [],  # whether prediction is correct
        'num_inliers': [],  # number of inliers
        'query_paths': [],
        'database_paths': []
    }
    
    skipped_predictions = 0
    
    # Checkpoint path for periodic saving
    checkpoint_dir = Path("checkpoints/image_matching")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if matcher_name is None:
        matcher_name = str(type(matcher).__name__).lower().replace('matcher', '').replace('_', '-')
    
    vpr_log_dir_path = Path(vpr_log_dir) if not isinstance(vpr_log_dir, Path) else vpr_log_dir
    vpr_exp_name = vpr_log_dir_path.parent.name
    timestamp = vpr_log_dir_path.name
    
    if not matcher_name or not vpr_exp_name or not timestamp:
        raise ValueError(
            f"Checkpoint naming failed: matcher_name={matcher_name}, "
            f"vpr_exp_name={vpr_exp_name}, timestamp={timestamp}, "
            f"vpr_log_dir={vpr_log_dir}"
        )
    
    checkpoint_path = checkpoint_dir / f"{matcher_name}_{vpr_exp_name}_{timestamp}_checkpoint.pkl"
    
    # Try to resume from checkpoint
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
    
    # Checkpoint interval (save every 1% of queries)
    checkpoint_interval = max(1, num_queries // 100)
    
    for q_idx in tqdm(range(start_idx, num_queries), desc="Matching queries", initial=start_idx, total=num_queries):
        gt_idx = ground_truth[q_idx].item()
        query_path = Path(queries_paths[q_idx])
        
        # Run matching for top-K predictions
        for rank in range(K):
            pred_idx = int(predictions[q_idx, rank].item())
            
            # Check if index is valid
            if pred_idx >= len(database_paths):
                skipped_predictions += 1
                if skipped_predictions <= 5 or skipped_predictions % 100 == 0:
                    print(f"\n[WARN] Query {q_idx}, rank {rank}: pred_idx {pred_idx} >= database size {len(database_paths)}, skipping")
                continue
            
            if pred_idx < 0:
                skipped_predictions += 1
                if skipped_predictions <= 5:
                    print(f"\n[WARN] Query {q_idx}, rank {rank}: pred_idx {pred_idx} is negative, skipping")
                continue
            
            database_path = Path(database_paths[pred_idx])
            
            is_correct = (pred_idx == gt_idx)
            
            num_inliers = run_image_matching(
                matcher, 
                query_path, 
                database_path, 
                device=device
            )
            
            results['query_ids'].append(int(q_idx))
            results['pred_ranks'].append(int(rank))
            results['is_correct'].append(bool(is_correct))
            results['num_inliers'].append(int(num_inliers))
            results['query_paths'].append(str(query_path))
            results['database_paths'].append(str(database_path))
        
        # Periodically save checkpoint
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
    
    # Remove checkpoint after completion
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            print(f"[INFO] Removed checkpoint file (computation completed)")
        except:
            pass
    
    # Report skipped predictions
    if skipped_predictions > 0:
        total_predictions = num_queries * K
        skip_ratio = skipped_predictions / total_predictions
        print(f"\n[WARN] Skipped {skipped_predictions}/{total_predictions} ({skip_ratio*100:.1f}%) predictions due to invalid indices")
        print(f"[WARN] This may affect the completeness of results.")
        print(f"[WARN] Consider re-running VPR experiment if database files don't match.")
    
    return results


def analyze_inliers_correlation(results):
    """
    Analyze correlation between inlier count and prediction correctness
    
    Args:
        results: matching results dictionary
    
    Returns:
        analysis: analysis results
    """
    if not results or len(results.get('num_inliers', [])) == 0:
        print("[WARN] No results to analyze!")
        return {
            'num_matches': 0,
            'num_correct': 0,
            'num_incorrect': 0,
            'correct_mean_inliers': 0,
            'correct_median_inliers': 0,
            'correct_std_inliers': 0,
            'incorrect_mean_inliers': 0,
            'incorrect_median_inliers': 0,
            'incorrect_std_inliers': 0,
        }
    
    inliers = np.array(results['num_inliers'], dtype=np.float64)
    is_correct = np.array(results['is_correct'], dtype=bool)
    
    if len(inliers) != len(is_correct):
        raise ValueError(f"Array length mismatch: inliers={len(inliers)}, is_correct={len(is_correct)}")
    
    correct_inliers = inliers[is_correct] if len(inliers) > 0 else np.array([])
    incorrect_inliers = inliers[~is_correct] if len(inliers) > 0 else np.array([])
    
    analysis = {
        'num_matches': len(inliers),
        'num_correct': int(np.sum(is_correct)),
        'num_incorrect': int(np.sum(~is_correct)),
        
        'correct_mean_inliers': float(np.mean(correct_inliers)) if len(correct_inliers) > 0 else 0,
        'correct_median_inliers': float(np.median(correct_inliers)) if len(correct_inliers) > 0 else 0,
        'correct_std_inliers': float(np.std(correct_inliers)) if len(correct_inliers) > 0 else 0,
        
        'incorrect_mean_inliers': float(np.mean(incorrect_inliers)) if len(incorrect_inliers) > 0 else 0,
        'incorrect_median_inliers': float(np.median(incorrect_inliers)) if len(incorrect_inliers) > 0 else 0,
        'incorrect_std_inliers': float(np.std(incorrect_inliers)) if len(incorrect_inliers) > 0 else 0,
    }
    
    if len(correct_inliers) > 0 and len(incorrect_inliers) > 0:
        analysis['difference'] = analysis['correct_mean_inliers'] - analysis['incorrect_mean_inliers']
        analysis['ratio'] = analysis['correct_mean_inliers'] / max(analysis['incorrect_mean_inliers'], 1)
    
    return analysis


def convert_to_python_types(obj):
    """Recursively convert NumPy types to Python native types for JSON serialization"""
    # Handle NumPy scalar types
    # NumPy 2.0 compatible: use type checking instead of potentially removed aliases
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
    """Recursively validate if object can be JSON serialized"""
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
            json.dumps(obj)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot serialize object at {path}: {type(obj).__name__} - {e}")


def test_json_serialization(data):
    """Test if data can be JSON serialized"""
    try:
        json_str = json.dumps(data, indent=2)
        json.loads(json_str)
        return True, None
    except Exception as e:
        return False, str(e)


def save_results(results, analysis, output_path):
    """Save results with validation and error handling"""
    print(f"\n[INFO] Preparing to save results to: {output_path}")
    
    output = {
        'results': results,
        'analysis': analysis
    }
    
    print("[INFO] Converting NumPy types to Python native types...")
    try:
        output = convert_to_python_types(output)
    except Exception as e:
        print(f"[ERROR] Failed to convert types: {e}")
        raise
    
    print("[INFO] Validating data integrity...")
    try:
        validate_json_serializable(output)
    except ValueError as e:
        print(f"[ERROR] Data validation failed: {e}")
        raise
    
    print("[INFO] Testing JSON serialization...")
    success, error_msg = test_json_serialization(output)
    if not success:
        print(f"[ERROR] JSON serialization test failed: {error_msg}")
        print("[INFO] Attempting to identify problematic data...")
        try:
            validate_json_serializable(output)
        except ValueError as e:
            print(f"[ERROR] Problematic data location: {e}")
        raise ValueError(f"Cannot serialize to JSON: {error_msg}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_path = output_path.with_suffix('.json.tmp')
    print(f"[INFO] Writing to temporary file: {temp_path}")
    
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise ValueError("File was not written or is empty")
        
        with open(temp_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        assert 'results' in loaded, "Missing 'results' key"
        assert 'analysis' in loaded, "Missing 'analysis' key"
        
        temp_path.rename(output_path)
        print(f"[OK] Results saved successfully to: {output_path}")
        print(f"[INFO] File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    except Exception as e:
        if temp_path.exists():
            print(f"[WARN] Keeping temporary file for recovery: {temp_path}")
        print(f"[ERROR] Failed to save results: {e}")
        raise


def print_analysis(analysis):
    """Print analysis results"""
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
    
    # Find VPR experiment log directory
    vpr_exp_name = f"{args.vpr_method}_{args.distance}_{args.dataset}"
    vpr_log_base = Path("logs/baseline") / vpr_exp_name
    
    if not vpr_log_base.exists():
        print(f"[ERROR] VPR log directory not found: {vpr_log_base}")
        return
    
    timestamp_dirs = sorted([d for d in vpr_log_base.iterdir() if d.is_dir()])
    if not timestamp_dirs:
        print(f"[ERROR] No timestamp directory found in {vpr_log_base}")
        return
    
    vpr_log_dir = timestamp_dirs[-1]
    print(f"Using VPR log: {vpr_log_dir}")
    
    # Initialize Image Matching model
    print(f"\nLoading matcher: {args.matcher}...")
    matcher = get_matcher(args.matcher, device=args.device)
    print(f"[OK] Matcher loaded")
    
    # Build data paths based on dataset
    if 'svox' in args.dataset:
        if 'night' in args.dataset:
            database_folder = "data/svox/images/test/gallery"
            queries_folder = "data/svox/images/test/queries_night"
        else:
            database_folder = "data/svox/images/test/gallery"
            queries_folder = "data/svox/images/test/queries"
    elif args.dataset == 'sfxs_val':
        database_folder = "data/sf_xs/val/database"
        queries_folder = "data/sf_xs/val/queries"
    else:
        dataset_name = args.dataset.replace('_test', '')
        database_folder = f"data/{dataset_name}/test/database"
        queries_folder = f"data/{dataset_name}/test/queries"
    
    print(f"Database folder: {database_folder}")
    print(f"Queries folder: {queries_folder}")
    
    if not Path(database_folder).exists():
        print(f"[ERROR] Database folder does not exist: {database_folder}")
        print(f"[INFO] Please check if the dataset is downloaded and the path is correct")
        return
    if not Path(queries_folder).exists():
        print(f"[ERROR] Queries folder does not exist: {queries_folder}")
        print(f"[INFO] Please check if the dataset is downloaded and the path is correct")
        return
    
    output_path = Path(args.output_dir) / f"{args.matcher}_{vpr_exp_name}.json"
    if output_path.exists():
        print(f"[INFO] Result file already exists: {output_path}")
        print("[INFO] Skipping computation. Delete the file to re-run.")
        return
    
    results = process_vpr_experiment(
        vpr_log_dir=vpr_log_dir,
        matcher=matcher,
        database_folder=database_folder,
        queries_folder=queries_folder,
        top_k=args.top_k,
        device=args.device,
        matcher_name=args.matcher
    )
    
    analysis = analyze_inliers_correlation(results)
    print_analysis(analysis)
    
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
    
    # Try to save as JSON format
    print(f"\n[INFO] Attempting to save as JSON format: {output_path}")
    try:
        save_results(results, analysis, output_path)
        print(f"[OK] JSON saved successfully!")
    except Exception as e:
        print(f"\n[WARN] Failed to save JSON format: {e}")
        print(f"[INFO] BUT: Results are safely saved in pickle format: {pickle_path}")
        print(f"[INFO] You can load the pickle file to recover all data:")
        print(f"      import pickle")
        print(f"      with open('{pickle_path}', 'rb') as f:")
        print(f"          data = pickle.load(f)")
        print(f"      results = data['results']")
        print(f"      analysis = data['analysis']")
        return
    
    print(f"\n[DONE] Experiment completed!")
    print(f"Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
