#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 VPR Baseline 实验的数据完整性
检查：
1. 实际加载的图像数量
2. VPR 预测中使用的图像数量
3. 是否有不匹配
"""

from pathlib import Path
from collections import defaultdict

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not available, will only check file system")

def count_images_in_folder(folder_path, recursive=False):
    """统计文件夹中的图像数量"""
    folder = Path(folder_path)
    if not folder.exists():
        return 0, []
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    
    if recursive:
        for ext in image_extensions:
            image_files.extend(folder.rglob(f'*{ext}'))
    else:
        for ext in image_extensions:
            image_files.extend(folder.glob(f'*{ext}'))
    
    # 去重并排序
    image_files = sorted(set(image_files))
    return len(image_files), image_files

def load_vpr_log(log_dir):
    """加载 VPR 日志"""
    if not TORCH_AVAILABLE:
        return None, None
    
    log_dir = Path(log_dir)
    
    # 找到最新的时间戳目录
    timestamp_dirs = sorted([d for d in log_dir.iterdir() if d.is_dir()])
    if not timestamp_dirs:
        return None, None
    
    latest_dir = timestamp_dirs[-1]
    z_data_path = latest_dir / "z_data.torch"
    
    if not z_data_path.exists():
        return None, None
    
    try:
        data = torch.load(z_data_path, map_location='cpu', weights_only=False)
        return data, latest_dir
    except Exception as e:
        print(f"  [ERROR] 无法加载 z_data.torch: {e}")
        return None, None

def verify_experiment(exp_name, database_folder, queries_folder, recursive=False):
    """验证单个实验的数据完整性"""
    print(f"\n{'='*80}")
    print(f"验证实验: {exp_name}")
    print(f"{'='*80}")
    
    # 1. 统计实际图像数量
    print(f"\n[1] 统计实际图像数量...")
    num_db_images, db_files = count_images_in_folder(database_folder, recursive)
    num_query_images, query_files = count_images_in_folder(queries_folder, recursive)
    
    print(f"  Database 图像: {num_db_images}")
    print(f"  Queries 图像: {num_query_images}")
    
    # 2. 加载 VPR 日志
    log_base = Path("logs/baseline") / exp_name
    if not log_base.exists():
        print(f"\n[ERROR] VPR 日志不存在: {log_base}")
        return False
    
    vpr_data, log_dir = load_vpr_log(log_base)
    if vpr_data is None:
        print(f"\n[ERROR] 无法加载 VPR 数据")
        return False
    
    # 3. 检查 VPR 预测中的索引范围
    predictions = vpr_data.get('predictions', None)
    positives_per_query = vpr_data.get('positives_per_query', [])
    
    if predictions is None:
        print(f"\n[ERROR] VPR 数据中没有 predictions")
        return False
    
    num_queries_vpr = predictions.shape[0]
    max_pred_idx = int(predictions.max().item()) if predictions.numel() > 0 else -1
    min_pred_idx = int(predictions.min().item()) if predictions.numel() > 0 else -1
    
    print(f"\n[2] VPR 预测信息:")
    print(f"  Queries 数量: {num_queries_vpr}")
    print(f"  预测索引范围: {min_pred_idx} 到 {max_pred_idx}")
    print(f"  Database 大小（VPR 期望）: 至少 {max_pred_idx + 1}")
    
    # 4. 检查不匹配
    print(f"\n[3] 数据完整性检查:")
    
    issues = []
    
    # 检查 queries 数量
    if num_query_images != num_queries_vpr:
        issues.append(f"Queries 数量不匹配: 实际 {num_query_images} vs VPR {num_queries_vpr}")
        print(f"  [WARN] Queries 数量不匹配!")
        print(f"    实际图像: {num_query_images}")
        print(f"    VPR 预测: {num_queries_vpr}")
        print(f"    差异: {abs(num_query_images - num_queries_vpr)}")
    
    # 检查 database 大小
    if max_pred_idx >= num_db_images:
        issues.append(f"Database 索引超出范围: 最大索引 {max_pred_idx} >= 实际数量 {num_db_images}")
        print(f"  [WARN] Database 索引超出范围!")
        print(f"    实际图像: {num_db_images}")
        print(f"    最大预测索引: {max_pred_idx}")
        print(f"    超出: {max_pred_idx - num_db_images + 1} 个索引")
    
    # 检查 ground truth
    if positives_per_query:
        max_gt_idx = max([max(pos) if pos else -1 for pos in positives_per_query])
        if max_gt_idx >= num_db_images:
            issues.append(f"Ground truth 索引超出范围: 最大索引 {max_gt_idx} >= 实际数量 {num_db_images}")
            print(f"  [WARN] Ground truth 索引超出范围!")
            print(f"    最大 GT 索引: {max_gt_idx}")
    
    if not issues:
        print(f"  [OK] 数据完整性检查通过!")
        return True
    else:
        print(f"\n  [ERROR] 发现 {len(issues)} 个问题:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
        return False

def main():
    """主函数"""
    print("="*80)
    print("VPR Baseline 数据完整性验证工具")
    print("="*80)
    
    # 定义所有实验
    experiments = [
        # CosPlace
        ("cosplace_l2_sf_xs_test", "data/sf_xs/test/database", "data/sf_xs/test/queries", False),
        ("cosplace_l2_tokyo_xs_test", "data/tokyo_xs/test/database", "data/tokyo_xs/test/queries", False),
        ("cosplace_l2_svox_sun_test", "data/svox/images/test/gallery", "data/svox/images/test/queries", True),
        ("cosplace_l2_svox_night_test", "data/svox/images/test/gallery", "data/svox/images/test/queries_night", True),
        ("cosplace_dot_product_sf_xs_test", "data/sf_xs/test/database", "data/sf_xs/test/queries", False),
        ("cosplace_dot_product_tokyo_xs_test", "data/tokyo_xs/test/database", "data/tokyo_xs/test/queries", False),
        ("cosplace_dot_product_svox_sun_test", "data/svox/images/test/gallery", "data/svox/images/test/queries", True),
        ("cosplace_dot_product_svox_night_test", "data/svox/images/test/gallery", "data/svox/images/test/queries_night", True),
        
        # NetVLAD
        ("netvlad_l2_sf_xs_test", "data/sf_xs/test/database", "data/sf_xs/test/queries", False),
        ("netvlad_l2_tokyo_xs_test", "data/tokyo_xs/test/database", "data/tokyo_xs/test/queries", False),
        ("netvlad_l2_svox_sun_test", "data/svox/images/test/gallery", "data/svox/images/test/queries", True),
        ("netvlad_l2_svox_night_test", "data/svox/images/test/gallery", "data/svox/images/test/queries_night", True),
        ("netvlad_dot_product_sf_xs_test", "data/sf_xs/test/database", "data/sf_xs/test/queries", False),
        ("netvlad_dot_product_tokyo_xs_test", "data/tokyo_xs/test/database", "data/tokyo_xs/test/queries", False),
        ("netvlad_dot_product_svox_sun_test", "data/svox/images/test/gallery", "data/svox/images/test/queries", True),
        ("netvlad_dot_product_svox_night_test", "data/svox/images/test/gallery", "data/svox/images/test/queries_night", True),
        
        # MixVPR
        ("mixvpr_l2_sf_xs_test", "data/sf_xs/test/database", "data/sf_xs/test/queries", False),
        ("mixvpr_l2_tokyo_xs_test", "data/tokyo_xs/test/database", "data/tokyo_xs/test/queries", False),
        ("mixvpr_l2_svox_sun_test", "data/svox/images/test/gallery", "data/svox/images/test/queries", True),
        ("mixvpr_l2_svox_night_test", "data/svox/images/test/gallery", "data/svox/images/test/queries_night", True),
        ("mixvpr_dot_product_sf_xs_test", "data/sf_xs/test/database", "data/sf_xs/test/queries", False),
        ("mixvpr_dot_product_tokyo_xs_test", "data/tokyo_xs/test/database", "data/tokyo_xs/test/queries", False),
        ("mixvpr_dot_product_svox_sun_test", "data/svox/images/test/gallery", "data/svox/images/test/queries", True),
        ("mixvpr_dot_product_svox_night_test", "data/svox/images/test/gallery", "data/svox/images/test/queries_night", True),
        
        # MegaLoc
        ("megaloc_l2_sf_xs_test", "data/sf_xs/test/database", "data/sf_xs/test/queries", False),
        ("megaloc_l2_tokyo_xs_test", "data/tokyo_xs/test/database", "data/tokyo_xs/test/queries", False),
        ("megaloc_l2_svox_sun_test", "data/svox/images/test/gallery", "data/svox/images/test/queries", True),
        ("megaloc_l2_svox_night_test", "data/svox/images/test/gallery", "data/svox/images/test/queries_night", True),
        ("megaloc_dot_product_sf_xs_test", "data/sf_xs/test/database", "data/sf_xs/test/queries", False),
        ("megaloc_dot_product_tokyo_xs_test", "data/tokyo_xs/test/database", "data/tokyo_xs/test/queries", False),
        ("megaloc_dot_product_svox_sun_test", "data/svox/images/test/gallery", "data/svox/images/test/queries", True),
        ("megaloc_dot_product_svox_night_test", "data/svox/images/test/gallery", "data/svox/images/test/queries_night", True),
    ]
    
    results = []
    for exp_name, db_folder, q_folder, recursive in experiments:
        is_valid = verify_experiment(exp_name, db_folder, q_folder, recursive)
        results.append((exp_name, is_valid))
    
    # 总结
    print("\n" + "="*80)
    print("验证总结")
    print("="*80)
    
    valid_count = sum(1 for _, is_valid in results if is_valid)
    total_count = len(results)
    
    print(f"\n总计: {total_count} 个实验")
    print(f"通过: {valid_count} 个")
    print(f"失败: {total_count - valid_count} 个")
    
    if valid_count < total_count:
        print(f"\n有问题的实验:")
        for exp_name, is_valid in results:
            if not is_valid:
                print(f"  - {exp_name}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
