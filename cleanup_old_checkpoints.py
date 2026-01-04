#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理旧格式的checkpoint文件（旧命名格式：matcher_date_time_checkpoint.pkl）
新格式：matcher_vpr_log_dir_checkpoint.pkl
"""

from pathlib import Path
import re

def is_old_format_checkpoint(filename):
    """检查是否是旧格式的checkpoint"""
    # 旧格式：matcher_YYYY-MM-DD_HH-MM-SS_checkpoint.pkl
    # 例如：superpoint-lg_2025-12-31_20-20-48_checkpoint.pkl
    pattern = r'^(superglue|loftr|superpoint-lg)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_checkpoint\.pkl$'
    return bool(re.match(pattern, filename))

def is_new_format_checkpoint(filename):
    """检查是否是新格式的checkpoint"""
    # 新格式：matcher_vpr_log_dir_checkpoint.pkl
    # 例如：superpoint-lg_mixvpr_dot_product_tokyo_xs_test_2025-12-31_20-20-48_checkpoint.pkl
    # 新格式包含VPR方法和数据集信息，不会只有日期时间
    if '_checkpoint.pkl' not in filename:
        return False
    
    # 如果只匹配日期时间模式，就是旧格式
    if re.match(r'^(superglue|loftr|superpoint-lg)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_checkpoint\.pkl$', filename):
        return False
    
    # 新格式应该包含VPR方法或数据集名称
    vpr_methods = ['cosplace', 'netvlad', 'mixvpr', 'megaloc']
    datasets = ['sf_xs', 'tokyo_xs', 'svox', 'sfxs_val']
    
    has_vpr = any(method in filename for method in vpr_methods)
    has_dataset = any(ds in filename for ds in datasets)
    
    return has_vpr or has_dataset

def main():
    checkpoint_dir = Path("checkpoints/image_matching")
    
    if not checkpoint_dir.exists():
        print(f"[INFO] Checkpoint目录不存在: {checkpoint_dir}")
        return
    
    all_checkpoints = list(checkpoint_dir.glob("*.pkl"))
    
    if not all_checkpoints:
        print("[INFO] 没有找到checkpoint文件")
        return
    
    old_format = []
    new_format = []
    unknown = []
    
    for ckpt in all_checkpoints:
        filename = ckpt.name
        if is_old_format_checkpoint(filename):
            old_format.append(ckpt)
        elif is_new_format_checkpoint(filename):
            new_format.append(ckpt)
        else:
            unknown.append(ckpt)
    
    print("="*80)
    print("Checkpoint文件分析")
    print("="*80)
    print(f"\n总计: {len(all_checkpoints)} 个checkpoint文件")
    print(f"  旧格式（需要清理）: {len(old_format)}")
    print(f"  新格式（保留）: {len(new_format)}")
    print(f"  未知格式: {len(unknown)}")
    
    if old_format:
        print("\n" + "="*80)
        print("旧格式checkpoint文件（建议删除）")
        print("="*80)
        for ckpt in sorted(old_format):
            size_mb = ckpt.stat().st_size / (1024*1024)
            print(f"  {ckpt.name} ({size_mb:.2f}MB)")
        
        print("\n" + "="*80)
        print("是否删除旧格式checkpoint？")
        print("="*80)
        print("\n这些是旧格式的checkpoint，命名容易混淆，建议删除。")
        print("新代码已经使用新格式，不会产生冲突。")
        print("\n删除命令:")
        print("```python")
        for ckpt in old_format:
            print(f"Path('{ckpt}').unlink()")
        print("```")
        
        print("\n或者批量删除:")
        print("```python")
        print("from pathlib import Path")
        print("import re")
        print()
        print("checkpoint_dir = Path('checkpoints/image_matching')")
        print("pattern = r'^(superglue|loftr|superpoint-lg)_\\d{4}-\\d{2}-\\d{2}_\\d{2}-\\d{2}-\\d{2}_checkpoint\\.pkl$'")
        print()
        print("for ckpt in checkpoint_dir.glob('*.pkl'):")
        print("    if re.match(pattern, ckpt.name):")
        print("        size_mb = ckpt.stat().st_size / (1024*1024)")
        print("        ckpt.unlink()")
        print("        print(f'✅ 已删除: {ckpt.name} ({size_mb:.2f}MB)')")
        print("```")
    
    if new_format:
        print("\n" + "="*80)
        print("新格式checkpoint文件（保留）")
        print("="*80)
        for ckpt in sorted(new_format):
            size_mb = ckpt.stat().st_size / (1024*1024)
            print(f"  {ckpt.name} ({size_mb:.2f}MB)")
    
    if unknown:
        print("\n" + "="*80)
        print("未知格式checkpoint文件（请检查）")
        print("="*80)
        for ckpt in sorted(unknown):
            size_mb = ckpt.stat().st_size / (1024*1024)
            print(f"  {ckpt.name} ({size_mb:.2f}MB)")

if __name__ == "__main__":
    main()
