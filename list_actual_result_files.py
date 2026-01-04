#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
列出实际存在的结果文件
"""

from pathlib import Path
import sys

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    
    results_dir = Path("results/image_matching")
    
    if not results_dir.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return
    
    print("="*80)
    print("实际存在的结果文件")
    print("="*80)
    
    # 列出所有JSON文件
    json_files = sorted(results_dir.glob("*.json"))
    pkl_files = sorted(results_dir.glob("*.pkl"))
    
    print(f"\n找到 {len(json_files)} 个JSON文件:")
    for f in json_files:
        size_mb = f.stat().st_size / (1024**2)
        print(f"  {f.name} ({size_mb:.2f} MB)")
    
    print(f"\n找到 {len(pkl_files)} 个PKL文件:")
    for f in pkl_files:
        size_mb = f.stat().st_size / (1024**2)
        print(f"  {f.name} ({size_mb:.2f} MB)")
    
    # 检查是否有distance=12的文件
    print("\n" + "="*80)
    print("检查异常文件名（distance=12）")
    print("="*80)
    
    abnormal_files = [f for f in json_files if '_12_' in f.name]
    if abnormal_files:
        print(f"\n⚠️  找到 {len(abnormal_files)} 个可能异常的文件（包含'12'）:")
        for f in abnormal_files:
            print(f"  {f.name}")
    else:
        print("\n✅ 没有发现包含'12'的文件名")
    
    # 检查sfxs_val相关的文件
    print("\n" + "="*80)
    print("sfxs_val 相关文件")
    print("="*80)
    
    sfxs_val_files = [f for f in json_files if 'sfxs_val' in f.name]
    if sfxs_val_files:
        print(f"\n找到 {len(sfxs_val_files)} 个sfxs_val文件:")
        for f in sfxs_val_files:
            size_mb = f.stat().st_size / (1024**2)
            print(f"  {f.name} ({size_mb:.2f} MB)")
    else:
        print("\n⚠️  没有找到sfxs_val相关的文件")
    
    # 按matcher分组
    print("\n" + "="*80)
    print("按Matcher分组")
    print("="*80)
    
    matchers = {}
    for f in json_files:
        parts = f.stem.split('_')
        if len(parts) >= 1:
            matcher = parts[0]
            if matcher not in matchers:
                matchers[matcher] = []
            matchers[matcher].append(f.name)
    
    for matcher, files in sorted(matchers.items()):
        print(f"\n{matcher}: {len(files)} 个文件")
        for fname in sorted(files)[:5]:  # 只显示前5个
            print(f"  {fname}")
        if len(files) > 5:
            print(f"  ... 还有 {len(files) - 5} 个文件")

if __name__ == "__main__":
    main()
