#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复损坏的JSON文件
从pickle文件恢复并重新保存为JSON
"""

import json
import pickle
from pathlib import Path
import sys


def fix_json_from_pickle(json_file):
    """从pickle文件修复JSON文件"""
    json_file = Path(json_file)
    
    if not json_file.exists():
        print(f"[ERROR] JSON file not found: {json_file}")
        return False
    
    pkl_file = json_file.with_suffix('.pkl')
    
    if not pkl_file.exists():
        print(f"[ERROR] Pickle file not found: {pkl_file}")
        print(f"[INFO] Cannot recover without pickle file")
        return False
    
    print(f"[INFO] Found pickle file: {pkl_file}")
    print(f"[INFO] Attempting to recover JSON from pickle...")
    
    try:
        # 加载pickle文件
        with open(pkl_file, 'rb') as f:
            pkl_data = pickle.load(f)
        
        # 转换为JSON格式
        json_data = {
            'results': pkl_data['results'],
            'analysis': pkl_data['analysis']
        }
        
        # 备份原JSON文件
        backup_file = json_file.with_suffix('.json.bak')
        if json_file.exists():
            import shutil
            shutil.copy2(json_file, backup_file)
            print(f"[INFO] Backed up original JSON to: {backup_file}")
        
        # 保存为JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # 验证新JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        if 'results' in loaded and 'analysis' in loaded:
            print(f"[OK] Successfully recovered: {json_file.name}")
            print(f"[INFO] File size: {json_file.stat().st_size / (1024*1024):.2f} MB")
            return True
        else:
            print(f"[ERROR] Recovered file is missing required keys")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to recover: {e}")
        return False


def fix_all_corrupted_json(results_dir="results/image_matching"):
    """修复所有损坏的JSON文件"""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        return
    
    json_files = list(results_dir.glob("*.json"))
    json_files = [f for f in json_files if 'summary' not in f.name and not f.name.endswith('.bak')]
    
    print(f"[INFO] Found {len(json_files)} JSON files to check")
    print(f"{'='*80}\n")
    
    fixed_count = 0
    skipped_count = 0
    error_count = 0
    
    for json_file in json_files:
        print(f"Checking: {json_file.name}")
        
        # 尝试加载JSON验证是否损坏
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证数据结构
            if 'results' not in data or 'analysis' not in data:
                print(f"  [WARN] Missing required keys, attempting to fix...")
                if fix_json_from_pickle(json_file):
                    fixed_count += 1
                else:
                    error_count += 1
            else:
                print(f"  [OK] File is valid")
                skipped_count += 1
                
        except json.JSONDecodeError as e:
            print(f"  [ERROR] JSON decode error: {e}")
            print(f"  [INFO] Attempting to fix from pickle...")
            if fix_json_from_pickle(json_file):
                fixed_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            print(f"  [ERROR] Unexpected error: {e}")
            error_count += 1
        
        print()
    
    print(f"{'='*80}")
    print(f"Summary:")
    print(f"  Valid files: {skipped_count}")
    print(f"  Fixed files: {fixed_count}")
    print(f"  Failed files: {error_count}")
    print(f"{'='*80}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fix corrupted JSON files from pickle backups')
    parser.add_argument('--file', type=str, default=None, help='Specific JSON file to fix')
    parser.add_argument('--dir', type=str, default='results/image_matching', 
                       help='Directory containing JSON files')
    
    args = parser.parse_args()
    
    if args.file:
        # 修复单个文件
        fix_json_from_pickle(args.file)
    else:
        # 修复所有文件
        fix_all_corrupted_json(args.dir)


if __name__ == "__main__":
    main()
