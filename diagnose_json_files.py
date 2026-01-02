#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断JSON文件问题
检查JSON文件是否完整、有效
"""

import json
import pickle
from pathlib import Path
import os


def check_json_file(json_file):
    """检查JSON文件是否有效"""
    json_file = Path(json_file)
    
    if not json_file.exists():
        print(f"❌ File does not exist: {json_file}")
        return False
    
    # 检查文件大小
    file_size = json_file.stat().st_size
    size_mb = file_size / (1024 * 1024)
    
    print(f"\n📄 File: {json_file.name}")
    print(f"   Size: {size_mb:.2f} MB ({file_size:,} bytes)")
    
    # 检查文件是否为空
    if file_size == 0:
        print(f"   ❌ File is empty!")
        return False
    
    # 尝试读取文件的前几行和后几行
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"   Total lines: {len(lines):,}")
        
        # 检查最后一行是否完整（应该以}或]结尾）
        if lines:
            last_line = lines[-1].strip()
            if not (last_line.endswith('}') or last_line.endswith(']') or last_line.endswith('}')):
                print(f"   ⚠️  Last line may be incomplete: ...{last_line[-50:]}")
        
        # 显示前3行和后3行
        print(f"\n   First 3 lines:")
        for i, line in enumerate(lines[:3], 1):
            print(f"     {i}: {line[:80]}...")
        
        print(f"\n   Last 3 lines:")
        for i, line in enumerate(lines[-3:], 1):
            print(f"     {len(lines)-3+i}: {line[:80]}...")
            
    except Exception as e:
        print(f"   ❌ Failed to read file: {e}")
        return False
    
    # 尝试解析JSON
    print(f"\n   Attempting to parse JSON...")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查数据结构
        if 'results' not in data:
            print(f"   ⚠️  Missing 'results' key")
        else:
            results_count = len(data['results'].get('query_ids', []))
            print(f"   ✅ Found 'results' with {results_count:,} entries")
        
        if 'analysis' not in data:
            print(f"   ⚠️  Missing 'analysis' key")
        else:
            print(f"   ✅ Found 'analysis'")
        
        print(f"   ✅ JSON is valid and complete!")
        return True
        
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON decode error!")
        print(f"      Error: {e}")
        print(f"      Line: {e.lineno if hasattr(e, 'lineno') else 'unknown'}")
        print(f"      Column: {e.colno if hasattr(e, 'colno') else 'unknown'}")
        print(f"      Position: {e.pos if hasattr(e, 'pos') else 'unknown'}")
        
        # 显示错误位置附近的内容
        if hasattr(e, 'lineno') and e.lineno:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    all_lines = f.readlines()
                    error_line_idx = e.lineno - 1
                    if 0 <= error_line_idx < len(all_lines):
                        error_line = all_lines[error_line_idx]
                        print(f"\n      Error line ({e.lineno}):")
                        print(f"      {error_line[:200]}")
                        if len(error_line) > 200:
                            print(f"      ... (truncated)")
            except:
                pass
        
        return False
        
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False


def check_pickle_file(pkl_file):
    """检查pickle文件"""
    pkl_file = Path(pkl_file)
    
    if not pkl_file.exists():
        return None
    
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        info = {
            'exists': True,
            'size_mb': pkl_file.stat().st_size / (1024 * 1024),
            'has_results': 'results' in data,
            'has_analysis': 'analysis' in data,
            'results_count': len(data.get('results', {}).get('query_ids', [])) if 'results' in data else 0
        }
        return info
    except Exception as e:
        return {'exists': True, 'error': str(e)}


def diagnose_all_json_files(results_dir="results/image_matching"):
    """诊断所有JSON文件"""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"❌ Directory not found: {results_dir}")
        return
    
    json_files = list(results_dir.glob("*.json"))
    json_files = [f for f in json_files if 'summary' not in f.name and not f.name.endswith('.bak')]
    
    print(f"{'='*80}")
    print(f"JSON File Diagnosis")
    print(f"{'='*80}")
    print(f"Found {len(json_files)} JSON files to check\n")
    
    valid_count = 0
    invalid_count = 0
    
    for json_file in sorted(json_files):
        is_valid = check_json_file(json_file)
        
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            
            # 检查是否有对应的pickle文件
            pkl_file = json_file.with_suffix('.pkl')
            pkl_info = check_pickle_file(pkl_file)
            
            if pkl_info:
                print(f"\n   📦 Pickle file found:")
                print(f"      Size: {pkl_info['size_mb']:.2f} MB")
                if 'error' in pkl_info:
                    print(f"      ⚠️  Error loading: {pkl_info['error']}")
                else:
                    print(f"      Has results: {pkl_info['has_results']}")
                    print(f"      Has analysis: {pkl_info['has_analysis']}")
                    if pkl_info['has_results']:
                        print(f"      Results count: {pkl_info['results_count']:,}")
                    print(f"      ✅ Can recover from pickle!")
            else:
                print(f"\n   ❌ No pickle file found - cannot recover")
        
        print()
    
    print(f"{'='*80}")
    print(f"Summary:")
    print(f"  Valid files: {valid_count}")
    print(f"  Invalid files: {invalid_count}")
    print(f"{'='*80}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose JSON files')
    parser.add_argument('--file', type=str, default=None, help='Specific JSON file to check')
    parser.add_argument('--dir', type=str, default='results/image_matching', 
                       help='Directory containing JSON files')
    
    args = parser.parse_args()
    
    if args.file:
        check_json_file(args.file)
    else:
        diagnose_all_json_files(args.dir)


if __name__ == "__main__":
    main()
