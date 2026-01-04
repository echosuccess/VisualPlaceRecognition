#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证已完成的Image Matching实验结果
检查是否有checkpoint冲突导致的数据损坏
"""

from pathlib import Path
import json
import pickle
from collections import defaultdict

def check_result_file(result_path):
    """检查单个结果文件的完整性"""
    result_path = Path(result_path)
    
    if not result_path.exists():
        return {'status': 'MISSING', 'error': 'File not found'}
    
    # 检查文件大小
    file_size = result_path.stat().st_size
    if file_size == 0:
        return {'status': 'ERROR', 'error': 'File is empty', 'size': file_size}
    
    # 尝试加载
    try:
        if result_path.suffix == '.json':
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:  # .pkl
            with open(result_path, 'rb') as f:
                data = pickle.load(f)
        
        # 检查关键字段
        required_fields = ['results', 'metadata']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            return {
                'status': 'ERROR',
                'error': f'Missing required fields: {missing_fields}',
                'size': file_size
            }
        
        # 检查results中的关键字段
        results = data.get('results', {})
        required_result_fields = ['query_ids', 'pred_ranks', 'num_inliers', 'is_correct']
        missing_result_fields = [f for f in required_result_fields if f not in results]
        
        if missing_result_fields:
            return {
                'status': 'ERROR',
                'error': f'Missing result fields: {missing_result_fields}',
                'size': file_size
            }
        
        # 检查数据一致性
        query_ids = results.get('query_ids', [])
        num_inliers = results.get('num_inliers', [])
        is_correct = results.get('is_correct', [])
        
        if len(query_ids) != len(num_inliers) or len(query_ids) != len(is_correct):
            return {
                'status': 'ERROR',
                'error': f'Data length mismatch: queries={len(query_ids)}, inliers={len(num_inliers)}, correct={len(is_correct)}',
                'size': file_size
            }
        
        return {
            'status': 'OK',
            'size': file_size,
            'num_queries': len(query_ids),
            'num_matches': len(num_inliers)
        }
        
    except json.JSONDecodeError as e:
        return {'status': 'ERROR', 'error': f'JSON decode error: {e}', 'size': file_size}
    except Exception as e:
        return {'status': 'ERROR', 'error': f'Load error: {e}', 'size': file_size}

def find_all_image_matching_results():
    """查找所有Image Matching结果文件"""
    results_dir = Path("results/image_matching")
    if not results_dir.exists():
        return {}
    
    # 查找所有.json和.pkl文件
    json_files = list(results_dir.glob("*.json"))
    pkl_files = list(results_dir.glob("*.pkl"))
    
    # 合并并去重（优先使用.json）
    all_files = {}
    for pkl_file in pkl_files:
        base_name = pkl_file.stem
        all_files[base_name] = {'pkl': pkl_file, 'json': None}
    
    for json_file in json_files:
        base_name = json_file.stem
        if base_name not in all_files:
            all_files[base_name] = {'pkl': None, 'json': None}
        all_files[base_name]['json'] = json_file
    
    return all_files

def check_checkpoint_conflicts_for_completed():
    """检查已完成实验的checkpoint冲突情况"""
    results_dir = Path("results/image_matching")
    checkpoint_dir = Path("checkpoints/image_matching")
    
    if not results_dir.exists():
        return {}
    
    # 解析所有已完成的结果文件名
    completed_experiments = {}
    for result_file in results_dir.glob("*.json"):
        # 格式: {matcher}_{vpr_method}_{distance}_{dataset}.json
        parts = result_file.stem.split('_')
        if len(parts) >= 4:
            matcher = parts[0]
            vpr_method = parts[1]
            distance = parts[2]
            dataset = '_'.join(parts[3:])
            
            # 查找对应的VPR日志时间戳
            vpr_log_base = Path(f"logs/baseline/{vpr_method}_{distance}_{dataset}")
            if vpr_log_base.exists():
                timestamp_dirs = sorted([d for d in vpr_log_base.iterdir() if d.is_dir()])
                if timestamp_dirs:
                    timestamp = timestamp_dirs[-1].name
                    completed_experiments[result_file.stem] = {
                        'matcher': matcher,
                        'vpr_method': vpr_method,
                        'dataset': dataset,
                        'timestamp': timestamp,
                        'result_file': result_file
                    }
    
    # 检查是否有多个matcher使用同一个VPR实验（潜在冲突）
    vpr_timestamp_groups = defaultdict(list)
    for exp_name, exp_info in completed_experiments.items():
        key = f"{exp_info['vpr_method']}_{exp_info['dataset']}_{exp_info['timestamp']}"
        vpr_timestamp_groups[key].append(exp_info)
    
    conflicts = {}
    for key, exps in vpr_timestamp_groups.items():
        if len(exps) > 1:
            # 多个matcher使用了同一个VPR实验
            # 检查是否有旧格式checkpoint
            timestamp = exps[0]['timestamp']
            old_checkpoint = checkpoint_dir / f"{timestamp}_checkpoint.pkl"
            
            if old_checkpoint.exists():
                conflicts[key] = {
                    'experiments': exps,
                    'old_checkpoint': old_checkpoint,
                    'risk': 'HIGH'  # 使用旧格式checkpoint，有冲突风险
                }
            else:
                # 检查是否有新格式checkpoint
                new_checkpoints = {}
                for exp in exps:
                    new_ckpt = checkpoint_dir / f"{exp['matcher']}_{timestamp}_checkpoint.pkl"
                    if new_ckpt.exists():
                        new_checkpoints[exp['matcher']] = new_ckpt
                
                if len(new_checkpoints) < len(exps):
                    conflicts[key] = {
                        'experiments': exps,
                        'new_checkpoints': new_checkpoints,
                        'risk': 'MEDIUM'  # 部分使用新格式，可能有不一致
                    }
    
    return conflicts

def main():
    print("="*80)
    print("验证已完成的Image Matching实验结果")
    print("="*80)
    
    # 1. 查找所有结果文件
    print("\n[1] 查找所有结果文件...")
    all_results = find_all_image_matching_results()
    print(f"找到 {len(all_results)} 个实验结果")
    
    # 2. 检查每个结果文件的完整性
    print("\n[2] 检查结果文件完整性...")
    verification_results = {}
    for base_name, files in all_results.items():
        # 优先检查.json文件
        if files['json']:
            result = check_result_file(files['json'])
            verification_results[base_name] = {
                'file': files['json'],
                'backup': files['pkl'],
                **result
            }
        elif files['pkl']:
            result = check_result_file(files['pkl'])
            verification_results[base_name] = {
                'file': files['pkl'],
                'backup': None,
                **result
            }
    
    # 统计
    status_counts = defaultdict(int)
    for result in verification_results.values():
        status_counts[result['status']] += 1
    
    print(f"\n验证结果统计:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    # 显示有问题的文件
    error_files = {k: v for k, v in verification_results.items() if v['status'] != 'OK'}
    if error_files:
        print(f"\n[WARN] 发现 {len(error_files)} 个有问题的文件:")
        for name, result in error_files.items():
            print(f"  - {name}: {result.get('error', 'Unknown error')}")
    
    # 3. 检查checkpoint冲突
    print("\n[3] 检查checkpoint冲突...")
    conflicts = check_checkpoint_conflicts_for_completed()
    
    if conflicts:
        print(f"\n[WARN] 发现 {len(conflicts)} 个潜在的checkpoint冲突:")
        for key, conflict_info in conflicts.items():
            print(f"\n  冲突组: {key}")
            print(f"  风险级别: {conflict_info['risk']}")
            print(f"  涉及的实验:")
            for exp in conflict_info['experiments']:
                print(f"    - {exp['matcher']} + {exp['vpr_method']} + {exp['dataset']}")
            
            if conflict_info['risk'] == 'HIGH':
                print(f"  [CRITICAL] 使用旧格式checkpoint: {conflict_info['old_checkpoint'].name}")
                print(f"    这些实验可能互相覆盖了checkpoint，数据可能不准确！")
    else:
        print("\n[OK] 未发现checkpoint冲突")
    
    # 4. 总结
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    
    ok_count = status_counts.get('OK', 0)
    error_count = status_counts.get('ERROR', 0) + status_counts.get('MISSING', 0)
    total_count = len(verification_results)
    
    print(f"\n已完成实验: {total_count}")
    print(f"  正常: {ok_count}")
    print(f"  有问题: {error_count}")
    print(f"  潜在冲突: {len(conflicts)}")
    
    if error_count == 0 and len(conflicts) == 0:
        print("\n[OK] 所有已完成的实验结果看起来正常！")
    elif len(conflicts) > 0:
        print("\n[WARN] 发现潜在的checkpoint冲突，建议：")
        print("  1. 检查冲突的实验结果是否合理")
        print("  2. 如果怀疑数据有问题，重新运行这些实验")
    else:
        print("\n[OK] 所有结果文件完整，但建议检查checkpoint冲突情况")

if __name__ == "__main__":
    main()
