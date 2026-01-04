#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整检查已完成实验的checkpoint冲突风险
"""

from pathlib import Path
import json
from collections import defaultdict

results_dir = Path("results/image_matching")
if not results_dir.exists():
    print("results/image_matching 目录不存在")
else:
    # 1. 解析所有已完成的结果文件
    completed_experiments = {}
    for result_file in results_dir.glob("*.json"):
        if result_file.name == 'batch_summary.json':
            continue
        
        parts = result_file.stem.split('_')
        if len(parts) >= 4:
            matcher = parts[0]
            vpr_method = parts[1]
            distance = parts[2]
            dataset = '_'.join(parts[3:])
            
            vpr_exp_key = f"{vpr_method}_{distance}_{dataset}"
            vpr_log_base = Path(f"logs/baseline/{vpr_exp_key}")
            
            timestamp = None
            if vpr_log_base.exists():
                timestamp_dirs = sorted([d for d in vpr_log_base.iterdir() if d.is_dir()])
                if timestamp_dirs:
                    timestamp = timestamp_dirs[-1].name
            
            completed_experiments[result_file.stem] = {
                'matcher': matcher,
                'vpr_method': vpr_method,
                'dataset': dataset,
                'vpr_exp_key': vpr_exp_key,
                'timestamp': timestamp,
                'result_file': result_file
            }
    
    print(f"找到 {len(completed_experiments)} 个已完成实验\n")
    
    # 2. 按VPR实验分组
    vpr_groups = defaultdict(list)
    for exp_name, exp_info in completed_experiments.items():
        if exp_info['timestamp']:
            key = f"{exp_info['vpr_exp_key']}_{exp_info['timestamp']}"
            vpr_groups[key].append(exp_info)
    
    # 3. 检查冲突
    conflicts = []
    safe_groups = []
    
    checkpoint_dir = Path("checkpoints/image_matching")
    
    for vpr_key, exps in vpr_groups.items():
        if len(exps) > 1:
            # 多个matcher使用了同一个VPR实验
            timestamp = exps[0]['timestamp']
            
            # 关键：检查是否有旧格式checkpoint
            old_checkpoint = checkpoint_dir / f"{timestamp}_checkpoint.pkl"
            has_old_checkpoint = old_checkpoint.exists()
            
            # 检查是否有新格式checkpoint
            new_checkpoints = {}
            for exp in exps:
                new_ckpt = checkpoint_dir / f"{exp['matcher']}_{timestamp}_checkpoint.pkl"
                if new_ckpt.exists():
                    new_checkpoints[exp['matcher']] = new_ckpt
            
            # 验证每个结果文件
            all_valid = True
            exp_details = []
            for exp in exps:
                result_file = exp['result_file']
                try:
                    with open(str(result_file), 'r') as f:
                        data = json.load(f)
                    num_queries = len(data.get('results', {}).get('query_ids', []))
                    num_matches = len(data.get('results', {}).get('num_inliers', []))
                    
                    if num_queries > 0 and num_matches > 0:
                        exp_details.append({
                            'matcher': exp['matcher'],
                            'num_queries': num_queries,
                            'num_matches': num_matches,
                            'valid': True
                        })
                    else:
                        exp_details.append({
                            'matcher': exp['matcher'],
                            'num_queries': num_queries,
                            'num_matches': num_matches,
                            'valid': False
                        })
                        all_valid = False
                except Exception as e:
                    exp_details.append({
                        'matcher': exp['matcher'],
                        'error': str(e),
                        'valid': False
                    })
                    all_valid = False
            
            # 判断冲突风险
            if has_old_checkpoint:
                # 有旧格式checkpoint = 高风险
                print(f"⚠️  [HIGH RISK] {vpr_key}")
                print(f"   发现旧格式checkpoint: {old_checkpoint.name}")
                print(f"   有 {len(exps)} 个matcher:")
                for detail in exp_details:
                    if detail['valid']:
                        print(f"      ✅ {detail['matcher']}: {detail['num_queries']} queries, {detail['num_matches']} matches")
                    else:
                        print(f"      ❌ {detail['matcher']}: {detail.get('error', '数据为空')}")
                
                if all_valid:
                    print(f"   ⚠️  所有结果文件完整，但存在checkpoint冲突风险！")
                    print(f"   ⚠️  建议：验证数据一致性，或重新运行以确保准确性")
                else:
                    print(f"   ❌ 部分结果文件有问题，数据可能混乱！")
                    print(f"   ❌ 建议：重新运行这些实验")
                
                conflicts.append({
                    'vpr_key': vpr_key,
                    'risk': 'HIGH',
                    'old_checkpoint': old_checkpoint,
                    'experiments': exps,
                    'all_valid': all_valid
                })
            elif len(new_checkpoints) > 0:
                # 有新格式checkpoint，但可能不完整
                if len(new_checkpoints) < len(exps):
                    print(f"⚠️  [MEDIUM RISK] {vpr_key}")
                    print(f"   部分matcher有新格式checkpoint:")
                    for matcher, ckpt in new_checkpoints.items():
                        print(f"      - {matcher}: {ckpt.name}")
                    print(f"   但缺少: {[e['matcher'] for e in exps if e['matcher'] not in new_checkpoints]}")
                    conflicts.append({
                        'vpr_key': vpr_key,
                        'risk': 'MEDIUM',
                        'new_checkpoints': new_checkpoints,
                        'experiments': exps,
                        'all_valid': all_valid
                    })
                else:
                    # 所有matcher都有新格式checkpoint，应该是安全的
                    safe_groups.append(vpr_key)
            else:
                # 没有checkpoint（可能已完成并删除了），检查结果文件
                if all_valid:
                    # 所有结果文件完整，应该是安全的
                    safe_groups.append(vpr_key)
                else:
                    print(f"⚠️  [LOW RISK] {vpr_key}")
                    print(f"   没有checkpoint，但部分结果文件有问题")
                    conflicts.append({
                        'vpr_key': vpr_key,
                        'risk': 'LOW',
                        'experiments': exps,
                        'all_valid': False
                    })
        else:
            # 只有一个matcher，应该是安全的
            safe_groups.append(vpr_key)
    
    # 4. 总结
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    
    print(f"\n总计VPR实验组: {len(vpr_groups)}")
    print(f"  安全（无冲突风险）: {len(safe_groups)}")
    print(f"  有冲突风险: {len(conflicts)}")
    
    if conflicts:
        high_risk = [c for c in conflicts if c['risk'] == 'HIGH']
        medium_risk = [c for c in conflicts if c['risk'] == 'MEDIUM']
        low_risk = [c for c in conflicts if c['risk'] == 'LOW']
        
        if high_risk:
            print(f"\n  [HIGH RISK] 高风险（存在旧格式checkpoint冲突）: {len(high_risk)}")
            for c in high_risk:
                print(f"    - {c['vpr_key']}")
            print(f"   ⚠️  建议：验证数据一致性，或重新运行以确保准确性")
        
        if medium_risk:
            print(f"\n  [MEDIUM RISK] 中等风险: {len(medium_risk)}")
            for c in medium_risk:
                print(f"    - {c['vpr_key']}")
        
        if low_risk:
            print(f"\n  [LOW RISK] 低风险: {len(low_risk)}")
            for c in low_risk:
                print(f"    - {c['vpr_key']}")
    else:
        print("\n✅ 未发现checkpoint冲突风险")
        print("   所有实验都使用新格式checkpoint或没有checkpoint冲突")
