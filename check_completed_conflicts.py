#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查已完成实验是否存在checkpoint冲突导致的数据混乱
"""

from pathlib import Path
import json
import pickle
from collections import defaultdict

def analyze_completed_experiments(results_dir_path="results/image_matching"):
    """分析已完成实验的checkpoint冲突情况"""
    print("="*80)
    print("检查已完成实验的Checkpoint冲突风险")
    print("="*80)
    
    results_dir = Path(results_dir_path)
    if not results_dir.exists():
        print(f"\n[ERROR] {results_dir_path} 目录不存在")
        print(f"如果结果文件在其他位置，请修改脚本中的 results_dir_path 参数")
        return
    
    # 1. 解析所有已完成的结果文件
    completed_experiments = {}
    for result_file in results_dir.glob("*.json"):
        # 排除非实验文件
        if result_file.name in ['batch_summary.json']:
            continue
            
        # 格式: {matcher}_{vpr_method}_{distance}_{dataset}.json
        parts = result_file.stem.split('_')
        if len(parts) >= 4:
            matcher = parts[0]
            vpr_method = parts[1]
            distance = parts[2]
            dataset = '_'.join(parts[3:])
            
            # 查找对应的VPR日志时间戳
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
    
    print(f"\n[1] 找到 {len(completed_experiments)} 个已完成实验")
    
    # 2. 按VPR实验分组，找出有多个matcher的实验
    vpr_groups = defaultdict(list)
    for exp_name, exp_info in completed_experiments.items():
        key = f"{exp_info['vpr_exp_key']}_{exp_info['timestamp']}"
        vpr_groups[key].append(exp_info)
    
    # 3. 检查每个VPR实验组
    print(f"\n[2] 分析VPR实验组（检查是否有多个matcher共享checkpoint）")
    
    conflicts = []
    safe_experiments = []
    
    for vpr_key, exps in vpr_groups.items():
        if len(exps) > 1:
            # 多个matcher使用了同一个VPR实验
            print(f"\n  VPR实验: {vpr_key}")
            print(f"    有 {len(exps)} 个matcher:")
            
            # 检查是否有旧格式checkpoint
            checkpoint_dir = Path("checkpoints/image_matching")
            timestamp = exps[0]['timestamp']
            old_checkpoint = checkpoint_dir / f"{timestamp}_checkpoint.pkl" if timestamp else None
            
            has_old_checkpoint = old_checkpoint and old_checkpoint.exists()
            
            # 验证每个结果文件
            all_valid = True
            for exp in exps:
                result_file = exp['result_file']
                size_mb = result_file.stat().st_size / (1024*1024)
                
                # 验证文件内容
                try:
                    with open(str(result_file), 'r') as f:
                        data = json.load(f)
                    num_queries = len(data.get('results', {}).get('query_ids', []))
                    num_matches = len(data.get('results', {}).get('num_inliers', []))
                    
                    if num_queries > 0 and num_matches > 0:
                        print(f"      ✅ {exp['matcher']}: {result_file.name}")
                        print(f"         大小: {size_mb:.1f}MB, 查询: {num_queries}, 匹配: {num_matches}")
                    else:
                        print(f"      ❌ {exp['matcher']}: 数据为空")
                        all_valid = False
                except Exception as e:
                    print(f"      ❌ {exp['matcher']}: 无法读取 - {e}")
                    all_valid = False
            
            # 判断冲突风险
            if has_old_checkpoint:
                print(f"     ⚠️  发现旧格式checkpoint: {old_checkpoint.name}")
                print(f"     ⚠️  这些matcher可能共享了checkpoint，存在冲突风险！")
                
                if all_valid:
                    print(f"     ✅ 但所有结果文件看起来完整，数据可能是正确的")
                    print(f"     ⚠️  建议：验证数据一致性（检查inliers分布是否合理）")
                    conflicts.append({
                        'vpr_key': vpr_key,
                        'experiments': exps,
                        'old_checkpoint': old_checkpoint,
                        'all_valid': True,
                        'risk': 'MEDIUM'  # 有冲突风险，但结果文件完整
                    })
                else:
                    print(f"     ❌ 部分结果文件有问题，数据可能混乱！")
                    conflicts.append({
                        'vpr_key': vpr_key,
                        'experiments': exps,
                        'old_checkpoint': old_checkpoint,
                        'all_valid': False,
                        'risk': 'HIGH'  # 高风险，数据可能混乱
                    })
            else:
                # 没有旧checkpoint，或者使用新格式
                if all_valid:
                    print(f"     ✅ 所有结果文件完整，且没有旧checkpoint冲突")
                    safe_experiments.append(vpr_key)
                else:
                    print(f"     ⚠️  部分结果文件有问题，但可能不是checkpoint冲突导致的")
        else:
            # 只有一个matcher，应该是安全的
            safe_experiments.append(vpr_key)
    
    # 4. 总结
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    
    print(f"\n总计VPR实验组: {len(vpr_groups)}")
    print(f"  安全（无冲突风险）: {len(safe_experiments)}")
    print(f"  有冲突风险: {len(conflicts)}")
    
    if conflicts:
        print(f"\n[WARN] 发现 {len(conflicts)} 个潜在的checkpoint冲突:")
        
        high_risk = [c for c in conflicts if c['risk'] == 'HIGH']
        medium_risk = [c for c in conflicts if c['risk'] == 'MEDIUM']
        
        if high_risk:
            print(f"\n  [HIGH RISK] 高风险（数据可能混乱）: {len(high_risk)}")
            for c in high_risk:
                print(f"    - {c['vpr_key']}")
                print(f"      建议：重新运行这些实验以确保数据准确性")
        
        if medium_risk:
            print(f"\n  [MEDIUM RISK] 中等风险（结果文件完整但可能有冲突）: {len(medium_risk)}")
            for c in medium_risk:
                print(f"    - {c['vpr_key']}")
                print(f"      建议：验证数据一致性，如果合理可以保留")
    else:
        print("\n[OK] 未发现明显的checkpoint冲突风险")
    
    # 5. 提供验证建议
    print("\n" + "="*80)
    print("验证建议")
    print("="*80)
    
    if conflicts:
        print("\n对于有冲突风险的实验，建议：")
        print("1. 检查结果文件的数据分布是否合理")
        print("2. 比较不同matcher的结果，看是否符合预期")
        print("3. 如果数据看起来异常，建议重新运行")
        print("\n验证脚本示例：")
        print("""
# 检查特定实验的数据分布
import json
from pathlib import Path

result_file = Path("results/image_matching/xxx.json")
with open(str(result_file), 'r') as f:
    data = json.load(f)

results = data['results']
inliers = results['num_inliers']
is_correct = results['is_correct']

# 检查正确和错误预测的inliers分布
correct_inliers = [inliers[i] for i in range(len(inliers)) if is_correct[i]]
incorrect_inliers = [inliers[i] for i in range(len(inliers)) if not is_correct[i]]

print(f"正确预测平均inliers: {sum(correct_inliers)/len(correct_inliers):.1f}")
print(f"错误预测平均inliers: {sum(incorrect_inliers)/len(incorrect_inliers):.1f}")

# 如果正确预测的inliers明显高于错误预测，说明数据可能是合理的
        """)
    else:
        print("\n所有已完成实验看起来是安全的，不需要额外验证。")

if __name__ == "__main__":
    analyze_completed_experiments()
