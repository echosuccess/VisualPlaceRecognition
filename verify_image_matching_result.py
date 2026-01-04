#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证Image Matching结果文件的完整性和正确性
"""

import json
import pickle
from pathlib import Path
import sys

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def verify_result_file(result_path):
    """验证单个结果文件"""
    result_path = Path(result_path)
    
    print(f"\n{'='*80}")
    print(f"验证文件: {result_path.name}")
    print(f"{'='*80}")
    
    # 检查文件名是否包含异常值
    if '_12_' in result_path.name:
        print(f"⚠️  警告: 文件名包含'12'，可能使用了错误的distance参数")
    
    # 1. 检查文件是否存在
    if not result_path.exists():
        print(f"❌ 文件不存在: {result_path}")
        return False
    
    # 2. 检查文件大小
    file_size = result_path.stat().st_size
    print(f"文件大小: {file_size / 1024**2:.2f} MB")
    
    if file_size < 1000:  # 小于1KB可能有问题
        print(f"⚠️  警告: 文件大小异常小，可能不完整")
    
    # 3. 尝试加载文件
    data = None
    file_format = None
    
    if result_path.suffix == '.json':
        file_format = 'JSON'
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ JSON文件可以正常加载")
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {e}")
            print(f"   尝试从.pkl文件恢复...")
            # 尝试从pkl恢复
            pkl_path = result_path.with_suffix('.pkl')
            if pkl_path.exists():
                try:
                    with open(pkl_path, 'rb') as f:
                        data = pickle.load(f)
                    print(f"✅ 从.pkl文件成功恢复数据")
                    file_format = 'PKL (recovered)'
                except Exception as e2:
                    print(f"❌ 无法从.pkl恢复: {e2}")
                    return False
            else:
                print(f"❌ 对应的.pkl文件不存在: {pkl_path}")
                return False
        except Exception as e:
            print(f"❌ 加载JSON文件时出错: {e}")
            return False
    
    elif result_path.suffix == '.pkl':
        file_format = 'PKL'
        try:
            with open(result_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✅ PKL文件可以正常加载")
        except Exception as e:
            print(f"❌ 加载PKL文件时出错: {e}")
            return False
    
    else:
        print(f"❌ 不支持的文件格式: {result_path.suffix}")
        return False
    
    # 4. 验证数据结构
    if data is None:
        print(f"❌ 数据为空")
        return False
    
    required_keys = ['results', 'config']
    missing_keys = [k for k in required_keys if k not in data]
    issues_found = []  # 用于收集所有问题
    
    if missing_keys:
        print(f"⚠️  缺少必需的键: {missing_keys}")
        if 'config' in missing_keys:
            issues_found.append("缺少'config'键，无法验证实验参数（可能是旧版本脚本生成的）")
    else:
        print(f"✅ 数据结构完整")
    
    # 5. 验证results内容
    if 'results' not in data:
        print(f"❌ 缺少'results'键")
        return False
    
    results = data['results']
    required_result_keys = ['query_ids', 'pred_ranks', 'is_correct', 'num_inliers']
    missing_result_keys = [k for k in required_result_keys if k not in results]
    
    if missing_result_keys:
        print(f"❌ results中缺少必需的键: {missing_result_keys}")
        return False
    
    # 6. 验证数据一致性
    query_ids = results.get('query_ids', [])
    pred_ranks = results.get('pred_ranks', [])
    is_correct = results.get('is_correct', [])
    num_inliers = results.get('num_inliers', [])
    
    num_entries = len(query_ids)
    print(f"\n数据统计:")
    print(f"  总条目数: {num_entries}")
    print(f"  query_ids长度: {len(query_ids)}")
    print(f"  pred_ranks长度: {len(pred_ranks)}")
    print(f"  is_correct长度: {len(is_correct)}")
    print(f"  num_inliers长度: {len(num_inliers)}")
    
    # 检查长度一致性
    lengths = [len(query_ids), len(pred_ranks), len(is_correct), len(num_inliers)]
    if len(set(lengths)) > 1:
        print(f"❌ 数据长度不一致!")
        return False
    else:
        print(f"✅ 所有数组长度一致")
    
    # 7. 验证数据范围
    if num_entries == 0:
        print(f"❌ 没有数据条目")
        return False
    
    # 检查inliers范围
    if num_inliers:
        min_inliers = min(num_inliers)
        max_inliers = max(num_inliers)
        mean_inliers = sum(num_inliers) / len(num_inliers)
        print(f"\nInliers统计:")
        print(f"  最小值: {min_inliers}")
        print(f"  最大值: {max_inliers}")
        print(f"  平均值: {mean_inliers:.2f}")
        
        if max_inliers < 0:
            print(f"⚠️  警告: 存在负数的inliers")
        if max_inliers > 10000:
            print(f"⚠️  警告: inliers值异常大")
    
    # 检查正确性分布
    issues_found = []
    if is_correct:
        num_correct = sum(1 for x in is_correct if x)
        num_incorrect = sum(1 for x in is_correct if not x)
        correct_ratio = num_correct / len(is_correct) if len(is_correct) > 0 else 0
        print(f"\n正确性分布:")
        print(f"  正确预测: {num_correct} ({correct_ratio*100:.2f}%)")
        print(f"  错误预测: {num_incorrect} ({(1-correct_ratio)*100:.2f}%)")
        
        # 检查正确率是否异常低
        if correct_ratio < 0.05:  # 低于5%
            issues_found.append(f"正确率异常低 ({correct_ratio*100:.2f}%)，可能使用了错误的参数或数据有问题")
        elif correct_ratio < 0.10:  # 低于10%
            issues_found.append(f"正确率较低 ({correct_ratio*100:.2f}%)，建议检查")
    
    # 8. 验证config
    if 'config' in data:
        config = data['config']
        print(f"\n配置信息:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # 检查distance参数
        if 'distance' in config:
            distance = config['distance']
            if distance not in ['l2', 'dot_product']:
                print(f"\n⚠️  警告: distance参数异常: '{distance}'")
                print(f"   应该是 'l2' 或 'dot_product'")
                if distance == '12':
                    issues_found.append(f"distance参数是'12'（错误），应该是'l2'或'dot_product'")
                    print(f"   ❌ 检测到 '12'，这是错误的参数值！")
                    print(f"   建议删除此文件并重新运行，使用正确的参数")
    
    # 9. 验证与VPR log的一致性
    if TORCH_AVAILABLE and 'config' in data and 'vpr_log_dir' in data['config']:
        vpr_log_dir = Path(data['config']['vpr_log_dir'])
        if vpr_log_dir.exists():
            z_data_path = vpr_log_dir / "z_data.torch"
            if z_data_path.exists():
                try:
                    z_data = torch.load(z_data_path, map_location='cpu', weights_only=False)
                    predictions = z_data.get('predictions', None)
                    if predictions is not None:
                        if isinstance(predictions, torch.Tensor):
                            vpr_num_queries = predictions.shape[0]
                        else:
                            vpr_num_queries = len(predictions)
                    else:
                        vpr_num_queries = 0
                    
                    # 统计唯一的query数量
                    unique_queries = len(set(query_ids)) if query_ids else 0
                    
                    print(f"\n与VPR log对比:")
                    print(f"  VPR queries数量: {vpr_num_queries}")
                    print(f"  结果中唯一queries: {unique_queries}")
                    
                    if unique_queries != vpr_num_queries:
                        print(f"⚠️  警告: query数量不匹配!")
                        print(f"   可能原因:")
                        print(f"     1. 部分queries被跳过")
                        print(f"     2. 数据不完整")
                    else:
                        print(f"✅ query数量匹配")
                except Exception as e:
                    print(f"⚠️  无法加载VPR log进行对比: {e}")
    elif not TORCH_AVAILABLE:
        print(f"\n⚠️  PyTorch不可用，跳过VPR log对比")
    
    # 检查文件名中的异常
    if '_12_' in result_path.name:
        issues_found.append("文件名包含'12'，说明使用了错误的distance参数（应该是'l2'或'dot_product'）")
    
    # 总结验证结果
    print(f"\n{'='*80}")
    print("验证总结")
    print(f"{'='*80}")
    
    if issues_found:
        print(f"❌ 发现 {len(issues_found)} 个严重问题:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
        print(f"\n⚠️  建议: 这个文件无效，需要删除并重新运行实验")
        print(f"   删除命令: rm {result_path}")
        print(f"   重新运行: 使用正确的参数 --distance l2 或 --distance dot_product")
        return False
    elif missing_keys:
        print(f"⚠️  警告: 缺少 {len(missing_keys)} 个键，但数据基本完整")
        print(f"✅ 文件结构基本正常")
        return True
    else:
        print(f"✅ 验证完成: 文件看起来正常")
        return True

def verify_experiment(matcher, vpr_method, dataset, distance):
    """验证特定实验的结果"""
    result_file = f"results/image_matching/{matcher}_{vpr_method}_{distance}_{dataset}.json"
    return verify_result_file(result_file)

def list_all_result_files():
    """列出所有结果文件"""
    results_dir = Path("results/image_matching")
    
    if not results_dir.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return []
    
    json_files = sorted(results_dir.glob("*.json"))
    return json_files

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    
    import argparse
    parser = argparse.ArgumentParser(description='验证Image Matching结果文件')
    parser.add_argument('--file', type=str, help='结果文件路径（JSON或PKL）')
    parser.add_argument('--matcher', type=str, help='Matcher名称')
    parser.add_argument('--vpr_method', type=str, help='VPR方法')
    parser.add_argument('--dataset', type=str, help='数据集名称')
    parser.add_argument('--distance', type=str, help='距离度量')
    parser.add_argument('--check_all', action='store_true', help='检查所有结果文件')
    parser.add_argument('--list', action='store_true', help='列出所有结果文件')
    
    args = parser.parse_args()
    
    if args.list:
        # 列出所有文件
        print("="*80)
        print("所有结果文件列表")
        print("="*80)
        files = list_all_result_files()
        if files:
            print(f"\n找到 {len(files)} 个JSON文件:\n")
            for f in files:
                size_mb = f.stat().st_size / (1024**2)
                print(f"  {f.name} ({size_mb:.2f} MB)")
            
            # 检查异常文件
            abnormal = [f for f in files if '_12_' in f.name]
            if abnormal:
                print(f"\n⚠️  发现 {len(abnormal)} 个可能异常的文件（包含'12'）:")
                for f in abnormal:
                    print(f"  {f.name}")
        else:
            print("没有找到结果文件")
        return
    
    if args.check_all:
        # 检查所有结果文件
        results_dir = Path("results/image_matching")
        if not results_dir.exists():
            print(f"❌ 结果目录不存在: {results_dir}")
            return
        
        json_files = list(results_dir.glob("*.json"))
        print(f"找到 {len(json_files)} 个JSON结果文件")
        
        success_count = 0
        fail_count = 0
        
        for json_file in sorted(json_files):
            if verify_result_file(json_file):
                success_count += 1
            else:
                fail_count += 1
        
        print(f"\n{'='*80}")
        print(f"总结:")
        print(f"  成功: {success_count}")
        print(f"  失败: {fail_count}")
        print(f"  总计: {len(json_files)}")
        print(f"{'='*80}")
    
    elif args.file:
        # 验证指定文件
        verify_result_file(args.file)
    
    elif args.matcher and args.vpr_method and args.dataset and args.distance:
        # 验证指定实验
        verify_experiment(args.matcher, args.vpr_method, args.dataset, args.distance)
    
    else:
        print("请指定要验证的文件或实验参数")
        print("\n示例:")
        print("  # 列出所有文件")
        print("  python verify_image_matching_result.py --list")
        print("\n  # 验证单个文件")
        print("  python verify_image_matching_result.py --file results/image_matching/loftr_cosplace_l2_sfxs_val.json")
        print("\n  # 验证特定实验")
        print("  python verify_image_matching_result.py --matcher loftr --vpr_method cosplace --dataset sfxs_val --distance l2")
        print("\n  # 验证所有文件")
        print("  python verify_image_matching_result.py --check_all")
        print("\n提示: 如果文件不存在，先运行 --list 查看实际存在的文件")

if __name__ == "__main__":
    main()
