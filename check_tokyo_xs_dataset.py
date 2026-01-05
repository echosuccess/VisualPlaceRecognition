"""
检查Tokyo-XS数据集的基本信息和完整性
用于判断数据集本身是否有问题
"""
import sys
from pathlib import Path
import json

sys.stdout.reconfigure(encoding='utf-8')

def count_images_in_directory(directory):
    """统计目录中的图像文件数量"""
    if not Path(directory).exists():
        return 0, []
    
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    
    for ext in extensions:
        # 递归搜索
        image_files.extend(list(Path(directory).rglob(f'*{ext}')))
    
    return len(image_files), sorted([str(f) for f in image_files])

def get_directory_size(directory):
    """计算目录总大小（MB）"""
    if not Path(directory).exists():
        return 0
    
    total_size = 0
    for file_path in Path(directory).rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size / (1024 * 1024)  # MB

def check_vpr_logs(dataset_name):
    """检查VPR日志中的queries数量"""
    vpr_log_base = Path("logs/baseline")
    vpr_methods = ['cosplace', 'netvlad', 'mixvpr', 'megaloc']
    distances = ['dot_product', 'l2']
    
    vpr_info = {}
    
    for vpr_method in vpr_methods:
        for distance in distances:
            exp_name = f"{vpr_method}_{distance}_{dataset_name}"
            exp_dir = vpr_log_base / exp_name
            
            if exp_dir.exists():
                timestamp_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
                if timestamp_dirs:
                    latest_dir = timestamp_dirs[-1]
                    z_data_path = latest_dir / "z_data.torch"
                    
                    if z_data_path.exists():
                        try:
                            import torch
                            z_data = torch.load(z_data_path, map_location='cpu', weights_only=False)
                            predictions = z_data.get('predictions', None)
                            if predictions is not None:
                                if isinstance(predictions, torch.Tensor):
                                    num_queries = predictions.shape[0]
                                else:
                                    num_queries = len(predictions)
                                
                                vpr_info[f"{vpr_method}_{distance}"] = {
                                    'num_queries': num_queries,
                                    'log_dir': str(latest_dir)
                                }
                        except Exception as e:
                            print(f"  ⚠️  无法加载 {exp_name}: {e}")
    
    return vpr_info

def check_image_matching_results(dataset_name):
    """检查Image Matching结果文件"""
    results_dir = Path("results/image_matching")
    if not results_dir.exists():
        return {}
    
    matchers = ['superpoint-lg', 'loftr', 'superglue']
    vpr_methods = ['cosplace', 'netvlad', 'mixvpr', 'megaloc']
    distances = ['dot_product', 'l2']
    
    result_info = {}
    
    for matcher in matchers:
        for vpr_method in vpr_methods:
            for distance in distances:
                file_name = f"{matcher}_{vpr_method}_{distance}_{dataset_name}.json"
                file_path = results_dir / file_name
                
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        results = data.get('results', {})
                        num_inliers = results.get('num_inliers', [])
                        is_correct = results.get('is_correct', [])
                        
                        if num_inliers and is_correct:
                            # 计算inlier比率
                            correct_inliers = [num_inliers[i] for i in range(len(num_inliers)) if is_correct[i]]
                            incorrect_inliers = [num_inliers[i] for i in range(len(num_inliers)) if not is_correct[i]]
                            
                            mean_correct = sum(correct_inliers) / len(correct_inliers) if correct_inliers else 0
                            mean_incorrect = sum(incorrect_inliers) / len(incorrect_inliers) if incorrect_inliers else 0
                            
                            inlier_ratio = mean_correct / max(mean_incorrect, 1)
                            
                            result_info[f"{matcher}_{vpr_method}_{distance}"] = {
                                'num_matches': len(num_inliers),
                                'inlier_ratio': inlier_ratio,
                                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                                'status': 'normal' if inlier_ratio >= 1.5 else 'abnormal'
                            }
                    except Exception as e:
                        print(f"  ⚠️  无法读取 {file_name}: {e}")
    
    return result_info

def compare_with_other_datasets():
    """对比Tokyo-XS与其他数据集的结果"""
    datasets = ['tokyo_xs_test', 'sf_xs_test', 'svox_night_test', 'svox_sun_test']
    results_dir = Path("results/image_matching")
    
    if not results_dir.exists():
        print("⚠️  结果目录不存在")
        return
    
    print("\n" + "=" * 80)
    print("数据集对比分析")
    print("=" * 80)
    
    # 选择一个代表性的实验组合
    matcher = 'superpoint-lg'
    vpr_method = 'cosplace'
    distance = 'dot_product'
    
    comparison = {}
    
    for dataset in datasets:
        file_name = f"{matcher}_{vpr_method}_{distance}_{dataset}.json"
        file_path = results_dir / file_name
        
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                results = data.get('results', {})
                num_inliers = results.get('num_inliers', [])
                is_correct = results.get('is_correct', [])
                
                if num_inliers and is_correct:
                    correct_inliers = [num_inliers[i] for i in range(len(num_inliers)) if is_correct[i]]
                    incorrect_inliers = [num_inliers[i] for i in range(len(num_inliers)) if not is_correct[i]]
                    
                    mean_correct = sum(correct_inliers) / len(correct_inliers) if correct_inliers else 0
                    mean_incorrect = sum(incorrect_inliers) / len(incorrect_inliers) if incorrect_inliers else 0
                    inlier_ratio = mean_correct / max(mean_incorrect, 1)
                    
                    num_correct = sum(1 for x in is_correct if x)
                    total = len(is_correct)
                    correct_ratio = num_correct / total if total > 0 else 0
                    
                    comparison[dataset] = {
                        'inlier_ratio': inlier_ratio,
                        'correct_ratio': correct_ratio,
                        'num_matches': total
                    }
            except Exception as e:
                print(f"  ⚠️  无法读取 {file_name}: {e}")
    
    if comparison:
        print(f"\n实验组合: {matcher} + {vpr_method} + {distance}")
        print(f"\n{'数据集':<20} {'Inlier比率':<15} {'Top-20正确率':<15} {'总匹配数':<15}")
        print("-" * 80)
        
        for dataset, stats in comparison.items():
            status = "✅" if stats['inlier_ratio'] >= 1.5 else "⚠️"
            print(f"{status} {dataset:<18} {stats['inlier_ratio']:<15.2f} {stats['correct_ratio']*100:<14.2f}% {stats['num_matches']:<15}")
        
        # 分析Tokyo-XS是否异常
        if 'tokyo_xs_test' in comparison:
            tokyo_stats = comparison['tokyo_xs_test']
            other_stats = [stats for name, stats in comparison.items() if name != 'tokyo_xs_test']
            
            if other_stats:
                avg_other_ratio = sum(s['inlier_ratio'] for s in other_stats) / len(other_stats)
                
                print(f"\n分析:")
                print(f"  Tokyo-XS Inlier比率: {tokyo_stats['inlier_ratio']:.2f}x")
                print(f"  其他数据集平均Inlier比率: {avg_other_ratio:.2f}x")
                
                if tokyo_stats['inlier_ratio'] < avg_other_ratio * 0.7:
                    print(f"  ⚠️  Tokyo-XS的Inlier比率明显低于其他数据集，可能是数据集本身的问题")
                elif tokyo_stats['inlier_ratio'] < 1.5:
                    print(f"  ⚠️  Tokyo-XS的Inlier比率偏低，但可能仍在可接受范围内")
                else:
                    print(f"  ✅ Tokyo-XS的Inlier比率正常")

def main():
    print("=" * 80)
    print("Tokyo-XS 数据集完整性检查")
    print("=" * 80)
    
    dataset_name = 'tokyo_xs_test'
    
    # 1. 检查数据目录
    print("\n1. 数据目录检查")
    print("-" * 80)
    
    database_dir = f"data/tokyo_xs/test/database"
    queries_dir = f"data/tokyo_xs/test/queries"
    
    print(f"\n数据库目录: {database_dir}")
    db_exists = Path(database_dir).exists()
    print(f"  存在: {'✅' if db_exists else '❌'}")
    
    if db_exists:
        db_count, db_files = count_images_in_directory(database_dir)
        db_size = get_directory_size(database_dir)
        print(f"  图像数量: {db_count}")
        print(f"  目录大小: {db_size:.2f} MB")
        if db_files:
            print(f"  示例文件: {db_files[0]}")
    
    print(f"\n查询目录: {queries_dir}")
    q_exists = Path(queries_dir).exists()
    print(f"  存在: {'✅' if q_exists else '❌'}")
    
    if q_exists:
        q_count, q_files = count_images_in_directory(queries_dir)
        q_size = get_directory_size(queries_dir)
        print(f"  图像数量: {q_count}")
        print(f"  目录大小: {q_size:.2f} MB")
        if q_files:
            print(f"  示例文件: {q_files[0]}")
    
    # 2. 检查VPR日志
    print("\n2. VPR日志检查")
    print("-" * 80)
    
    try:
        vpr_info = check_vpr_logs(dataset_name)
        if vpr_info:
            print(f"\n找到 {len(vpr_info)} 个VPR实验:")
            for exp_name, info in vpr_info.items():
                print(f"  {exp_name}: {info['num_queries']} queries")
        else:
            print("  未找到VPR日志")
    except Exception as e:
        print(f"  ⚠️  检查VPR日志时出错: {e}")
    
    # 3. 检查Image Matching结果
    print("\n3. Image Matching结果检查")
    print("-" * 80)
    
    result_info = check_image_matching_results(dataset_name)
    if result_info:
        print(f"\n找到 {len(result_info)} 个结果文件:")
        
        normal_count = sum(1 for info in result_info.values() if info['status'] == 'normal')
        abnormal_count = sum(1 for info in result_info.values() if info['status'] == 'abnormal')
        
        print(f"  ✅ 正常: {normal_count}")
        print(f"  ⚠️  异常: {abnormal_count}")
        
        if abnormal_count > 0:
            print(f"\n异常结果:")
            for exp_name, info in result_info.items():
                if info['status'] == 'abnormal':
                    print(f"  - {exp_name}: Inlier比率 {info['inlier_ratio']:.2f}x (文件大小: {info['file_size_mb']:.2f} MB)")
    else:
        print("  未找到结果文件")
    
    # 4. 与其他数据集对比
    compare_with_other_datasets()
    
    # 5. 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    
    if db_exists and q_exists:
        print(f"✅ 数据目录存在")
        print(f"   数据库: {db_count} 张图像")
        print(f"   查询: {q_count} 张图像")
        
        if result_info:
            abnormal_count = sum(1 for info in result_info.values() if info['status'] == 'abnormal')
            if abnormal_count > 0:
                print(f"\n⚠️  发现 {abnormal_count} 个异常结果文件")
                print(f"   如果重新运行后仍然异常，可能是数据集本身的问题（Tokyo-XS数据集可能比其他数据集更具挑战性）")
            else:
                print(f"\n✅ 所有结果文件看起来正常")
    else:
        print(f"❌ 数据目录不存在，请检查数据集是否已下载")

if __name__ == "__main__":
    main()
