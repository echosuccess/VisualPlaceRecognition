"""
快速统计数据集图像数量（Colab专用）
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

def count_images(directory):
    """统计目录中的图像文件"""
    if not Path(directory).exists():
        return 0, []
    
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    all_files = []
    
    for ext in extensions:
        all_files.extend(list(Path(directory).rglob(f'*{ext}')))
    
    return len(all_files), sorted([str(f.relative_to(Path(directory))) for f in all_files[:5]])

def main():
    datasets = {
        'tokyo_xs_test': {
            'database': 'data/tokyo_xs/test/database',
            'queries': 'data/tokyo_xs/test/queries'
        },
        'sf_xs_test': {
            'database': 'data/sf_xs/test/database',
            'queries': 'data/sf_xs/test/queries'
        },
        'svox_night_test': {
            'database': 'data/svox/images/test/gallery',
            'queries': 'data/svox/images/test/queries_night'
        },
        'svox_sun_test': {
            'database': 'data/svox/images/test/gallery',
            'queries': 'data/svox/images/test/queries'
        }
    }
    
    print("=" * 80)
    print("数据集图像数量统计")
    print("=" * 80)
    
    for dataset_name, paths in datasets.items():
        print(f"\n📁 {dataset_name}")
        print("-" * 80)
        
        # 数据库
        db_path = paths['database']
        db_count, db_samples = count_images(db_path)
        print(f"  数据库: {db_path}")
        if Path(db_path).exists():
            print(f"    ✅ 存在 - {db_count} 张图像")
            if db_samples:
                print(f"    示例: {db_samples[0]}")
        else:
            print(f"    ❌ 不存在")
        
        # 查询
        q_path = paths['queries']
        q_count, q_samples = count_images(q_path)
        print(f"  查询: {q_path}")
        if Path(q_path).exists():
            print(f"    ✅ 存在 - {q_count} 张图像")
            if q_samples:
                print(f"    示例: {q_samples[0]}")
        else:
            print(f"    ❌ 不存在")
        
        # 比例
        if db_count > 0 and q_count > 0:
            ratio = db_count / q_count
            print(f"  数据库/查询比例: {ratio:.2f}x")
    
    # 特别关注Tokyo-XS
    print("\n" + "=" * 80)
    print("Tokyo-XS 详细分析")
    print("=" * 80)
    
    tokyo_db = 'data/tokyo_xs/test/database'
    tokyo_q = 'data/tokyo_xs/test/queries'
    
    tokyo_db_count, _ = count_images(tokyo_db)
    tokyo_q_count, _ = count_images(tokyo_q)
    
    print(f"\nTokyo-XS:")
    print(f"  数据库图像: {tokyo_db_count}")
    print(f"  查询图像: {tokyo_q_count}")
    
    # 对比其他数据集
    print(f"\n对比其他数据集:")
    for name, paths in datasets.items():
        if name != 'tokyo_xs_test':
            db_count, _ = count_images(paths['database'])
            q_count, _ = count_images(paths['queries'])
            print(f"  {name}: 数据库={db_count}, 查询={q_count}")

if __name__ == "__main__":
    main()
