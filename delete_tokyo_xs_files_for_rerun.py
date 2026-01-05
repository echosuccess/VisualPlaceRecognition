"""
删除需要重新运行的Tokyo-XS Image Matching结果文件

根据完整Image_Matching统计_包含sfxs_val.md中的Cell 17-22
"""
import sys
from pathlib import Path

# 配置输出编码
sys.stdout.reconfigure(encoding='utf-8')

# 需要删除的文件列表（对应Cell 17-22）
files_to_delete = [
    # Cell 17: superglue + cosplace
    "superglue_cosplace_dot_product_tokyo_xs_test.json",
    "superglue_cosplace_dot_product_tokyo_xs_test.pkl",
    
    # Cell 18: superglue + mixvpr
    "superglue_mixvpr_dot_product_tokyo_xs_test.json",
    "superglue_mixvpr_dot_product_tokyo_xs_test.pkl",
    
    # Cell 19: superglue + megaloc
    "superglue_megaloc_dot_product_tokyo_xs_test.json",
    "superglue_megaloc_dot_product_tokyo_xs_test.pkl",
    
    # Cell 20: loftr + megaloc
    "loftr_megaloc_dot_product_tokyo_xs_test.json",
    "loftr_megaloc_dot_product_tokyo_xs_test.pkl",
    
    # Cell 21: superpoint-lg + cosplace
    "superpoint-lg_cosplace_dot_product_tokyo_xs_test.json",
    "superpoint-lg_cosplace_dot_product_tokyo_xs_test.pkl",
    
    # Cell 22: superpoint-lg + megaloc
    "superpoint-lg_megaloc_dot_product_tokyo_xs_test.json",
    "superpoint-lg_megaloc_dot_product_tokyo_xs_test.pkl",
]

# 结果目录（Colab和本地都支持）
results_dir = Path("results/image_matching")

print("=" * 80)
print("删除Tokyo-XS Image Matching结果文件（用于重新运行）")
print("=" * 80)
print(f"\n目标目录: {results_dir}")
print(f"需要删除的文件数: {len(files_to_delete)}")

if not results_dir.exists():
    print(f"\n⚠️  结果目录不存在: {results_dir}")
    print("   如果是在Colab上，请确保已挂载Drive或创建了results目录")
    sys.exit(1)

deleted_count = 0
not_found_count = 0

print("\n" + "=" * 80)
print("删除操作:")
print("=" * 80)

for filename in files_to_delete:
    file_path = results_dir / filename
    
    if file_path.exists():
        try:
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            file_path.unlink()
            print(f"✅ 已删除: {filename} ({file_size:.2f} MB)")
            deleted_count += 1
        except Exception as e:
            print(f"❌ 删除失败: {filename} - {e}")
    else:
        print(f"⚠️  文件不存在: {filename} (可能已经删除或未生成)")
        not_found_count += 1

print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print(f"✅ 成功删除: {deleted_count} 个文件")
print(f"⚠️  未找到: {not_found_count} 个文件")
print(f"📝 总计: {len(files_to_delete)} 个文件")

if deleted_count > 0:
    print("\n✅ 现在可以重新运行这些实验了！")
    print("\n在Colab中运行以下命令:")
    print("\n# Cell 17: superglue + cosplace")
    print("!python run_image_matching_baseline.py --matcher superglue --vpr_method cosplace --dataset tokyo_xs_test --distance dot_product --device cuda")
    print("\n# Cell 18: superglue + mixvpr")
    print("!python run_image_matching_baseline.py --matcher superglue --vpr_method mixvpr --dataset tokyo_xs_test --distance dot_product --device cuda")
    print("\n# Cell 19: superglue + megaloc")
    print("!python run_image_matching_baseline.py --matcher superglue --vpr_method megaloc --dataset tokyo_xs_test --distance dot_product --device cuda")
    print("\n# Cell 20: loftr + megaloc")
    print("!python run_image_matching_baseline.py --matcher loftr --vpr_method megaloc --dataset tokyo_xs_test --distance dot_product --device cuda")
    print("\n# Cell 21: superpoint-lg + cosplace")
    print("!python run_image_matching_baseline.py --matcher superpoint-lg --vpr_method cosplace --dataset tokyo_xs_test --distance dot_product --device cuda")
    print("\n# Cell 22: superpoint-lg + megaloc")
    print("!python run_image_matching_baseline.py --matcher superpoint-lg --vpr_method megaloc --dataset tokyo_xs_test --distance dot_product --device cuda")
else:
    print("\n⚠️  没有删除任何文件，请检查:")
    print("   1. 文件是否已经删除")
    print("   2. 结果目录路径是否正确")
    print("   3. 是否在正确的项目目录中运行")
