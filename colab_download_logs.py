#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab专用：打包并下载所有logs
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

def pack_logs():
    """打包logs目录"""
    
    print("="*70)
    print("📦 开始打包 logs 文件夹")
    print("="*70)
    
    # 检查logs目录是否存在
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("❌ logs 目录不存在！")
        return None
    
    # 统计logs信息
    log_files = list(logs_dir.rglob("*"))
    total_files = sum(1 for f in log_files if f.is_file())
    total_dirs = sum(1 for f in log_files if f.is_dir())
    
    print(f"\n📊 统计信息:")
    print(f"  - 文件夹: {total_dirs}")
    print(f"  - 文件: {total_files}")
    
    # 创建压缩文件名（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"vpr_logs_{timestamp}"
    
    print(f"\n🗜️ 正在压缩...")
    print(f"  目标文件: {zip_filename}.zip")
    
    try:
        # 使用shutil.make_archive创建zip文件（兼容Drive文件系统）
        archive_path = shutil.make_archive(
            base_name=zip_filename,
            format='zip',
            root_dir='.',
            base_dir='logs'
        )
        
        # 获取文件大小
        size_bytes = os.path.getsize(archive_path)
        size_mb = size_bytes / (1024 * 1024)
        
        print(f"\n✅ 打包完成！")
        print(f"  文件: {archive_path}")
        print(f"  大小: {size_mb:.2f} MB")
        
        return archive_path
        
    except Exception as e:
        print(f"\n❌ 打包失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def download_file_colab(file_path):
    """在Colab中下载文件到本地"""
    try:
        from google.colab import files
        print(f"\n⬇️ 开始下载到本地浏览器...")
        files.download(file_path)
        print(f"✅ 下载完成！文件已保存到浏览器的下载文件夹")
    except ImportError:
        print("\n💡 不在Colab环境中，跳过自动下载")
        print(f"   文件位置: {os.path.abspath(file_path)}")
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print(f"\n💡 替代方案：")
        print(f"   1. 在Colab左侧文件浏览器中找到: {file_path}")
        print(f"   2. 右键点击文件 → 下载")


def main():
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*20 + "VPR Logs 打包下载工具" + " "*25 + "║")
    print("╚" + "═"*68 + "╝")
    print("\n")
    
    # 1. 打包logs
    archive_path = pack_logs()
    
    if archive_path is None:
        return
    
    # 2. 下载文件
    print("\n" + "="*70)
    print("⬇️ 下载文件")
    print("="*70)
    download_file_colab(archive_path)
    
    print("\n" + "="*70)
    print("🎉 全部完成！")
    print("="*70)
    print("\n💡 提示:")
    print("  - 如果自动下载失败，请在左侧文件浏览器中手动下载")
    print("  - 下载后可以删除zip文件以节省Drive空间")
    print(f"  - 删除命令: !rm {archive_path}")
    print("\n")


if __name__ == "__main__":
    main()
