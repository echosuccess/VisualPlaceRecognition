#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查GPU性能和资源使用情况
"""

import torch
import subprocess
import sys

def check_gpu_info():
    """检查GPU信息"""
    print("="*80)
    print("GPU 信息检查")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用！")
        return
    
    print(f"✅ CUDA 可用")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"PyTorch 版本: {torch.__version__}")
    
    num_gpus = torch.cuda.device_count()
    print(f"\nGPU 数量: {num_gpus}")
    
    for i in range(num_gpus):
        print(f"\n--- GPU {i} ---")
        props = torch.cuda.get_device_properties(i)
        print(f"名称: {props.name}")
        print(f"总内存: {props.total_memory / 1024**3:.2f} GB")
        print(f"计算能力: {props.major}.{props.minor}")
        
        # 检查当前内存使用
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"已分配内存: {allocated:.2f} GB")
        print(f"已保留内存: {reserved:.2f} GB")
        print(f"可用内存: {(props.total_memory / 1024**3) - reserved:.2f} GB")

def check_nvidia_smi():
    """使用nvidia-smi检查GPU状态"""
    print("\n" + "="*80)
    print("nvidia-smi 输出")
    print("="*80)
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("⚠️  nvidia-smi 不可用")
    except FileNotFoundError:
        print("⚠️  nvidia-smi 命令未找到（可能不在Linux环境）")
    except subprocess.TimeoutExpired:
        print("⚠️  nvidia-smi 超时")
    except Exception as e:
        print(f"⚠️  运行 nvidia-smi 时出错: {e}")

def test_loftr_speed():
    """测试LoFTR速度"""
    print("\n" + "="*80)
    print("LoFTR 速度测试")
    print("="*80)
    
    try:
        from matching import get_matcher
        import time
        from PIL import Image
        import numpy as np
        import torchvision.transforms as transforms
        
        print("加载 LoFTR matcher...")
        matcher = get_matcher("loftr", device="cuda")
        print("✅ LoFTR 加载完成")
        
        # 创建测试图像（512x512）
        print("\n创建测试图像 (512x512)...")
        test_img = Image.new('RGB', (512, 512), color='white')
        transform = transforms.ToTensor()
        img_tensor = transform(test_img)
        
        print("运行 5 次匹配测试...")
        times = []
        for i in range(5):
            torch.cuda.synchronize()
            start = time.time()
            
            # 运行匹配
            mkpts0, mkpts1, _, _, _, _ = matcher._forward(img_tensor, img_tensor)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  测试 {i+1}: {elapsed:.3f} 秒")
        
        avg_time = sum(times) / len(times)
        print(f"\n平均时间: {avg_time:.3f} 秒/匹配")
        print(f"如果每个query需要匹配20个候选，预计: {avg_time * 20:.2f} 秒/query")
        
        if avg_time > 0.5:
            print("\n⚠️  警告: LoFTR速度较慢，可能原因:")
            print("  1. GPU性能不足（Colab免费版T4）")
            print("  2. 图像尺寸较大")
            print("  3. 内存不足")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def check_colab_runtime():
    """检查Colab运行时类型"""
    print("\n" + "="*80)
    print("Colab 运行时检查")
    print("="*80)
    
    try:
        import os
        runtime_type = os.environ.get('COLAB_GPU', 'unknown')
        print(f"Colab GPU 环境: {runtime_type}")
        
        # 检查是否是免费版
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name.lower()
            
            if 't4' in gpu_name:
                print("⚠️  检测到 T4 GPU（Colab免费版）")
                print("   T4性能有限，LoFTR会较慢")
            elif 'v100' in gpu_name or 'a100' in gpu_name:
                print("✅ 检测到高性能GPU（Colab Pro）")
            else:
                print(f"GPU类型: {props.name}")
    except Exception as e:
        print(f"⚠️  无法检查Colab运行时: {e}")

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("="*80)
    print("GPU 性能诊断工具")
    print("="*80)
    
    check_gpu_info()
    check_nvidia_smi()
    check_colab_runtime()
    test_loftr_speed()
    
    print("\n" + "="*80)
    print("建议")
    print("="*80)
    print("""
如果LoFTR速度很慢（>5秒/query），建议：

1. **使用Colab Pro**：
   - 免费版T4 GPU性能有限
   - Pro版有更好的GPU（V100/A100）

2. **让LoFTR继续运行**：
   - 已经有checkpoint机制，可以安全中断
   - 预计总时间：25-30小时（14278 queries × 7s/query）

3. **并行运行其他实验**：
   - 在另一个Colab实例运行SuperGlue
   - SuperGlue通常比LoFTR快3-5倍

4. **考虑使用EfficientLoFTR**：
   - 如果支持，可以尝试eloftr（更快但精度略低）

5. **检查是否有其他进程占用GPU**：
   - 运行 nvidia-smi 查看GPU使用率
   - 确保没有其他Python进程占用GPU
    """)

if __name__ == "__main__":
    main()
