#!/usr/bin/env python3
"""
测试MixVPR修复是否成功
"""

import sys
import torch
from pathlib import Path

# 添加VPR-methods-evaluation到路径
sys.path.insert(0, str(Path(__file__).parent / "VPR-methods-evaluation"))

print("="*80)
print("测试MixVPR修复")
print("="*80)

try:
    # 测试1：导入模块
    print("\n[1/3] 测试导入模块...")
    from vpr_models import mixvpr
    print("[OK] Import successful")
    
    # 测试2：创建模型
    print("\n[2/3] 测试创建模型...")
    model = mixvpr.get_mixvpr(descriptors_dimension=4096)
    print(f"[OK] Model created successfully")
    print(f"   模型类型: {type(model)}")
    
    # 测试3：测试forward
    print("\n[3/3] 测试forward...")
    model = model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 320, 320)
        output = model(dummy_input)
        print(f"[OK] Forward pass successful")
        print(f"   输入形状: {dummy_input.shape}")
        print(f"   输出形状: {output.shape}")
        print(f"   输出维度: {output.shape[1]} (期望: 4096)")
    
    print("\n" + "="*80)
    print("[SUCCESS] All tests passed! MixVPR fix is working!")
    print("="*80)
    
except Exception as e:
    print(f"\n[FAILED] Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

