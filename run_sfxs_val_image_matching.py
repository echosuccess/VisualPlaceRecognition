#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Image Matching experiments for SF-XS val (Extension 6.1 validation set)
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

def run_im_experiment(matcher, vpr_method, dataset, distance, device='cuda'):
    """Run a single Image Matching experiment"""
    
    cmd = [
        "python", "run_image_matching_baseline.py",
        "--matcher", matcher,
        "--vpr_method", vpr_method,
        "--dataset", dataset,
        "--distance", distance,
        "--device", device
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {matcher} + {vpr_method} + {dataset}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n[OK] {matcher} + {vpr_method} + {dataset} completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {matcher} + {vpr_method} + {dataset} failed: {e}")
        return False

def main():
    """Main function"""
    
    print("="*80)
    print("SF-XS val Image Matching Experiments (Extension 6.1 validation set)")
    print("="*80)
    
    vpr_methods = ['cosplace', 'netvlad', 'mixvpr', 'megaloc']
    missing_vpr = []
    
    for method in vpr_methods:
        vpr_dir = Path(f"logs/baseline/{method}_l2_sfxs_val")
        if not vpr_dir.exists():
            missing_vpr.append(method)
    
    if missing_vpr:
        print(f"\n[ERROR] The following VPR experiments are not completed:")
        for m in missing_vpr:
            print(f"  - {m}_l2_sfxs_val")
        print("\nPlease run first: python run_sfxs_val_vpr.py")
        return
    
    experiments = []
    matchers = ['superpoint-lg', 'loftr', 'superglue']
    
    for vpr_method in vpr_methods:
        for matcher in matchers:
            experiments.append({
                'matcher': matcher,
                'vpr_method': vpr_method,
                'dataset': 'sfxs_val',
                'distance': 'l2'
            })
    
    print(f"\nWill run {len(experiments)} Image Matching experiments")
    
    checkpoint_path = Path("checkpoints/sfxs_val_image_matching.json")
    
    checkpoint_completed = []
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            checkpoint_completed = checkpoint_data.get('completed', [])
            if checkpoint_completed:
                print(f"\n[CHECKPOINT] Resumed from checkpoint: {len(checkpoint_completed)} completed experiments")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")
    
    print("\nChecking completed experiments:")
    completed = []
    remaining = []
    
    for exp in experiments:
        exp_key = f"{exp['matcher']}_{exp['vpr_method']}"
        result_file = Path(f"results/image_matching/{exp['matcher']}_{exp['vpr_method']}_l2_{exp['dataset']}.json")
        result_file_pkl = Path(f"results/image_matching/{exp['matcher']}_{exp['vpr_method']}_l2_{exp['dataset']}.pkl")
        
        if result_file.exists() or result_file_pkl.exists() or exp_key in checkpoint_completed:
            completed.append(exp_key)
            if result_file.exists() or result_file_pkl.exists():
                print(f"  [OK] {exp['matcher']} + {exp['vpr_method']} (completed)")
            else:
                print(f"  [WARN] {exp['matcher']} + {exp['vpr_method']} (marked as completed in checkpoint, but file not found, will re-run)")
                remaining.append(exp)
        else:
            remaining.append(exp)
    
    if completed:
        print(f"\nCompleted: {len(completed)}/{len(experiments)}")
    
    if not remaining:
        print("\n[OK] All experiments completed!")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"[INFO] Checkpoint file cleaned up")
        return
    
    print(f"\nRemaining: {len(remaining)}/{len(experiments)}")
    print("\nStarting remaining experiments...")
    
    def save_checkpoint(completed_list, checkpoint_path):
        checkpoint_data = {
            'completed': completed_list,
            'timestamp': datetime.now().isoformat()
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        print(f"[CHECKPOINT] Saved: {checkpoint_path}")
    
    for i, exp in enumerate(remaining, 1):
        print(f"\nProgress: {i}/{len(remaining)}")
        success = run_im_experiment(
            exp['matcher'],
            exp['vpr_method'],
            exp['dataset'],
            exp['distance']
        )
        
        if success:
            result_file = Path(f"results/image_matching/{exp['matcher']}_{exp['vpr_method']}_l2_{exp['dataset']}.json")
            result_file_pkl = Path(f"results/image_matching/{exp['matcher']}_{exp['vpr_method']}_l2_{exp['dataset']}.pkl")
            if result_file.exists() or result_file_pkl.exists():
                exp_key = f"{exp['matcher']}_{exp['vpr_method']}"
                completed.append(exp_key)
                save_checkpoint(completed, checkpoint_path)
                print(f"[CHECKPOINT] Experiment {exp_key} completed, checkpoint saved")
        else:
            print(f"\n[WARN] Experiment failed, but continuing with next...")
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[INFO] Checkpoint file cleaned up (all experiments completed)")
    
    print("\nNext step: Run Extension 6.1")
    print("Command: python run_extension_6_1.py")

if __name__ == "__main__":
    main()
