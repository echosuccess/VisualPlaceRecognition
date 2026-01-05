#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run VPR experiments for SF-XS val (Extension 6.1 validation set)
"""

import subprocess
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime

def run_vpr_experiment(method, backbone, dim, image_size, log_name):
    """Run a single VPR experiment"""
    
    cmd = [
        "python", "VPR-methods-evaluation/main.py",
        "--method", method,
        "--backbone", backbone,
        "--descriptors_dimension", str(dim),
        "--image_size", *image_size.split(),
        "--database_folder", "data/sf_xs/val/database",
        "--queries_folder", "data/sf_xs/val/queries",
        "--distance_metric", "l2",
        "--log_dir", f"baseline/{log_name}",
        "--num_preds_to_save", "20",
        "--max_queries_to_save", "3",
        "--recall_values", "1", "5", "10", "20",
        "--save_for_uncertainty",
        "--num_workers", "8",
        "--batch_size", "32"
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {log_name}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n[OK] {log_name} completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {log_name} failed: {e}")
        return False

def save_checkpoint(completed_experiments, checkpoint_path):
    """Save checkpoint"""
    checkpoint_data = {
        'completed': completed_experiments,
        'timestamp': datetime.now().isoformat()
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    print(f"[CHECKPOINT] Saved: {checkpoint_path}")

def load_checkpoint(checkpoint_path):
    """Load checkpoint"""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            return checkpoint_data.get('completed', [])
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")
    return []

def main():
    """Main function"""
    
    print("="*80)
    print("SF-XS val VPR Experiments (Extension 6.1 validation set)")
    print("="*80)
    
    checkpoint_path = Path("checkpoints/sfxs_val_vpr.json")
    
    val_data_path = Path("data/sf_xs/val")
    if not val_data_path.exists():
        print(f"\n[ERROR] {val_data_path} does not exist!")
        print("Please prepare SF-XS val dataset first")
        return
    experiments = [
        {
            "method": "cosplace",
            "backbone": "ResNet18",
            "dim": "512",
            "image_size": "512 512",
            "log_name": "cosplace_l2_sfxs_val"
        },
        {
            "method": "netvlad",
            "backbone": "VGG16",
            "dim": "4096",
            "image_size": "512 512",
            "log_name": "netvlad_l2_sfxs_val"
        },
        {
            "method": "mixvpr",
            "backbone": "ResNet50",
            "dim": "4096",
            "image_size": "320 320",
            "log_name": "mixvpr_l2_sfxs_val"
        },
        {
            "method": "megaloc",
            "backbone": "Dinov2",
            "dim": "8448",
            "image_size": "224 224",
            "log_name": "megaloc_l2_sfxs_val"
        }
    ]
    
    print(f"\nWill run {len(experiments)} VPR experiments")
    print("\nExperiment list:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['log_name']}")
    
    checkpoint_completed = load_checkpoint(checkpoint_path)
    if checkpoint_completed:
        print(f"\n[CHECKPOINT] Resumed from checkpoint: {len(checkpoint_completed)} completed experiments")
    
    print("\nChecking completed experiments:")
    completed = []
    remaining = []
    
    for exp in experiments:
        log_dir = Path(f"logs/baseline/{exp['log_name']}")
        z_data_files = list(log_dir.glob("**/z_data.torch"))
        
        if z_data_files or exp['log_name'] in checkpoint_completed:
            completed.append(exp['log_name'])
            if z_data_files:
                print(f"  [OK] {exp['log_name']} (completed)")
            else:
                print(f"  [WARN] {exp['log_name']} (marked as completed in checkpoint, but file not found, will re-run)")
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
    
    for i, exp in enumerate(remaining, 1):
        print(f"\nProgress: {i}/{len(remaining)}")
        success = run_vpr_experiment(
            exp["method"],
            exp["backbone"],
            exp["dim"],
            exp["image_size"],
            exp["log_name"]
        )
        
        if success:
            log_dir = Path(f"logs/baseline/{exp['log_name']}")
            z_data_files = list(log_dir.glob("**/z_data.torch"))
            if z_data_files:
                completed.append(exp['log_name'])
                save_checkpoint(completed, checkpoint_path)
                print(f"[CHECKPOINT] Experiment {exp['log_name']} completed, checkpoint saved")
        else:
            print(f"\n[WARN] Experiment failed, but continuing with next...")
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[INFO] Checkpoint file cleaned up (all experiments completed)")
    
    print("\nNext step: Run SF-XS val Image Matching experiments")
    print("Command: python run_sfxs_val_image_matching.py")

if __name__ == "__main__":
    main()
