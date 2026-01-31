"""
Monitor VAE training progress and provide recommendations
Run this periodically while training to check status
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def monitor_training_progress(checkpoint_dir):
    """Monitor training progress from checkpoint directory"""
    
    checkpoint_dir = Path(checkpoint_dir)
    
    # Look for latest checkpoint
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
    if not checkpoints:
        print("No checkpoints found yet. Training may have just started.")
        return
    
    # Get latest epoch number
    latest_epoch = max([int(str(cp.stem).split('_')[-1]) for cp in checkpoints])
    
    print("="*70)
    print(f"TRAINING PROGRESS MONITOR - Latest Epoch: {latest_epoch}")
    print("="*70)
    
    # Check if there's a training log or metrics file
    # For now, provide guidance based on what we know
    
    print(f"\n📊 Current Status:")
    print(f"  Latest checkpoint: epoch_{latest_epoch}")
    print(f"  Training started from: epoch 54 (best model)")
    print(f"  Target: epoch 160")
    print(f"  Progress: {(latest_epoch/160)*100:.1f}% complete")
    print(f"  Remaining: {160 - latest_epoch} epochs")
    
    # Estimate time
    avg_time_per_epoch = 15  # minutes (average between 13-18)
    remaining_time = (160 - latest_epoch) * avg_time_per_epoch
    hours = remaining_time // 60
    minutes = remaining_time % 60
    
    print(f"\n⏱️  Estimated Time:")
    print(f"  Avg time/epoch: ~15 minutes")
    print(f"  Remaining time: ~{hours}h {minutes}m")
    
    # Current cycle info
    cycle_num = (latest_epoch // 40) + 1
    epoch_in_cycle = latest_epoch % 40
    beta_value = min(4.0, (epoch_in_cycle / 40) * 4.0)
    
    print(f"\n🔄 β-Annealing Status:")
    print(f"  Cycle: {cycle_num}/4")
    print(f"  Epoch in cycle: {epoch_in_cycle}/40")
    print(f"  Estimated β: {beta_value:.2f}")
    
    # Recommendations based on epoch
    print(f"\n💡 Recommendations:")
    
    if latest_epoch < 80:
        print("  ⏳ Continue training - Still in cycle 2")
        print("  📌 Check again at epoch 80 (end of cycle 2)")
        print("  🎯 Model may improve when β resets to 0 at epoch 80")
        
    elif latest_epoch < 120:
        print("  ⏳ Continue training - In cycle 3")
        print("  📌 Check again at epoch 120 (end of cycle 3)")
        print("  ⚠️  If no improvement over epoch 54, consider stopping")
        
    elif latest_epoch < 160:
        print("  ⏳ In final cycle (4)")
        print("  📌 Close to completion")
        print("  ✅ Let it finish unless validation loss is increasing")
        
    else:
        print("  ✅ Training complete!")
        print("  📊 Run final evaluation and compare with epoch 54")
    
    # Check if best_model.pt exists and compare
    best_model_path = checkpoint_dir / 'best_model.pt'
    if best_model_path.exists():
        print(f"\n✅ Best model checkpoint exists")
        print(f"  Location: {best_model_path}")
        print(f"  Note: This gets updated whenever validation improves")
    
    return latest_epoch


def main():
    checkpoint_dir = 'models/checkpoints/vae_zdim64_beta4.0_20251102_073456'
    
    try:
        latest_epoch = monitor_training_progress(checkpoint_dir)
        
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("="*70)
        print("\n1. Let training continue in the background")
        print("2. Check progress periodically with:")
        print("   python scripts/monitor_training.py")
        print("\n3. When training completes or you want to check quality:")
        print("   python scripts/assess_model_quality.py")
        print("\n4. To stop training: Press Ctrl+C in training terminal")
        
    except Exception as e:
        print(f"Error monitoring training: {e}")
        print("\nMake sure training has started and checkpoints are being saved.")


if __name__ == '__main__':
    main()
