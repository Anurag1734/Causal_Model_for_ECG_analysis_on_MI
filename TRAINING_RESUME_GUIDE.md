# Resume Training Guide

## Overview
The VAE training script now supports resuming from checkpoints to handle interruptions.

## Available Checkpoints
Your current training session has these checkpoints:
- `best_model.pt` - Best model (epoch 13, Val Loss: 34,857)
- `checkpoint_epoch_10.pt` - Periodic checkpoint at epoch 10

Location: `models/checkpoints/vae_zdim64_beta4.0_20251102_073456/`

## How to Resume Training

### Command to Resume from Best Model
```powershell
cd C:\Users\Acer\Desktop\Capstone\Capstone
.\cuda_env\Scripts\activate
python src/models/train_vae.py --batch_size 256 --epochs 160 --lr 1e-4 --num_workers 0 --resume --checkpoint_path models/checkpoints/vae_zdim64_beta4.0_20251102_073456/best_model.pt
```

### Command to Resume from Epoch 10 Checkpoint
```powershell
cd C:\Users\Acer\Desktop\Capstone\Capstone
.\cuda_env\Scripts\activate
python src/models/train_vae.py --batch_size 256 --epochs 160 --lr 1e-4 --num_workers 0 --resume --checkpoint_path models/checkpoints/vae_zdim64_beta4.0_20251102_073456/checkpoint_epoch_10.pt
```

## What Gets Restored

When resuming, the following states are restored:
- **Model weights** - Exact network parameters
- **Optimizer state** - Adam momentum and learning rate state
- **Scheduler state** - ReduceLROnPlateau history
- **Training epoch** - Continues from the next epoch
- **Best validation loss** - Tracks the best model across sessions
- **Training history** - All metrics from previous epochs
- **Early stopping counter** - Patience counter for early stopping

## Resume Behavior

1. **Epoch Continuation**: If you resume from epoch 13, training will start at epoch 14
2. **Output Directory**: Uses the same checkpoint directory (no new timestamp)
3. **Configuration**: Skips config saving (uses original config.json)
4. **Metrics**: Appends to existing training history
5. **Checkpoints**: Continues saving to the same directory

## Important Notes

- **Always use `--resume` flag** when resuming from a checkpoint
- **Checkpoint path is required** when using `--resume`
- **Use the same hyperparameters** as the original training (batch_size, lr, etc.)
- **Training will continue** from the saved epoch to the specified `--epochs` value
- **New checkpoints overwrite** old periodic checkpoints (best_model.pt is always kept)

## Recommended: Resume from Best Model

Since your best model is at epoch 13, it's recommended to resume from `best_model.pt`:
```powershell
python src/models/train_vae.py --batch_size 256 --epochs 160 --lr 1e-4 --num_workers 0 --resume --checkpoint_path models/checkpoints/vae_zdim64_beta4.0_20251102_073456/best_model.pt
```

This will:
- Start from epoch 14
- Continue to epoch 160 (147 more epochs)
- Take approximately 35-40 hours
- Save checkpoints to the same directory

## Verification

When you run the resume command, you should see:
```
Loading checkpoint from models/checkpoints/vae_zdim64_beta4.0_20251102_073456/best_model.pt...
✓ Checkpoint loaded successfully!
  Resuming from epoch 14
  Best val loss: 34856.XXXX (epoch 14)
  Early stopping counter: X/20

Starting training for 160 epochs (from epoch 14)
Cyclical β-annealing: 4 cycles × 40 epochs
```

## Troubleshooting

**If checkpoint not found:**
- Check the path is correct
- Ensure you're in the Capstone directory
- Verify the checkpoint file exists

**If CUDA out of memory:**
- Reduce batch_size to 128 or 64
- Restart with the same checkpoint

**If metrics look wrong:**
- Verify you're using the correct checkpoint path
- Check training history is continuous (no gaps in epochs)
