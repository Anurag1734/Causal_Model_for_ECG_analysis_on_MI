"""Quick diagnostic to check for data issues that might cause NaN."""

import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from data.ecg_dataset import ECGDataset, create_train_val_test_splits, create_dataloaders

print("=" * 80)
print("Data Diagnostics for NaN Issue")
print("=" * 80)

# Load metadata
metadata_path = Path('data/processed/ecg_features_with_demographics.parquet')
base_path = Path('data/raw/MIMIC-IV-ECG-1.0/files')

print(f"\n[OK] Loading metadata from {metadata_path}")
df = pd.read_parquet(metadata_path)

# Filter for training data
train_labels = ['Control_Symptomatic', 'MI_Pre-Incident']
df_train = df[df['Label'].isin(train_labels)].copy()
print(f"[OK] Training set: {len(df_train)} ECGs")

# Create splits
train_df, val_df, test_df = create_train_val_test_splits(df_train)

print(f"[OK] Train split: {len(train_df)} samples")

# Create small dataset to check first 10 samples
train_dataset = ECGDataset(train_df, base_path=str(base_path))

print(f"\n" + "=" * 80)
print("Checking first 10 samples for data issues...")
print("=" * 80)

issues = []
for i in range(min(10, len(train_dataset))):
    try:
        signal, metadata = train_dataset[i]
        
        # Check for NaN
        if torch.isnan(signal).any():
            issues.append(f"Sample {i}: Contains NaN values")
            print(f"  [{i}] ❌ NaN detected")
            continue
        
        # Check for Inf
        if torch.isinf(signal).any():
            issues.append(f"Sample {i}: Contains Inf values")
            print(f"  [{i}] ❌ Inf detected")
            continue
        
        # Check range
        min_val = signal.min().item()
        max_val = signal.max().item()
        mean_val = signal.mean().item()
        std_val = signal.std().item()
        
        if abs(min_val) > 100:
            issues.append(f"Sample {i}: Extreme min value {min_val:.2f}")
            print(f"  [{i}] ⚠️  Extreme min: {min_val:.2f}")
        elif abs(max_val) > 100:
            issues.append(f"Sample {i}: Extreme max value {max_val:.2f}")
            print(f"  [{i}] ⚠️  Extreme max: {max_val:.2f}")
        else:
            print(f"  [{i}] ✓ OK - Range: [{min_val:.2f}, {max_val:.2f}], Mean: {mean_val:.2f}, Std: {std_val:.2f}")
    
    except Exception as e:
        issues.append(f"Sample {i}: Failed to load - {str(e)}")
        print(f"  [{i}] ❌ Error: {str(e)}")

print(f"\n" + "=" * 80)
print("Summary")
print("=" * 80)

if issues:
    print(f"❌ Found {len(issues)} issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✓ All samples checked - no data issues detected")

# Test a small forward pass
print(f"\n" + "=" * 80)
print("Testing small batch forward pass...")
print("=" * 80)

from models.vae_conv1d import Conv1DVAE

model = Conv1DVAE(z_dim=64, beta=4.0)
model.eval()

# Get a batch of 4 samples
batch_signals = []
for i in range(4):
    signal, _ = train_dataset[i]
    batch_signals.append(signal)

batch = torch.stack(batch_signals)
print(f"✓ Created batch: {batch.shape}")
print(f"  Min: {batch.min():.4f}, Max: {batch.max():.4f}")
print(f"  Mean: {batch.mean():.4f}, Std: {batch.std():.4f}")

with torch.no_grad():
    x_recon, mu, logvar = model(batch)
    loss_dict = model.loss_function(batch, x_recon, mu, logvar)

print(f"\n✓ Forward pass successful!")
print(f"  Recon loss: {loss_dict['recon_loss'].item():.4f}")
print(f"  KL loss: {loss_dict['kl_loss'].item():.4f}")
print(f"  Total loss: {loss_dict['loss'].item():.4f}")

if torch.isnan(loss_dict['loss']):
    print(f"\n❌ NaN detected in loss!")
    print(f"  x_recon stats: min={x_recon.min():.4f}, max={x_recon.max():.4f}, has_nan={torch.isnan(x_recon).any()}")
    print(f"  mu stats: min={mu.min():.4f}, max={mu.max():.4f}, has_nan={torch.isnan(mu).any()}")
    print(f"  logvar stats: min={logvar.min():.4f}, max={logvar.max():.4f}, has_nan={torch.isnan(logvar).any()}")
else:
    print(f"✓ No NaN in loss")

print(f"\n" + "=" * 80)
print("Diagnosis complete!")
print("=" * 80)
