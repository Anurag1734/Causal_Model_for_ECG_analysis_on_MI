"""Filter out corrupted/unreadable WFDB files from metadata."""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path
import wfdb
from tqdm import tqdm

print("=" * 80)
print("Filtering Corrupted WFDB Files")
print("=" * 80)

# Load metadata
metadata_path = Path('data/processed/ecg_features_with_demographics.parquet')
base_path = Path('data/raw/MIMIC-IV-ECG-1.0/files')

print(f"\n[1/4] Loading metadata...")
df = pd.read_parquet(metadata_path)
print(f"  Total records: {len(df)}")

# Test each file
print(f"\n[2/4] Testing WFDB file readability...")
valid_indices = []
failed_files = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Testing files"):
    file_path = row['file_path']
    full_path = base_path / file_path
    
    try:
        # Try to read the header and signal
        record = wfdb.rdrecord(str(full_path))
        signal = record.p_signal
        
        # Check for valid signal
        if signal is None:
            failed_files.append((idx, file_path, "No signal data"))
            continue
        
        # Check for NaN/Inf
        if np.isnan(signal).any() or np.isinf(signal).any():
            failed_files.append((idx, file_path, "Contains NaN/Inf"))
            continue
        
        # Check shape
        if signal.shape[1] != 12:
            failed_files.append((idx, file_path, f"Wrong shape {signal.shape}"))
            continue
        
        # Valid file
        valid_indices.append(idx)
        
    except Exception as e:
        failed_files.append((idx, file_path, str(e)))

print(f"\n[3/4] Results:")
print(f"  Valid files: {len(valid_indices)}")
print(f"  Failed files: {len(failed_files)}")

if failed_files:
    print(f"\n  First 10 failures:")
    for idx, path, reason in failed_files[:10]:
        print(f"    [{idx}] {path}: {reason[:60]}")

# Filter metadata
print(f"\n[4/4] Creating filtered metadata...")
df_filtered = df.iloc[valid_indices].copy()
print(f"  Filtered records: {len(df_filtered)}")

# Save filtered version
output_path = Path('data/processed/ecg_features_with_demographics_filtered.parquet')
df_filtered.to_parquet(output_path, index=False)
print(f"  Saved to: {output_path}")

# Show label distribution
print(f"\n  Label distribution:")
print(df_filtered['Label'].value_counts())

print(f"\n" + "=" * 80)
print("Filtering complete!")
print("=" * 80)
print(f"\nTo use filtered data, update train_vae.py:")
print(f"  --metadata_path data/processed/ecg_features_with_demographics_filtered.parquet")
