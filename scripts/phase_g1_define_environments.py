"""
Phase G.1: Define Environment Labels

Extract ECG machine model from WFDB header files as proxy for systematic measurement differences.

Protocol:
1. Parse .hea header files using wfdb
2. Extract ECG machine model from comments
3. Map to environment labels:
   - 0: 'TC50' (PageWriter TC50, 0.05 Hz high-pass filter)
   - 1: 'TC70' (PageWriter TC70, 0.15 Hz high-pass filter)
   - 2: 'Other' (Philips XML, GE MUSE, etc.)

Rationale: Filter differences create spurious correlations with ST-segment baseline.
IRM will learn to ignore these artifacts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import wfdb
from tqdm import tqdm
import re

# Paths
MASTER_DATASET_PATH = "data/processed/master_dataset.parquet"
BASE_PATH = Path("data/raw/MIMIC-IV-ECG-1.0/files")
OUTPUT_PATH = "data/processed/master_dataset.parquet"

print("\n" + "=" * 80)
print("Phase G.1: Define Environment Labels")
print("=" * 80)

# Load master dataset
print(f"\nLoading master dataset from {MASTER_DATASET_PATH}...")
df = pd.read_parquet(MASTER_DATASET_PATH)
print(f"✓ Loaded {len(df)} records")

# Function to extract machine model from WFDB header
def extract_machine_model(file_path: str, base_path: Path) -> str:
    """
    Extract ECG machine model from WFDB header file.
    
    Parameters
    ----------
    file_path : str
        Relative path to WFDB file (e.g., "p10/p10020306/s41256771/41256771")
    base_path : Path
        Base directory containing WFDB files
    
    Returns
    -------
    machine_model : str
        Machine model name (e.g., "PageWriter TC50", "PageWriter TC70", "Other")
    """
    try:
        # Construct full path (remove extension if present)
        full_path = base_path / file_path
        if full_path.suffix:
            full_path = full_path.with_suffix('')
        
        # Read header
        header = wfdb.rdheader(str(full_path))
        
        # Extract comments (where machine info is stored)
        comments = header.comments if hasattr(header, 'comments') else []
        
        # Search for machine model in comments
        for comment in comments:
            comment_lower = comment.lower()
            
            # Check for PageWriter models
            if 'pagewriter tc50' in comment_lower or 'tc50' in comment_lower:
                return 'PageWriter TC50'
            elif 'pagewriter tc70' in comment_lower or 'tc70' in comment_lower:
                return 'PageWriter TC70'
            elif 'pagewriter' in comment_lower:
                return 'PageWriter Other'
            
            # Check for other common models
            elif 'philips' in comment_lower or 'xml' in comment_lower:
                return 'Philips XML'
            elif 'ge' in comment_lower or 'muse' in comment_lower:
                return 'GE MUSE'
            elif 'schiller' in comment_lower:
                return 'Schiller'
        
        # If no specific model found, check recorder field
        if hasattr(header, 'recorder'):
            recorder = header.recorder.lower() if header.recorder else ''
            if 'tc50' in recorder:
                return 'PageWriter TC50'
            elif 'tc70' in recorder:
                return 'PageWriter TC70'
            elif 'pagewriter' in recorder:
                return 'PageWriter Other'
            elif 'philips' in recorder or 'xml' in recorder:
                return 'Philips XML'
            elif 'ge' in recorder or 'muse' in recorder:
                return 'GE MUSE'
        
        # Default to "Unknown"
        return 'Unknown'
        
    except Exception as e:
        # If header cannot be read, return Unknown
        return 'Unknown'

# Extract machine models
print("\nExtracting ECG machine models from WFDB headers...")
print("(This may take several minutes for ~48K records)")

machine_models = []
failed_count = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing headers"):
    file_path = row['file_path']
    
    try:
        model = extract_machine_model(file_path, BASE_PATH)
        machine_models.append(model)
    except Exception as e:
        machine_models.append('Unknown')
        failed_count += 1

df['machine_model'] = machine_models

print(f"\n✓ Extracted machine models")
print(f"  Failed to parse: {failed_count}/{len(df)} ({100*failed_count/len(df):.1f}%)")

# Display machine model distribution
print("\nMachine Model Distribution:")
model_counts = df['machine_model'].value_counts()
for model, count in model_counts.items():
    pct = 100 * count / len(df)
    print(f"  {model:30s}: {count:6d} ({pct:5.1f}%)")

# Map to environment labels
print("\nMapping to environment labels...")

def map_to_environment(model: str) -> int:
    """
    Map machine model to environment label.
    
    0: TC50 (0.05 Hz high-pass filter)
    1: TC70 (0.15 Hz high-pass filter)
    2: Other
    """
    if 'TC50' in model:
        return 0
    elif 'TC70' in model:
        return 1
    else:
        return 2

df['environment_label'] = df['machine_model'].apply(map_to_environment)

# Create human-readable environment names
env_names = {0: 'TC50', 1: 'TC70', 2: 'Other'}
df['environment_name'] = df['environment_label'].map(env_names)

print("\n✓ Environment labels assigned")

# Display environment distribution
print("\nEnvironment Distribution:")
env_counts = df['environment_label'].value_counts().sort_index()
for env, count in env_counts.items():
    env_name = env_names[env]
    pct = 100 * count / len(df)
    print(f"  {env} ({env_name:10s}): {count:6d} ({pct:5.1f}%)")

# Check environment × label distribution
print("\nEnvironment × Label Distribution:")
env_label_counts = df.groupby(['environment_name', 'Label']).size().unstack(fill_value=0)
print(env_label_counts)

# Check if environments meet minimum sample requirement
print("\nEnvironment Sample Size Check (minimum: 500 samples):")
min_samples = 500
for env, count in env_counts.items():
    env_name = env_names[env]
    status = "✓ PASS" if count >= min_samples else "✗ FAIL"
    print(f"  {env_name:10s}: {count:6d} samples - {status}")

# Save updated master dataset
print(f"\nSaving updated master dataset to {OUTPUT_PATH}...")
df.to_parquet(OUTPUT_PATH, index=False)

print(f"\n✓ Master dataset updated with environment labels")
print(f"  Records: {len(df)}")
print(f"  New columns: machine_model, environment_label, environment_name")

print("\n" + "=" * 80)
print("✓ Phase G.1: Environment Definition - COMPLETE")
print("=" * 80)

print("\nNext Steps:")
print("  1. Proceed to Phase G.2: Check environment distribution")
print("  2. Proceed to Phase G.3: Train IRM model")
