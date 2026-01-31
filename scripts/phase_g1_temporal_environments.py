"""
Phase G.1 Alternative: Define Temporal Environments

Since MIMIC-IV-ECG headers don't contain machine model information,
we use temporal splits as environments (Plan B from protocol).

Rationale:
- Temporal shifts create distribution shifts similar to environment shifts
- Changing population demographics and practice patterns over time
- IRM will learn features robust to temporal distribution shift

Environment Assignment:
- Environment 0: Early period (2110-2142)
- Environment 1: Middle period (2143-2174)  
- Environment 2: Recent period (2175-2207)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
MASTER_DATASET_PATH = "data/processed/master_dataset.parquet"
OUTPUT_PATH = "data/processed/master_dataset.parquet"

print("\n" + "=" * 80)
print("Phase G.1 Alternative: Define Temporal Environments")
print("=" * 80)

# Load master dataset
print(f"\nLoading master dataset from {MASTER_DATASET_PATH}...")
df = pd.read_parquet(MASTER_DATASET_PATH)
print(f"✓ Loaded {len(df)} records")

# Check anchor_year availability
if 'anchor_year' not in df.columns:
    print("\n✗ ERROR: anchor_year column not found!")
    print("  Cannot create temporal environments")
    exit(1)

# Display year distribution
print("\nAnchor Year Distribution:")
year_counts = df['anchor_year'].value_counts().sort_index()
for year, count in year_counts.items():
    pct = 100 * count / len(df)
    print(f"  {year}: {count:6d} ({pct:5.1f}%)")

# Define temporal periods
def assign_temporal_environment(year):
    """
    Assign temporal environment based on anchor year.
    
    Note: MIMIC-IV anchor_year is obfuscated (shifted for privacy).
    Actual range: 2110-2207 (98 years of data).
    
    0: Early period (2110-2142) - ~33 years
    1: Middle period (2143-2174) - ~32 years
    2: Recent period (2175-2207) - ~33 years
    """
    if pd.isna(year):
        return 2  # Default to recent if missing
    elif year <= 2142:
        return 0
    elif year <= 2174:
        return 1
    else:
        return 2

df['environment_label'] = df['anchor_year'].apply(assign_temporal_environment)

# Create human-readable environment names
env_names = {
    0: 'Early (2110-2142)',
    1: 'Middle (2143-2174)',
    2: 'Recent (2175-2207)'
}
df['environment_name'] = df['environment_label'].map(env_names)
df['machine_model'] = df['environment_name']  # For compatibility

print("\n✓ Temporal environments assigned")

# Display environment distribution
print("\nTemporal Environment Distribution:")
env_counts = df['environment_label'].value_counts().sort_index()
for env, count in env_counts.items():
    env_name = env_names[env]
    pct = 100 * count / len(df)
    print(f"  {env} ({env_name:25s}): {count:6d} ({pct:5.1f}%)")

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
    print(f"  {env_name:25s}: {count:6d} samples - {status}")

# Save updated master dataset
print(f"\nSaving updated master dataset to {OUTPUT_PATH}...")
df.to_parquet(OUTPUT_PATH, index=False)

print(f"\n✓ Master dataset updated with temporal environment labels")
print(f"  Records: {len(df)}")
print(f"  Columns updated: machine_model, environment_label, environment_name")

print("\n" + "=" * 80)
print("✓ Phase G.1 (Temporal): Environment Definition - COMPLETE")
print("=" * 80)

print("\nNote:")
print("  Using temporal split as environment proxy (Plan B)")
print("  IRM will learn features robust to temporal distribution shifts")
print("\nNext Steps:")
print("  1. Run Phase G.2: python scripts/phase_g2_check_environments.py")
print("  2. Run Phase G.3: python scripts/phase_g3_train_irm.py")
