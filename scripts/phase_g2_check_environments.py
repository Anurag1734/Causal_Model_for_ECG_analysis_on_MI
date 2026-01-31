"""
Phase G.2: Check Environment Distribution

Verify that each environment has sufficient samples for reliable IRM training.

Requirements:
- Each environment should have ≥500 samples
- Balanced label distribution across environments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
MASTER_DATASET_PATH = "data/processed/master_dataset.parquet"

print("\n" + "=" * 80)
print("Phase G.2: Check Environment Distribution")
print("=" * 80)

# Load master dataset
print(f"\nLoading master dataset from {MASTER_DATASET_PATH}...")
df = pd.read_parquet(MASTER_DATASET_PATH)
print(f"✓ Loaded {len(df)} records")

# Check if environment labels exist
if 'environment_label' not in df.columns:
    print("\n✗ ERROR: environment_label column not found!")
    print("  Run Phase G.1 first: python scripts/phase_g1_define_environments.py")
    exit(1)

print("\n" + "=" * 80)
print("Environment Distribution Analysis")
print("=" * 80)

# Overall environment distribution
print("\n1. Overall Environment Distribution:")
env_counts = df['environment_label'].value_counts().sort_index()
env_names = {0: 'TC50', 1: 'TC70', 2: 'Other'}

for env, count in env_counts.items():
    env_name = env_names.get(env, f'Env_{env}')
    pct = 100 * count / len(df)
    print(f"   Environment {env} ({env_name:10s}): {count:7d} samples ({pct:5.1f}%)")

# Environment × Label cross-tabulation
print("\n2. Environment × Label Cross-Tabulation:")
if 'environment_name' in df.columns:
    crosstab = pd.crosstab(df['environment_name'], df['Label'], margins=True)
else:
    crosstab = pd.crosstab(df['environment_label'], df['Label'], margins=True)
print(crosstab)

# Label proportions within each environment
print("\n3. Label Proportions Within Each Environment:")
if 'environment_name' in df.columns:
    env_label_pct = pd.crosstab(
        df['environment_name'], 
        df['Label'], 
        normalize='index'
    ) * 100
else:
    env_label_pct = pd.crosstab(
        df['environment_label'], 
        df['Label'], 
        normalize='index'
    ) * 100
print(env_label_pct.round(1))

# Check minimum sample requirements
print("\n4. Sample Size Requirements Check:")
min_samples = 500
min_samples_per_class = 100

print(f"\n   Minimum total samples per environment: {min_samples}")
print(f"   Minimum samples per class per environment: {min_samples_per_class}")

all_pass = True

for env in env_counts.index:
    env_name = env_names.get(env, f'Env_{env}')
    total = env_counts[env]
    
    # Check total samples
    if total < min_samples:
        print(f"\n   ✗ Environment {env} ({env_name}): {total} total samples < {min_samples}")
        all_pass = False
    else:
        print(f"\n   ✓ Environment {env} ({env_name}): {total} total samples ≥ {min_samples}")
    
    # Check per-class samples
    env_df = df[df['environment_label'] == env]
    class_counts = env_df['Label'].value_counts()
    
    for label, count in class_counts.items():
        if count < min_samples_per_class:
            print(f"     ✗ {label:30s}: {count:5d} samples < {min_samples_per_class}")
            all_pass = False
        else:
            print(f"     ✓ {label:30s}: {count:5d} samples ≥ {min_samples_per_class}")

# Overall verdict
print("\n" + "=" * 80)
if all_pass:
    print("✓ VERDICT: All environments meet minimum requirements")
    print("  Proceed to Phase G.3: Train IRM model")
else:
    print("✗ VERDICT: Some environments do not meet requirements")
    print("  Consider:")
    print("    1. Merging small environments into 'Other' category")
    print("    2. Using only environments with sufficient samples")
    print("    3. Using temporal split as fallback (Plan B)")

print("=" * 80)

# Train/Test split distribution
if 'split' in df.columns:
    print("\n5. Environment Distribution by Train/Test Split:")
    split_env_crosstab = pd.crosstab(
        df['split'],
        df['environment_label' if 'environment_label' in df.columns else 'environment_name'],
        margins=True
    )
    print(split_env_crosstab)
    
    print("\n   Label Distribution in Train/Test:")
    for split in ['train', 'test']:
        split_df = df[df['split'] == split]
        print(f"\n   {split.upper()}:")
        label_counts = split_df['Label'].value_counts()
        for label, count in label_counts.items():
            pct = 100 * count / len(split_df)
            print(f"     {label:30s}: {count:6d} ({pct:5.1f}%)")

# Demographics by environment
print("\n6. Demographics by Environment:")
if 'age' in df.columns and 'sex' in df.columns:
    for env in env_counts.index:
        env_name = env_names.get(env, f'Env_{env}')
        env_df = df[df['environment_label'] == env]
        
        mean_age = env_df['age'].mean()
        std_age = env_df['age'].std()
        n_male = (env_df['sex'] == 'M').sum()
        n_female = (env_df['sex'] == 'F').sum()
        pct_male = 100 * n_male / len(env_df)
        
        print(f"\n   Environment {env} ({env_name}):")
        print(f"     Age: {mean_age:.1f} ± {std_age:.1f} years")
        print(f"     Sex: {pct_male:.1f}% Male, {100-pct_male:.1f}% Female")

print("\n" + "=" * 80)
print("✓ Phase G.2: Environment Distribution Check - COMPLETE")
print("=" * 80)
