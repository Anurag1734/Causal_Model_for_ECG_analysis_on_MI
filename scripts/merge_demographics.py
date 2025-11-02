"""
Merge demographics (age, sex, anchor_year) with ECG features
"""
import pandas as pd
import numpy as np

print("=" * 80)
print("MERGING DEMOGRAPHICS WITH ECG FEATURES")
print("=" * 80)

# Load ECG features
print("\n1. Loading ECG features...")
ecg_features = pd.read_parquet('data/processed/ecg_features_clean.parquet')
print(f"   ✓ Loaded {len(ecg_features):,} ECG records")

# Load cohort master (has demographics)
print("\n2. Loading cohort master with demographics...")
cohort = pd.read_parquet('data/processed/cohort_master.parquet')
print(f"   ✓ Loaded {len(cohort):,} cohort records")

# Check what demographic columns are available
print("\n3. Available columns in cohort:")
demo_cols = ['subject_id', 'anchor_age', 'gender', 'anchor_year']
available_demo = [col for col in demo_cols if col in cohort.columns]
print(f"   Found: {available_demo}")

# Alternative column names
if 'anchor_age' not in cohort.columns and 'age' in cohort.columns:
    available_demo.append('age')
if 'gender' not in cohort.columns and 'sex' in cohort.columns:
    available_demo.append('sex')

print(f"   Using: {available_demo}")

# Select only needed columns from cohort
demo_data = cohort[available_demo].drop_duplicates(subset=['subject_id'])
print(f"\n4. Unique subjects with demographics: {len(demo_data):,}")

# Merge with ECG features
print("\n5. Merging demographics with ECG features...")
ecg_with_demo = ecg_features.merge(
    demo_data,
    on='subject_id',
    how='left'
)

print(f"   ✓ Merged: {len(ecg_with_demo):,} records")

# Check for missing demographics
print("\n6. Checking for missing demographics:")
for col in available_demo:
    if col != 'subject_id':
        missing = ecg_with_demo[col].isna().sum()
        pct = 100 * missing / len(ecg_with_demo)
        if missing > 0:
            print(f"   ⚠️  {col}: {missing:,} missing ({pct:.1f}%)")
        else:
            print(f"   ✓ {col}: All present")

# Rename columns to standard names
rename_map = {
    'anchor_age': 'age',
    'gender': 'sex'
}
ecg_with_demo = ecg_with_demo.rename(columns=rename_map)

# Show demographics statistics
print("\n" + "=" * 80)
print("DEMOGRAPHICS STATISTICS")
print("=" * 80)

if 'age' in ecg_with_demo.columns:
    print(f"\nAge:")
    print(f"  Mean:   {ecg_with_demo['age'].mean():.1f} years")
    print(f"  Std:    {ecg_with_demo['age'].std():.1f} years")
    print(f"  Min:    {ecg_with_demo['age'].min():.0f} years")
    print(f"  Max:    {ecg_with_demo['age'].max():.0f} years")
    print(f"  Median: {ecg_with_demo['age'].median():.1f} years")

if 'sex' in ecg_with_demo.columns:
    print(f"\nSex distribution:")
    sex_counts = ecg_with_demo['sex'].value_counts()
    for sex, count in sex_counts.items():
        pct = 100 * count / len(ecg_with_demo)
        print(f"  {sex}: {count:,} ({pct:.1f}%)")

if 'anchor_year' in ecg_with_demo.columns:
    print(f"\nAnchor Year:")
    print(f"  Min:    {ecg_with_demo['anchor_year'].min():.0f}")
    print(f"  Max:    {ecg_with_demo['anchor_year'].max():.0f}")

# Demographics by label
print("\n" + "=" * 80)
print("DEMOGRAPHICS BY LABEL")
print("=" * 80)

for label in ecg_with_demo['primary_label'].unique():
    label_data = ecg_with_demo[ecg_with_demo['primary_label'] == label]
    print(f"\n{label} (n={len(label_data):,}):")
    
    if 'age' in ecg_with_demo.columns:
        print(f"  Age: {label_data['age'].mean():.1f} ± {label_data['age'].std():.1f} years")
    
    if 'sex' in ecg_with_demo.columns:
        sex_dist = label_data['sex'].value_counts()
        if len(sex_dist) > 0:
            print(f"  Sex: {dict(sex_dist)}")

# Reorder columns for better organization
print("\n" + "=" * 80)
print("REORDERING COLUMNS")
print("=" * 80)

# Define column order
id_cols = ['record_id', 'subject_id', 'hadm_id', 'ecg_time']
demo_cols = [col for col in ['age', 'sex', 'anchor_year'] if col in ecg_with_demo.columns]
feature_cols = [
    'heart_rate', 'pr_interval_ms', 'qrs_duration_ms', 'qt_interval_ms',
    'qtc_bazett', 'qtc_fridericia', 'rr_variability_ms',
    'st_deviation', 't_wave_inverted', 'q_wave_present'
]
label_col = ['primary_label']
other_cols = [col for col in ecg_with_demo.columns 
              if col not in id_cols + demo_cols + feature_cols + label_col]

# Reorder
column_order = id_cols + demo_cols + feature_cols + label_col + other_cols
ecg_with_demo = ecg_with_demo[column_order]

print(f"\nColumn order:")
for i, col in enumerate(column_order[:20], 1):  # Show first 20
    print(f"  {i:2d}. {col}")
if len(column_order) > 20:
    print(f"  ... and {len(column_order) - 20} more")

# Save merged data
output_path = 'data/processed/ecg_features_with_demographics.parquet'
ecg_with_demo.to_parquet(output_path, index=False)

print("\n" + "=" * 80)
print("✅ MERGE COMPLETE")
print("=" * 80)
print(f"\nOutput file: {output_path}")
print(f"Total records: {len(ecg_with_demo):,}")
print(f"Total columns: {len(ecg_with_demo.columns)}")
print(f"\nNew columns added:")
for col in demo_cols:
    if col in ecg_with_demo.columns:
        print(f"  ✓ {col}")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("1. Use 'ecg_features_with_demographics.parquet' for Phase E")
print("2. Demographics are now available for stratified analysis")
print("3. Proceed to VAE training")
