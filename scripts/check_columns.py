"""
Check actual columns in ecg_features_with_demographics.parquet
"""
import pandas as pd

print("=" * 80)
print("ACTUAL COLUMNS IN ecg_features_with_demographics.parquet")
print("=" * 80)

df = pd.read_parquet('data/processed/ecg_features_with_demographics.parquet')

print(f"\nTotal rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

print("\n" + "=" * 80)
print("COLUMN NAMES")
print("=" * 80)
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    non_null = df[col].notna().sum()
    print(f"  {i:2d}. {col:30s} | Type: {str(dtype):10s} | Non-null: {non_null:,}")

print("\n" + "=" * 80)
print("SAMPLE ROW (first record)")
print("=" * 80)
for col in df.columns:
    val = df[col].iloc[0]
    print(f"  {col:30s} = {val}")

print("\n" + "=" * 80)
print("NUMERIC COLUMN STATISTICS")
print("=" * 80)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    if col not in ['record_id', 'subject_id', 'hadm_id']:
        print(f"\n{col}:")
        print(f"  Mean:  {df[col].mean():8.2f}")
        print(f"  Std:   {df[col].std():8.2f}")
        print(f"  Min:   {df[col].min():8.2f}")
        print(f"  Max:   {df[col].max():8.2f}")
        print(f"  NaN:   {df[col].isna().sum():,}")

print("\n" + "=" * 80)
print("LABEL DISTRIBUTION")
print("=" * 80)
if 'primary_label' in df.columns:
    print(df['primary_label'].value_counts())

print("\n" + "=" * 80)
print("MISSING COLUMNS CHECK")
print("=" * 80)

expected_cols = [
    'record_id', 'subject_id', 'hadm_id', 'ecg_time',
    'heart_rate', 'pr_interval_ms', 'qrs_duration_ms', 'qt_interval_ms',
    'qtc_bazett', 'qtc_fridericia', 'rr_variability_ms',
    'st_deviation', 't_wave_inverted', 'q_wave_present',
    'primary_label'
]

missing = [col for col in expected_cols if col not in df.columns]
extra = [col for col in df.columns if col not in expected_cols]

if missing:
    print(f"\n❌ MISSING columns:")
    for col in missing:
        print(f"  - {col}")
else:
    print(f"\n✅ All expected columns present")

if extra:
    print(f"\n➕ EXTRA columns (not in expected list):")
    for col in extra:
        print(f"  - {col}")
