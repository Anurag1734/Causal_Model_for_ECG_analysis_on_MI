"""
Phase F: Master Dataset Creation

Steps:
F.1: Join all tables (cohort, ECG features, VAE latent embeddings, clinical features)
F.2: Handle missing data (MICE imputation + missing indicators)
F.3: Normalize continuous features (z-score)
F.4: Save master dataset

Output: master_dataset.parquet (single source of truth)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm

# Paths
ECG_FEATURES_PATH = "data/processed/ecg_features_with_demographics.parquet"
CLINICAL_FEATURES_PATH = "data/processed/clinical_features.parquet"
VAE_EMBEDDINGS_PATH = "models/vae_latent_embeddings.parquet"  # To be created
OUTPUT_PATH = "data/processed/master_dataset.parquet"
IMPUTER_PATH = "models/imputer.pkl"
SCALER_PATH = "models/scaler.pkl"

print("\n" + "=" * 80)
print("Phase F: Master Dataset Creation")
print("=" * 80)

# =============================================================================
# F.0: Generate VAE Latent Embeddings (if not exists)
# =============================================================================

if not Path(VAE_EMBEDDINGS_PATH).exists():
    print("\nGenerating VAE latent embeddings...")
    print("NOTE: This requires running VAE encoding on all ECG signals")
    print("Creating placeholder for now - you'll need to run encoding script")
    
    # For now, create empty placeholder
    # In production, you'd run: python scripts/encode_ecgs_to_latent.py
    print(f"⚠ WARNING: {VAE_EMBEDDINGS_PATH} not found!")
    print("  Create it using: python scripts/encode_ecgs_to_latent.py")
    print("  Continuing without latent features for now...")
    vae_embeddings_available = False
else:
    vae_embeddings_available = True

# =============================================================================
# F.1: Join All Tables
# =============================================================================

print("\n" + "=" * 80)
print("Phase F.1: Join All Tables")
print("=" * 80)

print("\nLoading datasets...")

# Load ECG features (this serves as cohort master)
print(f"  - ECG features: {ECG_FEATURES_PATH}")
df_ecg = pd.read_parquet(ECG_FEATURES_PATH)
print(f"    Loaded {len(df_ecg)} records")

# Load clinical features
print(f"  - Clinical features: {CLINICAL_FEATURES_PATH}")
df_clinical = pd.read_parquet(CLINICAL_FEATURES_PATH)
print(f"    Loaded {len(df_clinical)} records")

# Load VAE embeddings (if available)
if vae_embeddings_available:
    print(f"  - VAE embeddings: {VAE_EMBEDDINGS_PATH}")
    df_vae = pd.read_parquet(VAE_EMBEDDINGS_PATH)
    print(f"    Loaded {len(df_vae)} records")
else:
    df_vae = None

# Merge datasets
print("\nMerging datasets on record_id...")
df_master = df_ecg.copy()

# Merge clinical features
df_master = df_master.merge(
    df_clinical.drop(columns=['subject_id', 'hadm_id', 'ecg_time', 'admittime', 'age', 'sex'], errors='ignore'),
    on='record_id',
    how='left',
    suffixes=('', '_clinical')
)

# Merge VAE embeddings
if df_vae is not None:
    df_master = df_master.merge(
        df_vae,
        on='record_id',
        how='left',
        suffixes=('', '_vae')
    )

print(f"\n✓ Master dataset shape: {df_master.shape}")
print(f"  Records: {len(df_master)}")
print(f"  Features: {len(df_master.columns)}")

# =============================================================================
# F.2: Handle Missing Data
# =============================================================================

print("\n" + "=" * 80)
print("Phase F.2: Handle Missing Data")
print("=" * 80)

# Identify features for imputation (continuous variables only)
# Exclude identifiers, labels, binary flags, categorical variables
exclude_cols = [
    'record_id', 'subject_id', 'hadm_id', 'study_id', 'file_path',
    'ecg_time', 'admittime', 'anchor_year',
    'Label', 'sex',  # Categorical
    'sampling_rate', 'signal_length',  # Metadata
    't_wave_inverted', 'q_wave_present',  # Binary clinical
    'baseline_wander', 'hr_plausible', 'qrs_plausible', 'qt_plausible', 'qtc_plausible',
    'quality_flag', 'extraction_success',  # Quality flags
    'statin_use', 'diabetes', 'hypertension', 'ckd', 'comorbidity_chronic_mi'  # Binary comorbidities
]

# Get continuous feature columns
continuous_cols = [
    col for col in df_master.columns 
    if col not in exclude_cols 
    and not col.endswith('_missing')  # Exclude missing indicator columns
    and df_master[col].dtype in ['float64', 'int64']
    and col.startswith(('age', 'heart_rate', 'pr_interval', 'qrs_duration', 'qt_interval', 
                        'qtc', 'rr_variability', 'st_deviation',
                        'ldl', 'hdl', 'total_chol', 'triglycerides',
                        'creatinine', 'bun', 'glucose', 'potassium', 'bnp', 'nt_probnp',
                        'sbp', 'dbp', 'rr', 'spo2', 'temperature',
                        'z_ecg'))  # Include VAE latent features
]

print(f"\nIdentified {len(continuous_cols)} continuous features for imputation")

# Calculate missingness rates
print("\nMissingness Analysis:")
missing_rates = {}
for col in continuous_cols:
    if col in df_master.columns:
        missing_rate = df_master[col].isna().sum() / len(df_master)
        missing_rates[col] = missing_rate
        if missing_rate > 0.10:  # Report high-missingness features
            print(f"  {col:30s}: {missing_rate:6.1%} missing")

# Create missing indicators for features with >10% missingness
print("\nCreating missing indicators for high-missingness features...")
for col, missing_rate in missing_rates.items():
    if missing_rate > 0.10:
        indicator_col = f"{col}_missing"
        if indicator_col not in df_master.columns:
            df_master[indicator_col] = df_master[col].isna().astype(int)
            print(f"  Created: {indicator_col}")

# Separate data for imputation
print("\nPreparing data for imputation...")
df_to_impute = df_master[continuous_cols].copy()

# Check if we have enough data for MICE
n_complete = df_to_impute.notna().all(axis=1).sum()
complete_rate = n_complete / len(df_to_impute)
print(f"  Complete cases: {n_complete}/{len(df_to_impute)} ({complete_rate:.1%})")

# Perform imputation
if complete_rate > 0.10:  # Need at least 10% complete cases for MICE
    print("\nPerforming Multiple Imputation by Chained Equations (MICE)...")
    print("  (This may take several minutes for large datasets)")
    
    imputer = IterativeImputer(
        random_state=42,
        max_iter=10,
        verbose=0,
        imputation_order='ascending'
    )
    
    try:
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df_to_impute),
            columns=continuous_cols,
            index=df_to_impute.index
        )
        print("✓ MICE imputation completed")
        
        # Save imputer
        with open(IMPUTER_PATH, 'wb') as f:
            pickle.dump(imputer, f)
        print(f"✓ Imputer saved to {IMPUTER_PATH}")
        
    except Exception as e:
        print(f"⚠ MICE imputation failed: {e}")
        print("  Falling back to median imputation...")
        df_imputed = df_to_impute.fillna(df_to_impute.median())
        imputer = None
else:
    print("\n⚠ Too few complete cases for MICE (<10%)")
    print("  Using median imputation...")
    df_imputed = df_to_impute.fillna(df_to_impute.median())
    imputer = None

# Special handling for LDL (if missing and statin_use = 1)
if 'ldl' in df_imputed.columns and 'statin_use' in df_master.columns:
    print("\nSpecial imputation for LDL (statin users)...")
    
    # For patients on statins with missing LDL, impute higher values
    # (patients on statins likely had elevated LDL)
    statin_mask = (df_master['statin_use'] == 1) & (df_master['ldl'].isna())
    n_statin_imputed = statin_mask.sum()
    
    if n_statin_imputed > 0:
        # Impute LDL for statin users as 75th percentile (higher than median)
        ldl_75th = df_imputed['ldl'].quantile(0.75)
        df_imputed.loc[statin_mask, 'ldl'] = ldl_75th
        print(f"  Imputed {n_statin_imputed} statin users with LDL={ldl_75th:.2f} (75th percentile)")

# Replace imputed values in master dataset
for col in continuous_cols:
    if col in df_imputed.columns:
        df_master[col] = df_imputed[col]

# Verify no missing values remain in imputed columns
remaining_missing = df_master[continuous_cols].isna().sum().sum()
print(f"\n✓ Remaining missing values in continuous features: {remaining_missing}")

# =============================================================================
# F.3: Normalize Continuous Features
# =============================================================================

print("\n" + "=" * 80)
print("Phase F.3: Normalize Continuous Features")
print("=" * 80)

# Split data for normalization (use stratified split to preserve label distribution)
from sklearn.model_selection import train_test_split

# Get label for stratification
if 'Label' in df_master.columns:
    labels = df_master['Label']
else:
    labels = None

# Split 80/20 for train/test
train_indices, test_indices = train_test_split(
    df_master.index,
    test_size=0.2,
    random_state=42,
    stratify=labels if labels is not None else None
)

df_master['split'] = 'test'
df_master.loc[train_indices, 'split'] = 'train'

print(f"\nSplit sizes:")
print(f"  Train: {len(train_indices)} ({100*len(train_indices)/len(df_master):.1f}%)")
print(f"  Test:  {len(test_indices)} ({100*len(test_indices)/len(df_master):.1f}%)")

# Fit scaler on training data only
print("\nFitting StandardScaler on training data...")
scaler = StandardScaler()

X_train = df_master.loc[train_indices, continuous_cols]
scaler.fit(X_train)

# Transform all data
print("Applying z-score normalization...")
X_normalized = scaler.transform(df_master[continuous_cols])

# Create normalized column names
normalized_cols = [f"{col}_norm" for col in continuous_cols]

# Add normalized features to master dataset
df_normalized = pd.DataFrame(
    X_normalized,
    columns=normalized_cols,
    index=df_master.index
)

df_master = pd.concat([df_master, df_normalized], axis=1)

# Save scaler
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"\n✓ Scaler saved to {SCALER_PATH}")

# Print normalization statistics
print("\nNormalization Statistics (Training Set):")
for i, col in enumerate(continuous_cols[:10]):  # Show first 10
    mean = scaler.mean_[i]
    std = scaler.scale_[i]
    print(f"  {col:30s}: mean={mean:8.2f}, std={std:8.2f}")
if len(continuous_cols) > 10:
    print(f"  ... and {len(continuous_cols)-10} more features")

# =============================================================================
# F.4: Save Master Dataset
# =============================================================================

print("\n" + "=" * 80)
print("Phase F.4: Save Master Dataset")
print("=" * 80)

# Organize columns for final dataset
identifier_cols = ['record_id', 'subject_id', 'hadm_id', 'study_id']
label_col = ['Label']
split_col = ['split']
demographic_cols = ['age', 'sex', 'anchor_year']

# ECG features (original)
ecg_feature_cols = [
    col for col in df_master.columns
    if col.startswith(('heart_rate', 'pr_interval', 'qrs_duration', 'qt_interval',
                       'qtc', 'rr_variability', 'st_deviation', 't_wave', 'q_wave'))
    and not col.endswith('_norm')
]

# ECG metadata
ecg_metadata_cols = [
    'ecg_time', 'sampling_rate', 'signal_length', 'file_path',
    'baseline_wander', 'hr_plausible', 'qrs_plausible', 'qt_plausible', 
    'qtc_plausible', 'quality_flag', 'extraction_success'
]

# VAE latent features
vae_cols = [col for col in df_master.columns if col.startswith('z_ecg_') and not col.endswith('_norm')]

# Clinical features (original)
clinical_lab_cols = [
    col for col in df_master.columns
    if col in ['ldl', 'hdl', 'total_chol', 'triglycerides',
               'creatinine', 'bun', 'glucose', 'potassium', 'bnp', 'nt_probnp']
]

clinical_vital_cols = [
    col for col in df_master.columns
    if col in ['sbp', 'dbp', 'rr', 'spo2', 'temperature']
]

clinical_binary_cols = [
    col for col in df_master.columns
    if col in ['statin_use', 'diabetes', 'hypertension', 'ckd', 'comorbidity_chronic_mi']
]

# Missing indicators
missing_indicator_cols = [col for col in df_master.columns if col.endswith('_missing')]

# Normalized features
normalized_feature_cols = [col for col in df_master.columns if col.endswith('_norm')]

# Combine in logical order
final_columns = (
    [col for col in identifier_cols if col in df_master.columns] +
    [col for col in label_col if col in df_master.columns] +
    [col for col in split_col if col in df_master.columns] +
    [col for col in demographic_cols if col in df_master.columns] +
    [col for col in ecg_feature_cols if col in df_master.columns] +
    vae_cols +
    clinical_lab_cols +
    clinical_vital_cols +
    clinical_binary_cols +
    missing_indicator_cols +
    normalized_feature_cols +
    [col for col in ecg_metadata_cols if col in df_master.columns]
)

# Remove duplicates while preserving order
seen = set()
final_columns = [col for col in final_columns if not (col in seen or seen.add(col))]

# Create final dataset
df_final = df_master[final_columns].copy()

# Save
print(f"\nSaving master dataset to {OUTPUT_PATH}...")
df_final.to_parquet(OUTPUT_PATH, index=False)

print(f"\n✓ Master dataset saved!")
print(f"  Path: {OUTPUT_PATH}")
print(f"  Records: {len(df_final)}")
print(f"  Features: {len(df_final.columns)}")

# =============================================================================
# Summary Statistics
# =============================================================================

print("\n" + "=" * 80)
print("Master Dataset Summary")
print("=" * 80)

print(f"\nDataset Shape: {df_final.shape}")

print("\nColumn Groups:")
print(f"  Identifiers:        {len([c for c in final_columns if c in identifier_cols])}")
print(f"  Labels:             {len([c for c in final_columns if c in label_col])}")
print(f"  Demographics:       {len([c for c in final_columns if c in demographic_cols])}")
print(f"  ECG Features:       {len(ecg_feature_cols)}")
print(f"  VAE Latent:         {len(vae_cols)}")
print(f"  Clinical Labs:      {len(clinical_lab_cols)}")
print(f"  Clinical Vitals:    {len(clinical_vital_cols)}")
print(f"  Comorbidities:      {len(clinical_binary_cols)}")
print(f"  Missing Indicators: {len(missing_indicator_cols)}")
print(f"  Normalized:         {len(normalized_feature_cols)}")
print(f"  Metadata:           {len([c for c in final_columns if c in ecg_metadata_cols])}")

print("\nLabel Distribution:")
if 'Label' in df_final.columns:
    label_counts = df_final['Label'].value_counts()
    for label, count in label_counts.items():
        pct = 100 * count / len(df_final)
        print(f"  {label:30s}: {count:6d} ({pct:5.1f}%)")

print("\nSplit Distribution:")
if 'split' in df_final.columns:
    split_counts = df_final['split'].value_counts()
    for split, count in split_counts.items():
        pct = 100 * count / len(df_final)
        print(f"  {split:10s}: {count:6d} ({pct:5.1f}%)")

print("\nSample Features (first 5 rows):")
print(df_final[['record_id', 'Label', 'age', 'sex', 'heart_rate', 'ldl', 'statin_use', 'diabetes']].head())

# Create complete-case dataset for sensitivity analysis
df_complete_case = df_final[
    df_final[[col for col in continuous_cols if col in df_final.columns]].notna().all(axis=1)
].copy()

complete_case_path = OUTPUT_PATH.replace('.parquet', '_complete_case.parquet')
df_complete_case.to_parquet(complete_case_path, index=False)

print(f"\n✓ Complete-case dataset saved for sensitivity analysis:")
print(f"  Path: {complete_case_path}")
print(f"  Records: {len(df_complete_case)} ({100*len(df_complete_case)/len(df_final):.1f}% of full dataset)")

print("\n" + "=" * 80)
print("✓ Phase F: Master Dataset - COMPLETE")
print("=" * 80)

print("\nNext Steps:")
print("  1. Generate VAE latent embeddings (if not done):")
print("     python scripts/encode_ecgs_to_latent.py")
print("  2. Proceed to Phase G: Environment Assignment")
print("  3. Sensitivity analysis: Compare imputed vs complete-case results in Phase J")
