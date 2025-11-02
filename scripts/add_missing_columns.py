"""
Add missing columns (file_path, study_id, Label) to ecg_features_with_demographics.parquet
for VAE training compatibility.
"""

import pandas as pd
import duckdb

print("=" * 80)
print("Adding Missing Columns for VAE Training")
print("=" * 80)

# Load current metadata
print("\n✓ Loading ecg_features_with_demographics.parquet...")
df = pd.read_parquet('data/processed/ecg_features_with_demographics.parquet')
print(f"  Current shape: {df.shape}")
print(f"  Current columns: {list(df.columns[:5])}...")

# Connect to database
print("\n✓ Connecting to database...")
con = duckdb.connect('mimic_database.duckdb')

# Get file paths from record_list
print("\n✓ Fetching file paths from record_list...")
record_info = con.execute("""
    SELECT 
        CAST(file_name AS BIGINT) as record_id,
        study_id,
        path as file_path
    FROM record_list
""").fetchdf()

print(f"  Found {len(record_info)} records in database")

# Convert record_id to same type
print("\n✓ Converting record_id to int64...")
df['record_id'] = df['record_id'].astype('int64')
record_info['record_id'] = record_info['record_id'].astype('int64')

# Merge with current data
print("\n✓ Merging file paths with current data...")
df_merged = df.merge(record_info, on='record_id', how='left')
print(f"  Merged shape: {df_merged.shape}")

# Rename primary_label to Label
print("\n✓ Renaming primary_label to Label...")
df_merged = df_merged.rename(columns={'primary_label': 'Label'})

# Check for missing values
missing_file_path = df_merged['file_path'].isna().sum()
missing_study_id = df_merged['study_id'].isna().sum()

if missing_file_path > 0:
    print(f"  ⚠ Warning: {missing_file_path} records have missing file_path")
if missing_study_id > 0:
    print(f"  ⚠ Warning: {missing_study_id} records have missing study_id")

# Save updated file
output_path = 'data/processed/ecg_features_with_demographics.parquet'
print(f"\n✓ Saving updated file to {output_path}...")
df_merged.to_parquet(output_path, index=False)

# Summary
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"✓ Final shape: {df_merged.shape}")
print(f"✓ Columns added: file_path, study_id")
print(f"✓ Column renamed: primary_label → Label")
print(f"\n✓ Label distribution:")
print(df_merged['Label'].value_counts())
print(f"\n✓ Sample file_path values:")
print(df_merged['file_path'].head(3).tolist())

print("\n" + "=" * 80)
print("✓ Done! Run pre-flight check again:")
print("  python scripts/preflight_vae_training.py")
print("=" * 80)
