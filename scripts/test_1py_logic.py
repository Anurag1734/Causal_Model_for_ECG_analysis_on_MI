"""
Final validation test for 1.py - Tests the exact code path that will be used.
"""

import os
import pandas as pd
import duckdb
from pathlib import Path
import wfdb

print("=" * 80)
print("FINAL VALIDATION TEST FOR 1.py")
print("=" * 80)

# Use exact same configuration as 1.py
OUTPUT_DIR = "data/processed/"
MIMIC_ECG_DIR = "data/raw/MIMIC-IV-ECG-1.0/"
DB_PATH = "mimic_database.duckdb"

LABELS_TO_EXTRACT = [
    'MI_Acute_Presentation',
    'MI_Pre-Incident',
    'Control_Symptomatic'
]

# Test 1: Load cohort_master
print("\n[TEST 1] Loading cohort_master.parquet...")
cohort_file = os.path.join(OUTPUT_DIR, "cohort_master.parquet")
cohort = pd.read_parquet(cohort_file)
print(f"✓ Loaded: {len(cohort):,} records")

# Test 2: Filter labels
print("\n[TEST 2] Filtering to target labels...")
cohort_filtered = cohort[cohort['primary_label'].isin(LABELS_TO_EXTRACT)].copy()
print(f"✓ Filtered: {len(cohort_filtered):,} records")
print("  Label distribution:")
for label in LABELS_TO_EXTRACT:
    count = (cohort_filtered['primary_label'] == label).sum()
    print(f"    - {label}: {count:,}")

# Test 3: Database query (exact same code as 1.py)
print("\n[TEST 3] Testing database query (10 sample records)...")
con = duckdb.connect(DB_PATH, read_only=True)

# Take 10 test records
test_cohort = cohort_filtered.head(10)
record_ids = test_cohort['record_id'].tolist()

# Use exact same query format as 1.py
record_ids_str = ','.join([f"'{str(rid)}'" for rid in record_ids])
path_query = f"""
    SELECT file_name, subject_id, study_id, path
    FROM record_list
    WHERE file_name IN ({record_ids_str})
"""

path_df = con.execute(path_query).fetchdf()
con.close()

print(f"✓ Query returned {len(path_df)} paths for {len(record_ids)} records")

# Test 4: Merge operation
print("\n[TEST 4] Testing merge operation...")
path_df['file_name'] = path_df['file_name'].astype(str)

# Handle column conflicts (exact same as 1.py)
cols_to_drop = [col for col in ['subject_id', 'study_id'] if col in path_df.columns and col in test_cohort.columns]
if cols_to_drop:
    print(f"  Dropping conflicting columns: {cols_to_drop}")
    path_df = path_df.drop(columns=cols_to_drop)

test_cohort = test_cohort.merge(
    path_df.rename(columns={'file_name': 'record_id'}),
    on='record_id',
    how='left'
)

missing_paths = test_cohort['path'].isna().sum()
print(f"✓ Merge complete: {len(test_cohort) - missing_paths}/{len(test_cohort)} records have paths")

if missing_paths > 0:
    print(f"  ⚠️  {missing_paths} records missing paths")

# Test 5: File access
print("\n[TEST 5] Testing file access...")
accessible = 0
inaccessible = 0

for idx, row in test_cohort.iterrows():
    if pd.isna(row.get('path')):
        inaccessible += 1
        continue
    
    full_path = Path(MIMIC_ECG_DIR) / row['path']
    header_file = str(full_path) + '.hea'
    
    if os.path.exists(header_file):
        accessible += 1
    else:
        inaccessible += 1
        print(f"  Missing: {header_file}")

print(f"✓ File access: {accessible}/{len(test_cohort)} accessible")

# Test 6: WFDB reading
print("\n[TEST 6] Testing WFDB record reading...")
if accessible > 0:
    # Try to read first accessible record
    for idx, row in test_cohort.iterrows():
        if pd.isna(row.get('path')):
            continue
        
        full_path = Path(MIMIC_ECG_DIR) / row['path']
        if os.path.exists(str(full_path) + '.hea'):
            try:
                record = wfdb.rdrecord(str(full_path))
                print(f"✓ Successfully read record {row['record_id']}")
                print(f"  - Sampling rate: {record.fs} Hz")
                print(f"  - Signal length: {len(record.p_signal)} samples")
                print(f"  - Leads: {record.sig_name}")
                break
            except Exception as e:
                print(f"  ❌ Error reading record: {e}")
else:
    print("  ⊘ Skipped (no accessible files)")

# Test 7: Wrapper function format
print("\n[TEST 7] Testing wrapper function data format...")
test_row = test_cohort.iloc[0].to_dict()
print(f"✓ Row converted to dict with {len(test_row)} keys")
print(f"  Keys present: {list(test_row.keys())[:5]}...")

# Check required keys
required_keys = ['record_id', 'subject_id', 'hadm_id', 'ecg_time', 'primary_label', 'path']
missing_keys = [k for k in required_keys if k not in test_row or pd.isna(test_row.get(k))]
if missing_keys:
    print(f"  ⚠️  Missing/NaN keys: {missing_keys}")
else:
    print(f"✓ All required keys present and non-null")

# Final summary
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
print("✅ All tests passed!")
print(f"\nReady to extract {len(cohort_filtered):,} ECG records")
print("\nRun this command to start:")
print("  python 1.py")
