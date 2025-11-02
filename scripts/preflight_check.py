"""
Pre-flight check for Phase D extraction.
Validates all requirements before starting the full extraction.
"""

import os
import pandas as pd
import duckdb
from pathlib import Path

print("=" * 80)
print("PHASE D PRE-FLIGHT CHECK")
print("=" * 80)

# Configuration
OUTPUT_DIR = "data/processed/"
MIMIC_ECG_DIR = "data/raw/MIMIC-IV-ECG-1.0/"
DB_PATH = "mimic_database.duckdb"

LABELS_TO_EXTRACT = [
    'MI_Acute_Presentation',
    'MI_Pre-Incident',
    'Control_Symptomatic'
]

checks_passed = 0
checks_failed = 0

# Check 1: cohort_master.parquet exists
print("\n[1/7] Checking cohort_master.parquet...")
cohort_file = os.path.join(OUTPUT_DIR, "cohort_master.parquet")
if os.path.exists(cohort_file):
    cohort = pd.read_parquet(cohort_file)
    print(f"  ✓ Found: {len(cohort):,} records")
    checks_passed += 1
else:
    print(f"  ❌ MISSING: {cohort_file}")
    checks_failed += 1
    cohort = None

# Check 2: Verify labels exist
if cohort is not None:
    print("\n[2/7] Checking label distribution...")
    label_counts = cohort['primary_label'].value_counts()
    print("  Label counts:")
    for label in LABELS_TO_EXTRACT:
        count = label_counts.get(label, 0)
        print(f"    - {label}: {count:,}")
        if count == 0:
            print(f"      ❌ WARNING: No records for {label}")
            checks_failed += 1
        else:
            checks_passed += 1
else:
    print("\n[2/7] ❌ Skipping (no cohort loaded)")
    checks_failed += 3

# Check 3: Database exists
print("\n[3/7] Checking DuckDB database...")
if os.path.exists(DB_PATH):
    con = duckdb.connect(DB_PATH, read_only=True)
    record_count = con.execute("SELECT COUNT(*) FROM record_list").fetchone()[0]
    print(f"  ✓ Found: {record_count:,} records in database")
    con.close()
    checks_passed += 1
else:
    print(f"  ❌ MISSING: {DB_PATH}")
    checks_failed += 1

# Check 4: MIMIC-IV-ECG directory exists
print("\n[4/7] Checking MIMIC-IV-ECG directory...")
if os.path.exists(MIMIC_ECG_DIR):
    print(f"  ✓ Found: {MIMIC_ECG_DIR}")
    checks_passed += 1
else:
    print(f"  ❌ MISSING: {MIMIC_ECG_DIR}")
    checks_failed += 1

# Check 5: Test database query
print("\n[5/7] Testing database query...")
if cohort is not None and os.path.exists(DB_PATH):
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        cohort_filtered = cohort[cohort['primary_label'].isin(LABELS_TO_EXTRACT)]
        test_ids = cohort_filtered['record_id'].head(10).tolist()
        test_ids_str = ','.join([f"'{str(rid)}'" for rid in test_ids])
        
        query = f"""
            SELECT file_name, path
            FROM record_list
            WHERE file_name IN ({test_ids_str})
        """
        
        result = con.execute(query).fetchdf()
        con.close()
        
        if len(result) > 0:
            print(f"  ✓ Query successful: Found {len(result)} paths for 10 test records")
            checks_passed += 1
        else:
            print(f"  ❌ Query returned 0 results")
            checks_failed += 1
    except Exception as e:
        print(f"  ❌ Query failed: {e}")
        checks_failed += 1
else:
    print("  ⊘ Skipping (prerequisites missing)")

# Check 6: Test file access
print("\n[6/7] Testing file access...")
if cohort is not None and os.path.exists(DB_PATH) and os.path.exists(MIMIC_ECG_DIR):
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        cohort_filtered = cohort[cohort['primary_label'].isin(LABELS_TO_EXTRACT)]
        test_ids = cohort_filtered['record_id'].head(5).tolist()
        test_ids_str = ','.join([f"'{str(rid)}'" for rid in test_ids])
        
        query = f"""
            SELECT file_name, path
            FROM record_list
            WHERE file_name IN ({test_ids_str})
        """
        
        result = con.execute(query).fetchdf()
        con.close()
        
        accessible = 0
        for _, row in result.iterrows():
            full_path = Path(MIMIC_ECG_DIR) / row['path']
            header_file = str(full_path) + '.hea'
            if os.path.exists(header_file):
                accessible += 1
        
        if accessible > 0:
            print(f"  ✓ File access successful: {accessible}/{len(result)} test files accessible")
            checks_passed += 1
        else:
            print(f"  ❌ No test files accessible")
            checks_failed += 1
    except Exception as e:
        print(f"  ❌ File access test failed: {e}")
        checks_failed += 1
else:
    print("  ⊘ Skipping (prerequisites missing)")

# Check 7: Estimate extraction size
print("\n[7/7] Estimating extraction workload...")
if cohort is not None:
    cohort_filtered = cohort[cohort['primary_label'].isin(LABELS_TO_EXTRACT)]
    total_records = len(cohort_filtered)
    print(f"  Total records to extract: {total_records:,}")
    
    # Estimate time (0.87 seconds per ECG, with 15 cores)
    from multiprocessing import cpu_count
    n_workers = max(1, cpu_count() - 1)
    est_hours = (total_records * 0.87) / n_workers / 3600
    
    print(f"  Using {n_workers} CPU cores")
    print(f"  Estimated time: {est_hours:.1f} hours")
    
    # Estimate output size (assume 2KB per record)
    est_size_mb = (total_records * 2) / 1024
    print(f"  Estimated output size: {est_size_mb:.1f} MB")
    
    checks_passed += 1
else:
    print("  ⊘ Skipping (no cohort loaded)")

# Summary
print("\n" + "=" * 80)
print("PRE-FLIGHT CHECK SUMMARY")
print("=" * 80)
print(f"Checks passed: {checks_passed}")
print(f"Checks failed: {checks_failed}")

if checks_failed == 0:
    print("\n✅ ALL CHECKS PASSED - Ready to run Phase D extraction")
    print("\nRun this command to start:")
    print("  python 1.py")
else:
    print(f"\n❌ {checks_failed} CHECKS FAILED - Fix issues before running")
    print("\nPlease resolve the issues above before proceeding.")
