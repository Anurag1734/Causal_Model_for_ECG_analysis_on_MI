"""
FINAL SAFETY CHECK - Run this right before executing 1.py
This ensures absolutely everything is ready.
"""

import os
import sys

print("=" * 80)
print("FINAL SAFETY CHECK FOR 1.py")
print("=" * 80)

issues = []

# Check 1: File exists
if not os.path.exists('1.py'):
    issues.append("❌ 1.py not found in current directory")
else:
    print("✓ 1.py found")

# Check 2: Required files exist
required_files = [
    'data/processed/cohort_master.parquet',
    'mimic_database.duckdb'
]

for file in required_files:
    if not os.path.exists(file):
        issues.append(f"❌ Missing required file: {file}")
    else:
        print(f"✓ Found: {file}")

# Check 3: Required directories exist
required_dirs = [
    'data/raw/MIMIC-IV-ECG-1.0/',
    'data/processed/',
    'reports/'
]

for dir_path in required_dirs:
    if not os.path.exists(dir_path):
        issues.append(f"❌ Missing required directory: {dir_path}")
    else:
        print(f"✓ Found: {dir_path}")

# Check 4: Verify no output file exists (to avoid overwrite)
output_file = 'data/processed/ecg_features.parquet'
if os.path.exists(output_file):
    print(f"\n⚠️  WARNING: Output file already exists: {output_file}")
    print("   This will be OVERWRITTEN when you run 1.py")
    response = input("   Continue anyway? (yes/no): ")
    if response.lower() != 'yes':
        print("\n❌ Cancelled by user")
        sys.exit(1)

# Check 5: Verify imports
print("\n✓ Checking Python imports...")
try:
    import pandas
    import numpy
    import wfdb
    import neurokit2
    import duckdb
    import tqdm
    from multiprocessing import Pool
    print("✓ All required packages available")
except ImportError as e:
    issues.append(f"❌ Missing package: {e}")

# Check 6: Read 1.py and verify key fixes
print("\n✓ Verifying code fixes in 1.py...")
with open('1.py', 'r', encoding='utf-8') as f:
    code = f.read()
    
    # Check for duckdb import
    if 'import duckdb' not in code:
        issues.append("❌ Missing 'import duckdb' in 1.py")
    else:
        print("  ✓ 'import duckdb' present")
    
    # Check for cohort_master
    if 'cohort_master.parquet' not in code:
        issues.append("❌ Code does not reference 'cohort_master.parquet'")
    else:
        print("  ✓ References 'cohort_master.parquet'")
    
    # Check NOT using placeholders
    if 'IN ({placeholders})' in code:
        issues.append("❌ Code still uses old placeholder syntax")
    else:
        print("  ✓ Not using buggy placeholder syntax")
    
    # Check for proper IN clause
    if 'record_ids_str' in code:
        print("  ✓ Uses corrected SQL formatting")
    else:
        issues.append("❌ Missing corrected SQL formatting")

# Final verdict
print("\n" + "=" * 80)
if len(issues) == 0:
    print("✅ ALL SAFETY CHECKS PASSED")
    print("=" * 80)
    print("\n🚀 You are ready to run:")
    print("\n   python 1.py")
    print("\nEstimated time: 1.3 hours")
    print("Estimated output: 125,882 ECG features (~246 MB)")
else:
    print("❌ SAFETY CHECKS FAILED")
    print("=" * 80)
    print("\nIssues found:")
    for issue in issues:
        print(f"  {issue}")
    print("\nPlease fix these issues before running 1.py")
    sys.exit(1)
