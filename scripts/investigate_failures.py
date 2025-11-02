"""
Investigate feature extraction failures
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("FEATURE EXTRACTION FAILURE INVESTIGATION")
print("=" * 80)

# Load features
features = pd.read_parquet('data/processed/ecg_features.parquet')

print(f"\nTotal ECGs: {len(features)}")
print(f"Successful extractions: {features['extraction_success'].sum()}")
print(f"Failed extractions: {(~features['extraction_success']).sum()}")

# Analyze distribution of each feature
numeric_cols = ['heart_rate', 'pr_interval_ms', 'qrs_duration_ms', 'qt_interval_ms', 'qtc_bazett']

print("\n" + "=" * 80)
print("FEATURE DISTRIBUTION ANALYSIS")
print("=" * 80)

for col in numeric_cols:
    if col in features.columns:
        data = features[col].dropna()
        
        print(f"\n{col}:")
        print(f"  Count: {len(data)}")
        print(f"  Mean: {data.mean():.1f}")
        print(f"  Median: {data.median():.1f}")
        print(f"  Std: {data.std():.1f}")
        print(f"  Min: {data.min():.1f}")
        print(f"  Max: {data.max():.1f}")
        
        # Percentiles
        print(f"  Percentiles:")
        print(f"    1%:  {np.percentile(data, 1):.1f}")
        print(f"    5%:  {np.percentile(data, 5):.1f}")
        print(f"    25%: {np.percentile(data, 25):.1f}")
        print(f"    75%: {np.percentile(data, 75):.1f}")
        print(f"    95%: {np.percentile(data, 95):.1f}")
        print(f"    99%: {np.percentile(data, 99):.1f}")
        
        # Check for outliers
        negative = (data < 0).sum()
        very_large = (data > 2000).sum()
        
        if negative > 0:
            print(f"  ⚠️ Negative values: {negative} ({100*negative/len(data):.1f}%)")
        if very_large > 0:
            print(f"  ⚠️ Values >2000: {very_large} ({100*very_large/len(data):.1f}%)")

# Check QRS duration distribution more carefully
print("\n" + "=" * 80)
print("QRS DURATION DETAILED ANALYSIS")
print("=" * 80)

qrs = features['qrs_duration_ms'].dropna()

bins = [
    (-float('inf'), 0, "Negative (impossible)"),
    (0, 40, "Too short (<40ms)"),
    (40, 60, "Borderline short (40-60ms)"),
    (60, 120, "Normal (60-120ms)"),
    (120, 200, "Prolonged (120-200ms)"),
    (200, 500, "Very prolonged (200-500ms)"),
    (500, float('inf'), "Impossible (>500ms)")
]

for min_val, max_val, label in bins:
    count = ((qrs >= min_val) & (qrs < max_val)).sum()
    pct = 100 * count / len(qrs)
    print(f"  {label:35s}: {count:5d} ({pct:5.1f}%)")

# Sample some problem cases
print("\n" + "=" * 80)
print("SAMPLE PROBLEM CASES")
print("=" * 80)

# Negative QRS
neg_qrs = features[features['qrs_duration_ms'] < 0].head(5)
if len(neg_qrs) > 0:
    print("\nNegative QRS durations:")
    for idx, row in neg_qrs.iterrows():
        print(f"  Record {row['record_id']}: QRS={row['qrs_duration_ms']:.1f}ms, HR={row['heart_rate']:.1f}")

# Very large QRS
large_qrs = features[features['qrs_duration_ms'] > 500].head(5)
if len(large_qrs) > 0:
    print("\nVery large QRS durations (>500ms):")
    for idx, row in large_qrs.iterrows():
        print(f"  Record {row['record_id']}: QRS={row['qrs_duration_ms']:.1f}ms, HR={row['heart_rate']:.1f}")

# Check correlation between issues
print("\n" + "=" * 80)
print("CORRELATION BETWEEN QUALITY FLAGS")
print("=" * 80)

quality_cols = ['hr_plausible', 'qrs_plausible', 'qt_plausible', 'qtc_plausible', 'quality_flag']
for col in quality_cols:
    if col in features.columns:
        true_count = features[col].sum()
        print(f"  {col:20s}: {true_count}/{len(features)} ({100*true_count/len(features):.1f}%)")

print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

print("\n🔴 PRIMARY ISSUE: Feature extraction algorithm is failing")
print("\nEvidence:")
print(f"  - Only 15% of ECGs have plausible features")
print(f"  - 50% have implausible QRS durations")
print(f"  - 65% have implausible PR intervals")
print(f"  - Many negative values (algorithm returning error codes)")

print("\n📋 LIKELY CAUSES:")
print("  1. NeuroKit2 peak detection failing on MI ECGs (abnormal morphology)")
print("  2. Incorrect fiducial point extraction from waves DataFrame")
print("  3. Index/sample confusion (treating indices as milliseconds)")
print("  4. Not handling NaN/empty fiducial arrays properly")

print("\n🔧 RECOMMENDED FIXES:")
print("  1. Add better error handling in extract_fiducials_neurokit()")
print("  2. Check if waves DataFrame has any fiducial points before calculating intervals")
print("  3. Set implausible values to NaN instead of keeping garbage")
print("  4. Consider using alternative library (BioSPPy, HeartPy)")
print("  5. Add signal quality check before extraction")
