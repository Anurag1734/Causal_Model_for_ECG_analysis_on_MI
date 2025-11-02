"""
Clean ECG features by filtering out physiologically implausible values
"""
import pandas as pd
import numpy as np

print("=" * 80)
print("ECG FEATURES CLEANING")
print("=" * 80)

# Load extracted features
features = pd.read_parquet('data/processed/ecg_features.parquet')
print(f"\n1. Initial features: {len(features)}")

# Display current statistics
print("\n2. Current feature statistics (before cleaning):")
numeric_cols = ['heart_rate', 'pr_interval_ms', 'qrs_duration_ms', 'qt_interval_ms', 'qtc_bazett', 'qtc_fridericia']
for col in numeric_cols:
    if col in features.columns:
        print(f"   {col:20s} - Mean: {features[col].mean():8.1f}, Min: {features[col].min():8.1f}, Max: {features[col].max():8.1f}")

# Define physiologically plausible ranges
plausible_ranges = {
    'heart_rate': (30, 200),
    'pr_interval_ms': (80, 300),
    'qrs_duration_ms': (40, 200),
    'qt_interval_ms': (200, 600),
    'qtc_bazett': (300, 700),
    'qtc_fridericia': (300, 700)
}

print("\n3. Applying plausibility filters:")
print("   Ranges:")
for feature, (min_val, max_val) in plausible_ranges.items():
    print(f"   - {feature:20s}: {min_val:6.0f} - {max_val:6.0f}")

# Create mask for plausible values
plausible_mask = np.ones(len(features), dtype=bool)

for feature, (min_val, max_val) in plausible_ranges.items():
    if feature in features.columns:
        feature_mask = (features[feature] >= min_val) & (features[feature] <= max_val)
        plausible_mask &= feature_mask
        removed = (~feature_mask).sum()
        print(f"\n   {feature}: Removed {removed} ({100*removed/len(features):.1f}%)")

# Filter to plausible features only
features_clean = features[plausible_mask].copy()

print("\n" + "=" * 80)
print(f"4. After plausibility filter: {len(features_clean)} ({100*len(features_clean)/len(features):.1f}%)")
print("=" * 80)

# Display cleaned statistics
print("\n5. Cleaned feature statistics:")
for col in numeric_cols:
    if col in features_clean.columns:
        data = features_clean[col].dropna()
        if len(data) > 0:
            print(f"   {col:20s} - Mean: {data.mean():8.1f}, Std: {data.std():8.1f}, Min: {data.min():8.1f}, Max: {data.max():8.1f}")

# Quality control summary
if 'quality_flag' in features_clean.columns:
    high_quality = features_clean['quality_flag'].sum()
    print(f"\n6. Quality Control:")
    print(f"   High Quality ECGs: {high_quality}/{len(features_clean)} ({100*high_quality/len(features_clean):.1f}%)")

# Label distribution
if 'primary_label' in features_clean.columns:
    print(f"\n7. Label Distribution (Clean):")
    label_counts = features_clean['primary_label'].value_counts()
    for label, count in label_counts.items():
        print(f"   {label}: {count}")

# Save cleaned features
output_path = 'data/processed/ecg_features_clean.parquet'
features_clean.to_parquet(output_path, index=False)
print(f"\n✓ Saved cleaned features to: {output_path}")

# Power analysis
min_required = 500
print("\n" + "=" * 80)
print("POWER ANALYSIS")
print("=" * 80)
print(f"Minimum required: {min_required}")
print(f"Clean sample size: {len(features_clean)}")
print(f"Power multiplier: {len(features_clean)/min_required:.1f}x")

if len(features_clean) >= 5000:
    print("\n✅ EXCELLENT POWER (10x minimum) - Proceed to Phase E")
elif len(features_clean) >= 3000:
    print("\n✅ GOOD POWER (6x minimum) - Proceed to Phase E")
elif len(features_clean) >= 1000:
    print("\n⚠️ MODERATE POWER (2x minimum) - Some subgroups underpowered")
else:
    print("\n🔴 INSUFFICIENT POWER - Must fix feature extractor")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
if len(features_clean) >= 3000:
    print("1. ✅ Sufficient power - ready for Phase E")
    print("2. Optional: Validate on PTB-XL+ to verify accuracy")
    print("3. Proceed to VAE training")
else:
    print("1. 🔴 Investigate feature extraction failures")
    print("2. Run PTB-XL+ validation to diagnose issues")
    print("3. Consider using different ECG processing library")
    print("4. Re-run feature extraction after fixes")
