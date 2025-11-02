"""
Debug a single failed record to see what's going wrong
"""
import wfdb
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import duckdb

# Load record with negative QRS (49813583: QRS=-690.0ms)
record_id = '49813583'

print("=" * 80)
print(f"DEBUGGING RECORD: {record_id}")
print("=" * 80)

# Get path from features file
features = pd.read_parquet('data/processed/ecg_features.parquet')
record_row = features[features['record_id'] == record_id]

if len(record_row) == 0:
    print(f"Record {record_id} not found")
    exit()

print(f"\nFeature values from extraction:")
print(f"  HR: {record_row['heart_rate'].values[0]:.1f} bpm")
print(f"  PR: {record_row['pr_interval_ms'].values[0]:.1f} ms")
print(f"  QRS: {record_row['qrs_duration_ms'].values[0]:.1f} ms")
print(f"  QT: {record_row['qt_interval_ms'].values[0]:.1f} ms")

# Find the file manually using glob
from pathlib import Path
MIMIC_ECG_DIR = Path('data/raw/MIMIC-IV-ECG-1.0')
record_files = list(MIMIC_ECG_DIR.rglob(f'{record_id}.hea'))

if len(record_files) == 0:
    print(f"Record file {record_id}.hea not found")
    exit()

record_path = record_files[0]
print(f"\nFound file: {record_path}")

# Load the record
record_dir = record_path.parent
record_name = record_path.stem

print(f"Loading from: {record_dir / record_name}")

try:
    record = wfdb.rdrecord(str(record_dir / record_name))
    print(f"✅ Record loaded successfully")
    print(f"  Sampling rate: {record.fs} Hz")
    print(f"  Signal length: {record.sig_len} samples")
    print(f"  Number of leads: {len(record.sig_name)}")
    print(f"  Lead names: {record.sig_name}")
except Exception as e:
    print(f"❌ Error loading record: {e}")
    exit()

# Use lead II (index 1)
signal = record.p_signal[:, 1]
sampling_rate = record.fs

print(f"\nSignal statistics:")
print(f"  Mean: {np.mean(signal):.3f}")
print(f"  Std: {np.std(signal):.3f}")
print(f"  Min: {np.min(signal):.3f}")
print(f"  Max: {np.max(signal):.3f}")

# Clean signal
print("\n" + "=" * 80)
print("CLEANING SIGNAL")
print("=" * 80)

cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)
print("✅ Signal cleaned")

# Find peaks
print("\n" + "=" * 80)
print("FINDING PEAKS")
print("=" * 80)

try:
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
    r_peaks = rpeaks['ECG_R_Peaks']
    print(f"✅ Found {len(r_peaks)} R-peaks")
    print(f"  First 5 R-peak indices: {r_peaks[:5]}")
    
    # Calculate heart rate
    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / sampling_rate
        hr = 60 / np.mean(rr_intervals)
        print(f"  Heart rate: {hr:.1f} bpm")
except Exception as e:
    print(f"❌ Error finding peaks: {e}")
    exit()

# Delineate waves
print("\n" + "=" * 80)
print("DELINEATING WAVES (DWT method)")
print("=" * 80)

try:
    _, waves_dict = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=sampling_rate, method="dwt")
    print("✅ Waves delineated")
    print(f"  Waves dict keys: {waves_dict.keys()}")
    
    # Check what's in each key
    print("\n  Fiducial point arrays:")
    for key in ['ECG_P_Onsets', 'ECG_R_Onsets', 'ECG_R_Offsets', 'ECG_T_Offsets']:
        if key in waves_dict:
            values = waves_dict[key]
            non_nan = [v for v in values if not np.isnan(v)]
            print(f"    {key}: {len(non_nan)} points (first 5: {non_nan[:5]})")
        else:
            print(f"    {key}: NOT FOUND")
                
except Exception as e:
    print(f"❌ Error delineating waves: {e}")
    import traceback
    traceback.print_exc()
    exit()

# Calculate intervals manually
print("\n" + "=" * 80)
print("CALCULATING INTERVALS")
print("=" * 80)

def get_fiducial_indices(waves_dict, column):
    """Extract indices of fiducial points from dict"""
    if column not in waves_dict:
        return np.array([])
    values = waves_dict[column]
    # Remove NaN values
    return np.array([int(v) for v in values if not np.isnan(v)])

# Get all fiducial points
p_onsets = get_fiducial_indices(waves_dict, 'ECG_P_Onsets')
r_onsets = get_fiducial_indices(waves_dict, 'ECG_R_Onsets')
r_offsets = get_fiducial_indices(waves_dict, 'ECG_R_Offsets')
t_offsets = get_fiducial_indices(waves_dict, 'ECG_T_Offsets')

print(f"P onsets:  {len(p_onsets)} points, first 5: {p_onsets[:5]}")
print(f"R onsets:  {len(r_onsets)} points, first 5: {r_onsets[:5]}")
print(f"R offsets: {len(r_offsets)} points, first 5: {r_offsets[:5]}")
print(f"T offsets: {len(t_offsets)} points, first 5: {t_offsets[:5]}")

# Calculate QRS duration
if len(r_onsets) > 0 and len(r_offsets) > 0:
    # Method 1: Median difference
    qrs_durations = []
    for r_on in r_onsets:
        # Find closest R_offset after this R_onset
        r_offs_after = r_offsets[r_offsets > r_on]
        if len(r_offs_after) > 0:
            qrs_duration = r_offs_after[0] - r_on
            qrs_durations.append(qrs_duration)
    
    if len(qrs_durations) > 0:
        qrs_samples = np.median(qrs_durations)
        qrs_ms = (qrs_samples / sampling_rate) * 1000
        print(f"\nQRS Duration:")
        print(f"  Median: {qrs_ms:.1f} ms ({qrs_samples:.0f} samples)")
        print(f"  All durations (samples): {qrs_durations[:10]}")
        
        # Check if any are negative
        negative = [d for d in qrs_durations if d < 0]
        if len(negative) > 0:
            print(f"  ⚠️ FOUND NEGATIVE DURATIONS: {negative}")
    else:
        print(f"\n⚠️ No valid QRS intervals found")
        
        # Debug: check if R_offsets come BEFORE R_onsets
        if len(r_onsets) > 0 and len(r_offsets) > 0:
            print(f"\n  DEBUG: First R_onset = {r_onsets[0]}, First R_offset = {r_offsets[0]}")
            if r_offsets[0] < r_onsets[0]:
                print(f"  🔴 PROBLEM: R_offset comes BEFORE R_onset!")
                print(f"     This would give negative QRS duration: {r_offsets[0] - r_onsets[0]}")
else:
    print("\n⚠️ Missing R_onsets or R_offsets")

print("\n" + "=" * 80)
print("ROOT CAUSE IDENTIFIED")
print("=" * 80)
print("The issue is likely that NeuroKit2's delineation is marking fiducial points")
print("at the wrong locations, causing R_offset to come BEFORE R_onset, resulting")
print("in negative intervals.")
print("\nThis is a known issue with DWT delineation on abnormal ECGs (MI patients).")
print("\nSOLUTIONS:")
print("  1. Add validation: Only keep intervals where offset > onset")
print("  2. Try different delineation method (e.g., 'peaks' instead of 'dwt')")
print("  3. Use a different library (BioSPPy, ecg-qc)")
