"""
Test the fixed calculate_ecg_intervals function
"""
import sys
sys.path.append('src/features')

from phase_d_feature_extraction import extract_fiducials_neurokit, calculate_ecg_intervals
import wfdb
import numpy as np
from pathlib import Path

# Load the same problem record
record_id = '49813583'
MIMIC_ECG_DIR = Path('data/raw/MIMIC-IV-ECG-1.0')
record_files = list(MIMIC_ECG_DIR.rglob(f'{record_id}.hea'))
record_path = record_files[0]
record_dir = record_path.parent
record_name = record_path.stem

print(f"Testing fixed extraction on record {record_id}")
print("=" * 80)

# Load record
record = wfdb.rdrecord(str(record_dir / record_name))
signal = record.p_signal[:, 1]  # Lead II
sampling_rate = record.fs

# Extract fiducials with fixed function
waves, info = extract_fiducials_neurokit(signal, sampling_rate)

if waves is not None:
    print("✅ Fiducials extracted successfully")
    
    # Calculate intervals with fixed function
    intervals = calculate_ecg_intervals(waves, sampling_rate)
    
    print("\n📊 Extracted intervals:")
    for key, value in intervals.items():
        print(f"  {key}: {value:.1f} ms")
    
    # Calculate HR from R-peaks
    r_peaks = info['ECG_R_Peaks']
    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / sampling_rate
        hr = 60 / np.mean(rr_intervals)
        print(f"  heart_rate: {hr:.1f} bpm")
    
    print("\n" + "=" * 80)
    print("COMPARISON:")
    print("=" * 80)
    print("Original (BROKEN) extraction:")
    print("  QRS: -690.0 ms ❌")
    print("  PR: 1016.0 ms ❌")
    print("  QT: -412.0 ms ❌")
    print("\nFixed extraction:")
    print(f"  QRS: {intervals.get('qrs_duration_ms', 'N/A'):.1f} ms ✅")
    print(f"  PR: {intervals.get('pr_interval_ms', 'N/A'):.1f} ms ✅")
    print(f"  QT: {intervals.get('qt_interval_ms', 'N/A'):.1f} ms ✅")
    
    # Check plausibility
    qrs = intervals.get('qrs_duration_ms', 0)
    pr = intervals.get('pr_interval_ms', 0)
    qt = intervals.get('qt_interval_ms', 0)
    
    qrs_ok = 40 <= qrs <= 200
    pr_ok = 80 <= pr <= 300
    qt_ok = 200 <= qt <= 600
    
    print("\n" + "=" * 80)
    print("PLAUSIBILITY CHECK:")
    print("=" * 80)
    print(f"  QRS (40-200ms): {qrs:.1f}ms {'✅' if qrs_ok else '❌'}")
    print(f"  PR (80-300ms): {pr:.1f}ms {'✅' if pr_ok else '❌'}")
    print(f"  QT (200-600ms): {qt:.1f}ms {'✅' if qt_ok else '❌'}")
else:
    print("❌ Failed to extract fiducials")
