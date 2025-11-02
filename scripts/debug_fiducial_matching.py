"""
Debug the fiducial matching logic
"""
import sys
sys.path.append('src/features')

from phase_d_feature_extraction import extract_fiducials_neurokit
import wfdb
import numpy as np
from pathlib import Path

# Load record
record_id = '49813583'
MIMIC_ECG_DIR = Path('data/raw/MIMIC-IV-ECG-1.0')
record_files = list(MIMIC_ECG_DIR.rglob(f'{record_id}.hea'))
record_path = record_files[0]
record_dir = record_path.parent
record_name = record_path.stem

record = wfdb.rdrecord(str(record_dir / record_name))
signal = record.p_signal[:, 1]
sampling_rate = record.fs

# Extract fiducials
waves, info = extract_fiducials_neurokit(signal, sampling_rate)

def get_fiducial_indices(waves_dict, key):
    """Extract non-NaN fiducial indices from dict."""
    if key not in waves_dict:
        return np.array([])
    values = waves_dict[key]
    return np.array([int(v) for v in values if not np.isnan(v)])

# Extract fiducial indices
qrs_onsets = get_fiducial_indices(waves, 'ECG_R_Onsets')
qrs_offsets = get_fiducial_indices(waves, 'ECG_R_Offsets')
t_offsets = get_fiducial_indices(waves, 'ECG_T_Offsets')
p_onsets = get_fiducial_indices(waves, 'ECG_P_Onsets')

print("Fiducial Points:")
print(f"  P onsets:  {p_onsets}")
print(f"  R onsets:  {qrs_onsets}")
print(f"  R offsets: {qrs_offsets}")
print(f"  T offsets: {t_offsets}")

# Calculate QRS manually
print("\nManual QRS calculation:")
for i, r_on in enumerate(qrs_onsets):
    r_offs_after = qrs_offsets[qrs_offsets > r_on]
    if len(r_offs_after) > 0:
        duration_samples = r_offs_after[0] - r_on
        duration_ms = (duration_samples / sampling_rate) * 1000
        print(f"  Beat {i}: R_onset={r_on}, R_offset={r_offs_after[0]}, Duration={duration_samples} samples = {duration_ms:.1f}ms")

print(f"\nSampling rate: {sampling_rate} Hz")
print(f"So 1 sample = {1000/sampling_rate:.2f} ms")
