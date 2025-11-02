"""
Test what ecg_delineate actually returns
"""
import neurokit2 as nk
import numpy as np
import wfdb
from pathlib import Path

# Load a real ECG
record_id = '49813583'
MIMIC_ECG_DIR = Path('data/raw/MIMIC-IV-ECG-1.0')
record_files = list(MIMIC_ECG_DIR.rglob(f'{record_id}.hea'))
record_path = record_files[0]
record_dir = record_path.parent
record_name = record_path.stem

record = wfdb.rdrecord(str(record_dir / record_name))
signal = record.p_signal[:, 1]
sampling_rate = record.fs

# Process
cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)
peaks, rpeaks_dict = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)

print("=" * 80)
print("WHAT DOES ecg_delineate RETURN?")
print("=" * 80)

# Call ecg_delineate
result = nk.ecg_delineate(cleaned, rpeaks_dict, sampling_rate=sampling_rate, method="dwt")

print(f"\nType of result: {type(result)}")
print(f"Length of result: {len(result)}")

if isinstance(result, tuple):
    print(f"\nIt's a TUPLE with {len(result)} elements:")
    for i, item in enumerate(result):
        print(f"  Element {i}: type={type(item)}")
        if isinstance(item, dict):
            print(f"    Keys: {list(item.keys())[:5]}")
            if 'ECG_R_Onsets' in item:
                print(f"    ECG_R_Onsets type: {type(item['ECG_R_Onsets'])}")
                print(f"    ECG_R_Onsets length: {len(item['ECG_R_Onsets'])}")
                print(f"    ECG_R_Onsets first 5: {item['ECG_R_Onsets'][:5]}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# The correct way to unpack
if len(result) == 2:
    signals, waves = result
    print("✅ ecg_delineate returns (signals_dataframe, waves_dict)")
    print(f"  signals type: {type(signals)}")
    print(f"  waves type: {type(waves)}")
    
    if isinstance(waves, dict) and 'ECG_R_Onsets' in waves:
        print(f"\n  waves['ECG_R_Onsets'] = {waves['ECG_R_Onsets'][:5]}")
        print(f"  These are INDICES (sample positions), not a DataFrame!")
