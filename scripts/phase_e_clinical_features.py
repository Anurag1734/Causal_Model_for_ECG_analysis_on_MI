"""
Phase E: Clinical Features Extraction (Leakage-Vetted)

CRITICAL RULES:
1. ☠ NEVER USE TROPONIN AS A FEATURE ☠
2. ALL features must be temporally BEFORE ecg_time
3. Use conservative time windows to prevent leakage

Features Extracted:
- E.2: Laboratory values (lipids, renal, metabolic, cardiac non-troponin)
- E.3: Vital signs (BP, RR, SpO2, temp)
- E.4: Medications (statin use)
- E.5: Comorbidities (diabetes, hypertension, CKD, chronic MI)
- E.6: Demographics (age, sex)

Output: clinical_features.parquet
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm

# Paths
DB_PATH = "data/mimic_database.duckdb"
METADATA_PATH = "data/processed/ecg_features_with_demographics.parquet"
OUTPUT_PATH = "data/processed/clinical_features.parquet"

# Connect to database
print("Connecting to MIMIC database...")
conn = duckdb.connect(DB_PATH, read_only=True)

# Load ECG metadata (already has record_id, ecg_time, subject_id, hadm_id)
print(f"Loading ECG metadata from {METADATA_PATH}...")
df_ecg = pd.read_parquet(METADATA_PATH)
print(f"Loaded {len(df_ecg)} ECG records")

# Get admittime from database
print("Querying admission times from database...")
query = """
SELECT DISTINCT
    e.hadm_id,
    a.admittime
FROM (SELECT DISTINCT hadm_id FROM df_ecg WHERE hadm_id IS NOT NULL) e
INNER JOIN admissions a ON e.hadm_id = a.hadm_id
"""
df_admittime = conn.execute(query).df()
df_ecg = df_ecg.merge(df_admittime, on='hadm_id', how='left')
print(f"Matched {df_ecg['admittime'].notna().sum()} records with admission times")

# Initialize results DataFrame
results = df_ecg[['record_id', 'subject_id', 'hadm_id', 'ecg_time', 'admittime']].copy()

print("\n" + "=" * 80)
print("Phase E.2: Laboratory Values")
print("=" * 80)

# =============================================================================
# E.2.1: Long-Term Risk Factors (Last Value Within 12 Months Prior to Admission)
# =============================================================================
print("\nQuerying lipid panel (12-month lookback from admission)...")

lipid_itemids = {
    'ldl': [50813],  # LDL cholesterol
    'hdl': [50904],  # HDL cholesterol (corrected itemid)
    'total_chol': [50810, 50811],  # Total cholesterol
    'triglycerides': [50817]  # Triglycerides
}

for lab_name, itemids in lipid_itemids.items():
    print(f"  - {lab_name}...")
    
    itemid_list = ",".join(map(str, itemids))
    
    query = f"""
    WITH ecg_admissions AS (
        SELECT DISTINCT
            e.record_id,
            e.subject_id,
            e.hadm_id,
            e.admittime,
            e.ecg_time
        FROM (
            SELECT 
                record_id,
                subject_id,
                hadm_id,
                admittime,
                ecg_time
            FROM df_ecg
        ) e
    ),
    lipid_values AS (
        SELECT
            ea.record_id,
            le.valuenum,
            le.charttime,
            ROW_NUMBER() OVER (
                PARTITION BY ea.record_id 
                ORDER BY le.charttime DESC
            ) as rn
        FROM ecg_admissions ea
        INNER JOIN labevents le
            ON ea.subject_id = le.subject_id
        WHERE le.itemid IN ({itemid_list})
            AND le.valuenum IS NOT NULL
            AND le.charttime >= (ea.admittime - INTERVAL '12 months')
            AND le.charttime <= ea.admittime
    )
    SELECT
        record_id,
        valuenum as {lab_name}
    FROM lipid_values
    WHERE rn = 1
    """
    
    df_lab = conn.execute(query).df()
    
    # Merge with results
    results = results.merge(df_lab, on='record_id', how='left')
    
    # Create missing flag
    results[f'{lab_name}_missing'] = results[lab_name].isna().astype(int)
    
    n_available = results[lab_name].notna().sum()
    pct_available = 100 * n_available / len(results)
    print(f"    Available: {n_available}/{len(results)} ({pct_available:.1f}%)")

# =============================================================================
# E.2.2: Acute State (Nearest Value Before ECG, Within 24 Hours)
# =============================================================================
print("\nQuerying acute labs (24-hour lookback from ECG time)...")

acute_lab_itemids = {
    'creatinine': [50912],  # Creatinine (mg/dL)
    'bun': [51006],  # Blood Urea Nitrogen
    'glucose': [50809, 50931],  # Glucose (mg/dL)
    'potassium': [50822, 50971],  # Potassium (mEq/L)
    'bnp': [50963],  # BNP (Brain Natriuretic Peptide)
    'nt_probnp': [50956]  # NT-proBNP
}

for lab_name, itemids in acute_lab_itemids.items():
    print(f"  - {lab_name}...")
    
    itemid_list = ",".join(map(str, itemids))
    
    query = f"""
    WITH ecg_records AS (
        SELECT 
            record_id,
            subject_id,
            ecg_time
        FROM df_ecg
    ),
    acute_labs AS (
        SELECT
            er.record_id,
            le.valuenum,
            le.charttime,
            ROW_NUMBER() OVER (
                PARTITION BY er.record_id 
                ORDER BY le.charttime DESC
            ) as rn
        FROM ecg_records er
        INNER JOIN labevents le
            ON er.subject_id = le.subject_id
        WHERE le.itemid IN ({itemid_list})
            AND le.valuenum IS NOT NULL
            AND le.charttime >= (er.ecg_time - INTERVAL '24 hours')
            AND le.charttime < er.ecg_time
    )
    SELECT
        record_id,
        valuenum as {lab_name}
    FROM acute_labs
    WHERE rn = 1
    """
    
    df_lab = conn.execute(query).df()
    
    # Merge with results
    results = results.merge(df_lab, on='record_id', how='left')
    
    # Create missing flag
    results[f'{lab_name}_missing'] = results[lab_name].isna().astype(int)
    
    n_available = results[lab_name].notna().sum()
    pct_available = 100 * n_available / len(results)
    print(f"    Available: {n_available}/{len(results)} ({pct_available:.1f}%)")

print("\n" + "=" * 80)
print("Phase E.3: Vital Signs")
print("=" * 80)

# =============================================================================
# E.3: Query Vital Signs (-60 minutes to 0 minutes relative to ecg_time)
# =============================================================================
print("\nQuerying vital signs (60-minute lookback from ECG time)...")

vital_itemids = {
    'sbp': [220050, 220179],  # Systolic BP (Non-Invasive & Invasive)
    'dbp': [220051, 220180],  # Diastolic BP
    'rr': [220210, 224690],  # Respiratory Rate
    'spo2': [220277],  # Oxygen Saturation (SpO2)
    'temperature': [223761, 223762]  # Temperature (C)
}

for vital_name, itemids in vital_itemids.items():
    print(f"  - {vital_name}...")
    
    itemid_list = ",".join(map(str, itemids))
    
    query = f"""
    WITH ecg_records AS (
        SELECT 
            record_id,
            subject_id,
            ecg_time
        FROM df_ecg
    ),
    vitals AS (
        SELECT
            er.record_id,
            ce.valuenum,
            ce.charttime,
            ROW_NUMBER() OVER (
                PARTITION BY er.record_id 
                ORDER BY ce.charttime DESC
            ) as rn
        FROM ecg_records er
        INNER JOIN chartevents ce
            ON er.subject_id = ce.subject_id
        WHERE ce.itemid IN ({itemid_list})
            AND ce.valuenum IS NOT NULL
            AND ce.charttime >= (er.ecg_time - INTERVAL '60 minutes')
            AND ce.charttime < er.ecg_time
    )
    SELECT
        record_id,
        valuenum as {vital_name}
    FROM vitals
    WHERE rn = 1
    """
    
    df_vital = conn.execute(query).df()
    
    # Merge with results
    results = results.merge(df_vital, on='record_id', how='left')
    
    # Convert temperature from Fahrenheit to Celsius if needed
    if vital_name == 'temperature' and results['temperature'].notna().any():
        temp_mean = results['temperature'].mean()
        if temp_mean > 45:  # Likely Fahrenheit
            results['temperature'] = (results['temperature'] - 32) * 5/9
            print(f"    (Converted from Fahrenheit to Celsius)")
    
    # Create missing flag
    results[f'{vital_name}_missing'] = results[vital_name].isna().astype(int)
    
    n_available = results[vital_name].notna().sum()
    pct_available = 100 * n_available / len(results)
    print(f"    Available: {n_available}/{len(results)} ({pct_available:.1f}%)")

print("\n" + "=" * 80)
print("Phase E.4: Medications (Statin Use)")
print("=" * 80)

# =============================================================================
# E.4: Query Statin Use (Binary Feature)
# =============================================================================
print("\nQuerying statin prescriptions...")

# Statin medication names
statin_names = [
    'Atorvastatin', 'Simvastatin', 'Rosuvastatin', 
    'Pravastatin', 'Lovastatin', 'Fluvastatin', 'Pitavastatin'
]

statin_pattern = '|'.join(statin_names)

query = f"""
WITH ecg_admissions AS (
    SELECT DISTINCT
        record_id,
        subject_id,
        hadm_id,
        admittime
    FROM df_ecg
),
statin_prescriptions AS (
    SELECT DISTINCT
        ea.record_id,
        1 as statin_use
    FROM ecg_admissions ea
    INNER JOIN prescriptions p
        ON ea.subject_id = p.subject_id
    WHERE p.starttime < ea.admittime
        AND (
            p.drug ILIKE '%Atorvastatin%' OR
            p.drug ILIKE '%Simvastatin%' OR
            p.drug ILIKE '%Rosuvastatin%' OR
            p.drug ILIKE '%Pravastatin%' OR
            p.drug ILIKE '%Lovastatin%' OR
            p.drug ILIKE '%Fluvastatin%' OR
            p.drug ILIKE '%Pitavastatin%'
        )
)
SELECT
    record_id,
    statin_use
FROM statin_prescriptions
"""

df_statin = conn.execute(query).df()

# Merge with results
results = results.merge(df_statin, on='record_id', how='left')
results['statin_use'] = results['statin_use'].fillna(0).astype(int)

n_statin_users = results['statin_use'].sum()
pct_statin = 100 * n_statin_users / len(results)
print(f"  Statin users: {n_statin_users}/{len(results)} ({pct_statin:.1f}%)")

print("\n" + "=" * 80)
print("Phase E.5: Comorbidities")
print("=" * 80)

# =============================================================================
# E.5: Query Comorbidities (From Prior Admissions Only)
# =============================================================================
print("\nQuerying comorbidities (from prior admissions)...")

# Define ICD code patterns for comorbidities
comorbidities = {
    'diabetes': {
        'icd9': ['250%'],
        'icd10': ['E08%', 'E09%', 'E10%', 'E11%', 'E12%', 'E13%']
    },
    'hypertension': {
        'icd9': ['401%', '402%', '403%', '404%', '405%'],
        'icd10': ['I10%', 'I11%', 'I12%', 'I13%', 'I15%']
    },
    'ckd': {
        'icd9': ['585%'],
        'icd10': ['N18%']
    }
}

for condition, codes in comorbidities.items():
    print(f"  - {condition}...")
    
    # Build ICD code conditions
    icd9_conditions = " OR ".join([f"d.icd_code LIKE '{code}'" for code in codes['icd9']])
    icd10_conditions = " OR ".join([f"d.icd_code LIKE '{code}'" for code in codes['icd10']])
    
    query = f"""
    WITH ecg_admissions AS (
        SELECT DISTINCT
            record_id,
            subject_id,
            hadm_id,
            admittime
        FROM df_ecg
    ),
    prior_diagnoses AS (
        SELECT DISTINCT
            ea.record_id,
            1 as {condition}
        FROM ecg_admissions ea
        INNER JOIN admissions a_prior
            ON ea.subject_id = a_prior.subject_id
            AND a_prior.admittime < ea.admittime  -- Prior admissions only
        INNER JOIN diagnoses_icd d
            ON a_prior.hadm_id = d.hadm_id
        WHERE (
            (d.icd_version = 9 AND ({icd9_conditions})) OR
            (d.icd_version = 10 AND ({icd10_conditions}))
        )
    )
    SELECT
        record_id,
        {condition}
    FROM prior_diagnoses
    """
    
    df_condition = conn.execute(query).df()
    
    # Merge with results
    results = results.merge(df_condition, on='record_id', how='left')
    results[condition] = results[condition].fillna(0).astype(int)
    
    n_positive = results[condition].sum()
    pct_positive = 100 * n_positive / len(results)
    print(f"    Positive: {n_positive}/{len(results)} ({pct_positive:.1f}%)")

# Add comorbidity_chronic_mi (already computed in Phase C.6)
print("  - comorbidity_chronic_mi...")
if 'Comorbidity_Chronic_MI' in df_ecg.columns:
    results['comorbidity_chronic_mi'] = df_ecg['Comorbidity_Chronic_MI'].astype(int)
else:
    # Recompute if not available
    print("    (Not found in metadata, recomputing...)")
    
    query = """
    WITH ecg_admissions AS (
        SELECT DISTINCT
            record_id,
            subject_id,
            hadm_id,
            admittime
        FROM df_ecg
    ),
    chronic_mi AS (
        SELECT DISTINCT
            ea.record_id,
            1 as comorbidity_chronic_mi
        FROM ecg_admissions ea
        INNER JOIN admissions a_prior
            ON ea.subject_id = a_prior.subject_id
            AND a_prior.admittime < ea.admittime
        INNER JOIN diagnoses_icd d
            ON a_prior.hadm_id = d.hadm_id
        WHERE (
            (d.icd_version = 9 AND d.icd_code LIKE '412%') OR
            (d.icd_version = 10 AND d.icd_code LIKE 'I25.2%')
        )
    )
    SELECT
        record_id,
        comorbidity_chronic_mi
    FROM chronic_mi
    """
    
    df_chronic_mi = conn.execute(query).df()
    results = results.merge(df_chronic_mi, on='record_id', how='left')
    results['comorbidity_chronic_mi'] = results['comorbidity_chronic_mi'].fillna(0).astype(int)

n_chronic_mi = results['comorbidity_chronic_mi'].sum()
pct_chronic_mi = 100 * n_chronic_mi / len(results)
print(f"    Positive: {n_chronic_mi}/{len(results)} ({pct_chronic_mi:.1f}%)")

print("\n" + "=" * 80)
print("Phase E.6: Demographics")
print("=" * 80)

# =============================================================================
# E.6: Demographics (Already in metadata from Phase C)
# =============================================================================
print("\nAdding demographics from metadata...")

# Age and gender should already be in df_ecg
if 'age' in df_ecg.columns:
    results['age'] = df_ecg['age']
    print(f"  age: mean={results['age'].mean():.1f}, std={results['age'].std():.1f}")
else:
    print("  WARNING: age not found in metadata!")

if 'sex' in df_ecg.columns:
    results['sex'] = df_ecg['sex']
    n_male = (results['sex'] == 'M').sum()
    n_female = (results['sex'] == 'F').sum()
    print(f"  sex: Male={n_male}, Female={n_female}")
else:
    print("  WARNING: sex not found in metadata!")

print("\n" + "=" * 80)
print("Phase E.7: Save Clinical Features")
print("=" * 80)

# =============================================================================
# E.7: Save Clinical Features
# =============================================================================

# Select final columns
clinical_columns = [
    'record_id',
    # Labs (long-term)
    'ldl', 'ldl_missing',
    'hdl', 'hdl_missing',
    'total_chol', 'total_chol_missing',
    'triglycerides', 'triglycerides_missing',
    # Labs (acute)
    'creatinine', 'creatinine_missing',
    'bun', 'bun_missing',
    'glucose', 'glucose_missing',
    'potassium', 'potassium_missing',
    'bnp', 'bnp_missing',
    'nt_probnp', 'nt_probnp_missing',
    # Vitals
    'sbp', 'sbp_missing',
    'dbp', 'dbp_missing',
    'rr', 'rr_missing',
    'spo2', 'spo2_missing',
    'temperature', 'temperature_missing',
    # Medications
    'statin_use',
    # Comorbidities
    'diabetes',
    'hypertension',
    'ckd',
    'comorbidity_chronic_mi',
    # Demographics
    'age',
    'sex'
]

# Check which columns are available
available_columns = [col for col in clinical_columns if col in results.columns]
missing_columns = [col for col in clinical_columns if col not in results.columns]

if missing_columns:
    print(f"\nWARNING: Missing columns: {missing_columns}")

df_clinical = results[available_columns].copy()

# Save
print(f"\nSaving clinical features to {OUTPUT_PATH}...")
df_clinical.to_parquet(OUTPUT_PATH, index=False)

print(f"\n✓ Saved {len(df_clinical)} records with {len(df_clinical.columns)} features")

# Summary statistics
print("\n" + "=" * 80)
print("Clinical Features Summary")
print("=" * 80)

print("\nLaboratory Values (Long-Term):")
for lab in ['ldl', 'hdl', 'total_chol', 'triglycerides']:
    if lab in df_clinical.columns:
        available = df_clinical[lab].notna().sum()
        pct = 100 * available / len(df_clinical)
        mean_val = df_clinical[lab].mean()
        print(f"  {lab:20s}: {available:6d}/{len(df_clinical)} ({pct:5.1f}%) | mean={mean_val:7.2f}")

print("\nLaboratory Values (Acute):")
for lab in ['creatinine', 'bun', 'glucose', 'potassium', 'bnp', 'nt_probnp']:
    if lab in df_clinical.columns:
        available = df_clinical[lab].notna().sum()
        pct = 100 * available / len(df_clinical)
        mean_val = df_clinical[lab].mean()
        print(f"  {lab:20s}: {available:6d}/{len(df_clinical)} ({pct:5.1f}%) | mean={mean_val:7.2f}")

print("\nVital Signs:")
for vital in ['sbp', 'dbp', 'rr', 'spo2', 'temperature']:
    if vital in df_clinical.columns:
        available = df_clinical[vital].notna().sum()
        pct = 100 * available / len(df_clinical)
        mean_val = df_clinical[vital].mean()
        print(f"  {vital:20s}: {available:6d}/{len(df_clinical)} ({pct:5.1f}%) | mean={mean_val:7.2f}")

print("\nMedications:")
if 'statin_use' in df_clinical.columns:
    n_users = df_clinical['statin_use'].sum()
    pct = 100 * n_users / len(df_clinical)
    print(f"  statin_use:          {n_users:6d}/{len(df_clinical)} ({pct:5.1f}%)")

print("\nComorbidities:")
for cond in ['diabetes', 'hypertension', 'ckd', 'comorbidity_chronic_mi']:
    if cond in df_clinical.columns:
        n_positive = df_clinical[cond].sum()
        pct = 100 * n_positive / len(df_clinical)
        print(f"  {cond:20s}: {n_positive:6d}/{len(df_clinical)} ({pct:5.1f}%)")

print("\nDemographics:")
if 'age' in df_clinical.columns:
    print(f"  age:    mean={df_clinical['age'].mean():.1f}, std={df_clinical['age'].std():.1f}")
if 'sex' in df_clinical.columns:
    n_male = (df_clinical['sex'] == 'M').sum()
    n_female = (df_clinical['sex'] == 'F').sum()
    print(f"  sex:    Male={n_male} ({100*n_male/len(df_clinical):.1f}%), Female={n_female} ({100*n_female/len(df_clinical):.1f}%)")

# Close database connection
conn.close()

print("\n" + "=" * 80)
print("✓ Phase E: Clinical Features - COMPLETE")
print("=" * 80)
print(f"\nOutput: {OUTPUT_PATH}")
print(f"Records: {len(df_clinical)}")
print(f"Features: {len(df_clinical.columns)}")
