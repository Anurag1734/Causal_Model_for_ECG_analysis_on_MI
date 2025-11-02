"""
Adjudication Analysis Script
Calculates agreement metrics between automated labels and clinician assessments.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

# Load adjudication results
adjudication = pd.read_csv('data/processed/adjudication_sample_REVIEWED.csv')

print("=" * 80)
print("ADJUDICATION RESULTS ANALYSIS")
print("=" * 80)

# Clean up labels
adjudication['automated_label_clean'] = adjudication['automated_label'].replace({
    'MI_Acute_Presentation': 'MI_Acute',
    'Control_Symptomatic': 'No_MI'
})

# Remove uncertain cases for primary analysis
primary_analysis = adjudication[adjudication['clinician_label'] != 'Uncertain'].copy()

print(f"\nTotal cases reviewed: {len(adjudication)}")
print(f"Uncertain cases: {(adjudication['clinician_label'] == 'Uncertain').sum()}")
print(f"Cases in primary analysis: {len(primary_analysis)}")

# Calculate agreement
agreement = (primary_analysis['automated_label_clean'] == primary_analysis['clinician_label']).sum()
total = len(primary_analysis)
agreement_rate = 100 * agreement / total

print("\n" + "-" * 80)
print("OVERALL AGREEMENT")
print("-" * 80)
print(f"Agreement: {agreement}/{total} ({agreement_rate:.1f}%)")
print(f"Disagreement: {total - agreement}/{total} ({100 - agreement_rate:.1f}%)")

# Cohen's Kappa
kappa = cohen_kappa_score(primary_analysis['automated_label_clean'], 
                          primary_analysis['clinician_label'])
print(f"Cohen's Kappa: {kappa:.3f}")

if kappa >= 0.80:
    kappa_interp = "Excellent"
elif kappa >= 0.60:
    kappa_interp = "Good"
elif kappa >= 0.40:
    kappa_interp = "Moderate"
else:
    kappa_interp = "Poor"

print(f"Interpretation: {kappa_interp}")

# Confusion Matrix
print("\n" + "-" * 80)
print("CONFUSION MATRIX")
print("-" * 80)
cm = confusion_matrix(primary_analysis['clinician_label'], 
                      primary_analysis['automated_label_clean'],
                      labels=['MI_Acute', 'No_MI'])

print("                Automated")
print("              MI_Acute  No_MI")
print(f"Clinician MI_Acute  {cm[0,0]:5d}   {cm[0,1]:5d}")
print(f"          No_MI     {cm[1,0]:5d}   {cm[1,1]:5d}")

# Classification Report
print("\n" + "-" * 80)
print("CLASSIFICATION METRICS")
print("-" * 80)
print(classification_report(primary_analysis['clinician_label'],
                          primary_analysis['automated_label_clean'],
                          labels=['MI_Acute', 'No_MI'],
                          target_names=['MI_Acute', 'No_MI']))

# Positive Predictive Value (Precision) and Negative Predictive Value
tn, fp, fn, tp = cm.ravel()
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print("-" * 80)
print("CLINICAL METRICS")
print("-" * 80)
print(f"Sensitivity (True Positive Rate): {sensitivity:.1%}")
print(f"Specificity (True Negative Rate): {specificity:.1%}")
print(f"Positive Predictive Value (PPV):  {ppv:.1%}")
print(f"Negative Predictive Value (NPV):  {npv:.1%}")

# Disagreement Analysis
print("\n" + "=" * 80)
print("DISAGREEMENT CASES")
print("=" * 80)

disagreements = primary_analysis[
    primary_analysis['automated_label_clean'] != primary_analysis['clinician_label']
]

if len(disagreements) > 0:
    print(f"\nFound {len(disagreements)} disagreement cases:")
    print("\nFalse Positives (Algorithm said MI, Clinician said No):")
    fp_cases = disagreements[
        (disagreements['automated_label_clean'] == 'MI_Acute') &
        (disagreements['clinician_label'] == 'No_MI')
    ]
    print(f"  Count: {len(fp_cases)}")
    if len(fp_cases) > 0:
        print(fp_cases[['record_id', 'max_troponin', 'admission_type', 
                       'hours_from_mi', 'notes']].head(10).to_string(index=False))
    
    print("\nFalse Negatives (Algorithm said No MI, Clinician said MI):")
    fn_cases = disagreements[
        (disagreements['automated_label_clean'] == 'No_MI') &
        (disagreements['clinician_label'] == 'MI_Acute')
    ]
    print(f"  Count: {len(fn_cases)}")
    if len(fn_cases) > 0:
        print(fn_cases[['record_id', 'max_troponin', 'admission_type',
                       'hours_from_mi', 'notes']].head(10).to_string(index=False))

# Decision
print("\n" + "=" * 80)
print("VALIDATION DECISION")
print("=" * 80)

if agreement_rate >= 80 and kappa >= 0.60:
    decision = "✅ ALGORITHM VALIDATED"
    print(decision)
    print("Recommendation: Proceed to Phase D (Feature Extraction)")
elif agreement_rate >= 70:
    decision = "⚠️  BORDERLINE VALIDATION"
    print(decision)
    print("Recommendation: Review disagreement cases, consider threshold adjustment")
else:
    decision = "❌ ALGORITHM NEEDS REVISION"
    print(decision)
    print("Recommendation: Revise MI definition logic, re-adjudicate")

print("=" * 80)

# Save results
results_summary = pd.DataFrame([{
    'total_cases': len(adjudication),
    'uncertain_cases': (adjudication['clinician_label'] == 'Uncertain').sum(),
    'agreement_rate': agreement_rate,
    'cohens_kappa': kappa,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'ppv': ppv,
    'npv': npv,
    'false_positives': len(fp_cases),
    'false_negatives': len(fn_cases),
    'decision': decision
}])

results_summary.to_csv('data/processed/adjudication_results_summary.csv', index=False)
print("\n✓ Results saved to: data/processed/adjudication_results_summary.csv")

# Save disagreement cases for detailed review
if len(disagreements) > 0:
    disagreements.to_csv('data/processed/adjudication_disagreements.csv', index=False)
    print("✓ Disagreements saved to: data/processed/adjudication_disagreements.csv")
