# MIMIC-IV MI Cohort Feature Extraction Report

**Date**: 2025-10-26 13:04:46

**Total Records**: 125,882

## Extraction Summary

- **Successful**: 125,882/125,882 (100.0%)
- **Failed**: 0/125,882 (0.0%)

## Feature Statistics (n=125,882)

| Feature | Mean | Std | Median | Min | Max |
|---------|------|-----|--------|-----|-----|
| heart_rate | 82.9 | 22.6 | 79.2 | 6.3 | 198.0 |
| pr_interval_ms | 125.9 | 187.9 | 100.0 | 8.0 | 5618.0 |
| qrs_duration_ms | 247.1 | 394.7 | 170.0 | 50.0 | 7708.0 |
| qt_interval_ms | 444.5 | 374.5 | 394.0 | 18.0 | 8262.0 |
| qtc_bazett | 520.1 | 494.3 | 454.4 | 23.7 | 11091.9 |
| qtc_fridericia | 492.5 | 447.9 | 434.5 | 21.6 | 9816.3 |
| rr_variability_ms | 76.5 | 146.4 | 24.1 | 0.0 | 4157.0 |

## Quality Control

- **High Quality ECGs**: 84,514/125,882 (67.1%)

### Plausibility Checks:

- **Hr Plausible**: 123,861/125,882 (98.4%)
- **Qrs Plausible**: 88,193/125,882 (70.1%)
- **Qt Plausible**: 109,737/125,882 (87.2%)
- **Qtc Plausible**: 97,722/125,882 (77.6%)

## Stratification by Label

| Label | Count | Mean HR | Std HR |
|-------|-------|---------|--------|
| Control_Symptomatic | 108,270 | 82.7 | 22.7 |
| MI_Acute_Presentation | 8,122 | 83.1 | 21.1 |
| MI_Pre-Incident | 8,099 | 84.2 | 22.5 |

## Output Files

- Feature matrix: `data/processed/ecg_features.parquet`
- This report: `reports/feature_extraction_report.md`
