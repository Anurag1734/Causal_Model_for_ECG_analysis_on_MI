# Latent Space Interpretability Report

**Model:** models/checkpoints/vae_zdim64_beta4.0_20251102_073456/best_model.pt

**z_dim:** 64

**beta:** 4.0

## Summary

- **Interpretable dimensions:** 60/64
- **Overall plausibility:** 99.78%

## Criteria

1. ≥10 interpretable dimensions: ✓ PASS (60/10)
2. ≥95% plausible ECGs: ✓ PASS (99.8%/95%)

## Decision: **PROCEED ✓**

**Recommendation:** VAE has learned meaningful, disentangled features. Proceed to Phase E-F (Master Dataset).

## Top 10 Most Interpretable Dimensions

| Dimension | Avg Change | Max Change | Monotonic | Plausibility |
|-----------|------------|------------|-----------|-------------|
| z_ecg_11 | 0.325301 | 0.533479 | Yes | 100.0% |
| z_ecg_44 | 0.319849 | 0.560383 | Yes | 100.0% |
| z_ecg_62 | 0.316706 | 0.536071 | Yes | 100.0% |
| z_ecg_32 | 0.311908 | 0.503145 | Yes | 100.0% |
| z_ecg_53 | 0.311246 | 0.495103 | Yes | 100.0% |
| z_ecg_39 | 0.310153 | 0.466289 | Yes | 100.0% |
| z_ecg_27 | 0.310021 | 0.469685 | Yes | 100.0% |
| z_ecg_24 | 0.309024 | 0.507514 | Yes | 100.0% |
| z_ecg_30 | 0.308936 | 0.479282 | Yes | 100.0% |
| z_ecg_45 | 0.308812 | 0.469445 | Yes | 100.0% |

## Plausibility Statistics

- Mean: 99.78%
- Median: 100.00%
- Min: 85.71%
- Max: 100.00%

Dimensions with 100% plausibility: 63
Dimensions with <50% plausibility: 0

## Next Steps

1. **Manual Annotation**: Review all 64 dimension plots and annotate interpretable ones
2. **Fill latent_dimension_descriptions.csv**: Add physiological descriptions
3. **Proceed to Phase E-F**: Merge latent features with clinical data

## Dimension Descriptions (Manual Annotation Required)

*After reviewing dimension traversal plots, fill in descriptions for interpretable dimensions.*

**Examples of Good Annotations:**
- z_ecg_1: Heart rate (slow 50 bpm → fast 120 bpm)
- z_ecg_5: ST-segment elevation in leads V2-V4 (0mm → +3mm)
- z_ecg_12: QRS duration (narrow 80ms → wide 180ms, suggests LBBB pattern)
- z_ecg_23: T-wave inversion in inferior leads (II, III, aVF)

**Examples of Bad (Entangled) Dimensions:**
- z_ecg_17: Changes HR + ST-segment + T-wave simultaneously (entangled)
- z_ecg_42: Produces noisy, implausible signals (not interpretable)
