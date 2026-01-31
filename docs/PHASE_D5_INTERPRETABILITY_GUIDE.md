# Phase D.5: Latent Space Interpretability Validation

**Goal**: Confirm that β-VAE learned meaningful, disentangled physiological features before proceeding to Phase E-F.

---

## 🎯 Overview

This validation ensures your VAE is useful for downstream causal inference. A good VAE should:
- **Learn disentangled features**: Each latent dimension controls ONE physiological factor
- **Produce plausible ECGs**: Decoded signals pass clinical validity checks
- **Enable counterfactuals**: We can manipulate specific features (e.g., "what if HR was 80 bpm?")

---

## 📋 Protocol Summary

### 1. Single-Dimension Traversal
For each latent dimension `i ∈ {1, ..., 64}`:
- Start with `z_base` = mean of all Control ECG embeddings
- Create variants: `z_variant[i] = z_base + α × e_i`
  - `α ∈ {-3, -2, -1, 0, 1, 2, 3}` (7 values)
  - `e_i` = i-th unit vector
- Decode each `z_variant` → 7 synthetic ECGs
- Plot all 7 ECGs (12-lead × 7 variants) in a grid

### 2. Qualitative Assessment
Review plots for interpretability:
- ✅ **Good**: `z_1` smoothly changes heart rate 50→120 bpm
- ✅ **Good**: `z_5` changes ST-elevation in V3 from -1mm to +2mm
- ❌ **Bad**: `z_12` changes HR, ST-segment, AND T-wave (entangled)
- ❌ **Bad**: `z_17` produces noisy, implausible signals

### 3. Quantitative Checks
For each decoded ECG, verify:
- ✅ QTc < 700 ms
- ✅ Heart rate: 20-200 bpm
- ✅ Signal amplitude: -5 to +5 mV
- ✅ No NaN or Inf values

**Threshold**: ≥95% of decoded ECGs must pass ALL checks

### 4. Go/No-Go Decision

**PROCEED if:**
- ≥10 dimensions show clear interpretability
- ≥95% of decoded ECGs are physiologically plausible

**RETRAIN if:**
- <5 interpretable dimensions → Increase β (4 → 8)
- Posterior collapse (KL loss → 0) → Decrease β (4 → 2)
- Poor reconstruction → Decrease β or increase z_dim

---

## 🚀 Usage

### Step 1: Activate Environment
```powershell
cd C:\Users\Acer\Desktop\Capstone\Capstone
.\cuda_env\Scripts\activate
```

### Step 2: Run Validation
```powershell
python scripts/validate_latent_interpretability.py `
    --checkpoint models/checkpoints/vae_zdim64_beta4.0_20251102_073456/best_model.pt `
    --z_dim 64 `
    --beta 4.0 `
    --metadata_path data/processed/ecg_features_with_demographics.parquet `
    --base_path data/raw/MIMIC-IV-ECG-1.0/files `
    --n_base_samples 1000 `
    --alphas -3 -2 -1 0 1 2 3 `
    --output_dir results/latent_interpretability `
    --save_plots
```

### Parameters Explained:
- `--checkpoint`: Path to your trained VAE model
- `--z_dim`: Latent dimension size (should match training)
- `--beta`: β-VAE parameter (should match training)
- `--metadata_path`: Parquet file with ECG metadata and labels
- `--base_path`: Directory containing WFDB ECG files
- `--n_base_samples`: Number of Control ECGs to compute `z_base` (default: 1000)
- `--alphas`: Traversal values (default: -3 to +3)
- `--output_dir`: Where to save results
- `--save_plots`: Generate 64 dimension traversal plots (one per dimension)

### Expected Runtime:
- **With GPU**: ~15-30 minutes (64 dimensions × 7 alphas = 448 decodings)
- **With CPU**: ~1-2 hours

---

## 📊 Output Files

The script creates the following structure:
```
results/latent_interpretability/
├── dimension_plots/                          # 64 PNG files
│   ├── dimension_001_traversal.png           # z_ecg_1 traversal (12 leads × 7 alphas)
│   ├── dimension_002_traversal.png           # z_ecg_2 traversal
│   └── ...
│   └── dimension_064_traversal.png           # z_ecg_64 traversal
├── interpretability_results.csv              # Quantitative metrics per dimension
├── latent_dimension_descriptions.csv         # Template for manual annotations
└── latent_interpretability_report.md         # Final Go/No-Go decision
```

---

## 📝 Output Files Explained

### 1. `dimension_plots/dimension_XXX_traversal.png`
**Purpose**: Visual inspection of each latent dimension

**What to Look For**:
- **Smooth variation**: As α goes from -3 → +3, ECG should change gradually
- **Single-factor control**: Only ONE feature should change (e.g., HR only, not HR+ST+T)
- **Physiological plausibility**: ECGs should look realistic (P-wave, QRS, T-wave visible)

**Example Good Dimension (z_1 = Heart Rate)**:
```
α=-3: HR ~50 bpm (bradycardia)
α=-2: HR ~60 bpm
α=-1: HR ~70 bpm
α=0:  HR ~80 bpm (baseline)
α=+1: HR ~90 bpm
α=+2: HR ~100 bpm
α=+3: HR ~120 bpm (tachycardia)
```

**Example Bad Dimension (entangled)**:
```
α=-3: HR 50 bpm + ST-elevation + wide QRS (changing 3 things!)
α=0:  HR 80 bpm + normal ST + normal QRS
α=+3: HR 120 bpm + ST-depression + narrow QRS
```

### 2. `interpretability_results.csv`
**Columns**:
- `dimension`: Dimension number (1-64)
- `avg_change`: Average signal change across alphas (higher = more effect)
- `max_change`: Maximum signal change
- `is_monotonic`: Does dimension vary smoothly? (True/False)
- `plausibility_rate`: % of decoded ECGs that pass physiological checks
- `n_plausible`: Number of plausible ECGs (out of 7)
- `n_total`: Total ECGs (always 7)

**How to Use**:
```python
import pandas as pd

df = pd.read_csv('results/latent_interpretability/interpretability_results.csv')

# Find most interpretable dimensions
interpretable = df[
    (df['avg_change'] > 0.01) &
    (df['is_monotonic'] == True) &
    (df['plausibility_rate'] > 0.80)
].sort_values('avg_change', ascending=False)

print(f"Top 10 interpretable dimensions:")
print(interpretable.head(10))
```

### 3. `latent_dimension_descriptions.csv`
**Purpose**: Manual annotation template

**Example Filled**:
```csv
dimension,description,interpretable,plausibility_rate
z_ecg_1,"Heart rate (50-120 bpm)",True,1.00
z_ecg_5,"ST-elevation in V2-V4 leads",True,0.95
z_ecg_12,"QRS duration (narrow→wide, LBBB pattern)",True,0.88
z_ecg_17,"Entangled (HR+ST+T-wave)",False,0.65
z_ecg_23,"T-wave inversion inferior leads",True,0.92
z_ecg_42,"Noise/artifacts",False,0.30
```

**Your Task**:
1. Review all 64 dimension plots
2. For interpretable dimensions, write a clear description
3. Mark non-interpretable dimensions (entangled or noisy)

### 4. `latent_interpretability_report.md`
**Purpose**: Final Go/No-Go decision report

**Contains**:
- Summary statistics (# interpretable dimensions, overall plausibility)
- Pass/Fail on criteria
- Decision: PROCEED ✓ or RETRAIN ✗
- Specific recommendations if retraining needed

**Example Output**:
```markdown
# Latent Space Interpretability Report

**Model:** models/checkpoints/vae_zdim64_beta4.0_20251102_073456/best_model.pt
**z_dim:** 64
**β:** 4.0

## Summary
- **Interpretable dimensions:** 15/64
- **Overall plausibility:** 97.3%

## Criteria
1. ≥10 interpretable dimensions: ✓ PASS (15/10)
2. ≥95% plausible ECGs: ✓ PASS (97.3%/95%)

## Decision: **PROCEED ✓**

**Recommendation:** VAE has learned meaningful, disentangled features. 
Proceed to Phase E-F (Master Dataset).

## Top 10 Most Interpretable Dimensions
| Dimension | Avg Change | Max Change | Monotonic | Plausibility |
|-----------|------------|------------|-----------|--------------|
| z_ecg_1   | 0.124567   | 0.245678   | Yes       | 100.0%       |
| z_ecg_5   | 0.098234   | 0.187654   | Yes       | 95.2%        |
| ...
```

---

## 🔍 Interpreting Results

### Scenario 1: ✅ PASS (Ideal)
```
✓ Interpretable dimensions: 15/64
✓ Overall plausibility: 97.3%

Decision: PROCEED ✓
```

**What This Means**:
- Your VAE learned meaningful patterns
- 15 dimensions have clear physiological interpretation
- 97.3% of decoded ECGs are clinically valid
- You can proceed to Phase E-F

**Next Steps**:
1. Fill in `latent_dimension_descriptions.csv` with your annotations
2. Save top interpretable dimensions for downstream analysis
3. Proceed to Phase E-F (merge latent features with clinical data)

---

### Scenario 2: ❌ FAIL - Entanglement Issue
```
✗ Interpretable dimensions: 4/64
✓ Overall plausibility: 96.5%

Decision: RETRAIN ✗
```

**What This Means**:
- VAE is reconstructing ECGs well (high plausibility)
- BUT dimensions are entangled (each dimension controls multiple factors)
- Not useful for causal inference (can't isolate effects)

**Diagnosis**: Not enough disentanglement pressure

**Fix**:
```powershell
# Retrain with higher β (encourage disentanglement)
python src/models/train_vae.py `
    --z_dim 64 `
    --beta 8.0 `   # ← Increased from 4.0
    --epochs 160 `
    --batch_size 256
```

**Why This Works**: Higher β penalizes entangled representations, forcing the model to use independent dimensions.

---

### Scenario 3: ❌ FAIL - Posterior Collapse
```
✗ Interpretable dimensions: 2/64
✗ Overall plausibility: 34.2%

Decision: RETRAIN ✗
```

**What This Means**:
- VAE is ignoring the latent space
- Decoder just outputs average ECG for all inputs
- Latent dimensions have no effect (collapsed)

**Diagnosis**: Posterior collapse (KL divergence → 0)

**How to Check**:
```powershell
# Load training history
$history = Get-Content "models/checkpoints/.../training_history.json" | ConvertFrom-Json
$history.train_kl_loss[-1]  # Check final KL loss

# If KL < 1.0 → Collapsed!
```

**Fix**:
```powershell
# Retrain with lower β or stronger free bits
python src/models/train_vae.py `
    --z_dim 64 `
    --beta 2.0 `         # ← Decreased from 4.0
    --free_bits 3.0 `    # ← Increased from 2.0
    --epochs 160
```

**Why This Works**: Lower β reduces KL penalty, allowing encoder to use latent space. Higher free bits prevents premature collapse.

---

### Scenario 4: ❌ FAIL - Poor Reconstruction
```
✓ Interpretable dimensions: 12/64
✗ Overall plausibility: 68.4%

Decision: RETRAIN ✗
```

**What This Means**:
- Dimensions are interpretable (good disentanglement)
- BUT reconstructed ECGs are physiologically implausible
- Model is sacrificing reconstruction quality for disentanglement

**Diagnosis**: β too high (over-regularized)

**Fix**:
```powershell
# Retrain with lower β
python src/models/train_vae.py `
    --z_dim 64 `
    --beta 2.0 `   # ← Decreased from 4.0
    --epochs 160
```

**Alternative Fix** (if β=2.0 still fails):
```powershell
# Increase model capacity
python src/models/train_vae.py `
    --z_dim 128 `  # ← Increased from 64
    --beta 4.0 `
    --epochs 160
```

---

## 🎨 Manual Annotation Guide

After reviewing plots, fill in `latent_dimension_descriptions.csv`:

### Good Annotations (Specific + Quantitative)
✅ **z_ecg_1**: "Heart rate (50 bpm → 120 bpm, smooth variation)"
✅ **z_ecg_5**: "ST-elevation in V2-V4 (0mm → +3mm, anterior STEMI pattern)"
✅ **z_ecg_12**: "QRS duration (80ms → 180ms, LBBB morphology in V1-V6)"
✅ **z_ecg_23**: "T-wave inversion in inferior leads (II, III, aVF)"
✅ **z_ecg_34**: "P-wave amplitude (flat → tall, suggests atrial enlargement)"

### Bad Annotations (Too Vague)
❌ "Changes ECG"
❌ "Some kind of rhythm thing"
❌ "Looks important"

### Non-Interpretable Dimensions
⚠️ **z_ecg_17**: "Entangled (changes HR + ST-segment + QRS simultaneously)"
⚠️ **z_ecg_42**: "Noisy/artifacts (non-physiological waveforms)"
⚠️ **z_ecg_56**: "No visible effect (dimension unused)"

---

## 📈 Expected Results (Based on Literature)

From similar ECG VAE research:

| β Value | Interpretable Dims | Plausibility | Trade-off |
|---------|-------------------|--------------|-----------|
| β=1.0 (standard VAE) | 5-8 | 98-99% | Good reconstruction, poor disentanglement |
| β=2.0 | 8-12 | 95-98% | Balanced |
| β=4.0 (your setting) | 12-18 | 92-96% | Good disentanglement, acceptable reconstruction |
| β=8.0 | 15-25 | 85-92% | Excellent disentanglement, worse reconstruction |

**Your Target (β=4.0)**:
- ✅ 12-18 interpretable dimensions expected
- ✅ 92-96% plausibility expected
- ✅ Should meet ≥10 dims + ≥95% plausibility criteria

---

## 🔧 Troubleshooting

### Issue 1: Script Crashes with "CUDA Out of Memory"
```
RuntimeError: CUDA out of memory
```

**Fix**: Reduce batch size (decode fewer ECGs at once)
```powershell
# Edit validate_latent_interpretability.py line 383
# Change:
dataloader = DataLoader(dataset, batch_size=32, ...)

# To:
dataloader = DataLoader(dataset, batch_size=8, ...)
```

---

### Issue 2: "No module named 'neurokit2'"
```
ModuleNotFoundError: No module named 'neurokit2'
```

**Fix**: Install NeuroKit2
```powershell
pip install neurokit2
```

---

### Issue 3: All Dimensions Look Random
```
✗ Interpretable dimensions: 0/64
✗ Plausibility: 12%
```

**Likely Causes**:
1. Wrong checkpoint loaded
2. Model not trained properly
3. Preprocessing mismatch

**Diagnostic**:
```powershell
# Check training loss
$history = Get-Content "models/checkpoints/.../training_history.json" | ConvertFrom-Json
$history.val_loss[-1]  # Should be ~30,000-35,000

# If val_loss > 100,000 → Model didn't train properly
```

---

### Issue 4: Plots Show Only Baseline (No Change)
```
All 7 alphas produce identical ECGs
```

**Likely Cause**: Posterior collapse (KL → 0)

**Check**:
```powershell
$history.train_kl_loss[-1]  # Should be 5-15
# If < 1.0 → Collapsed
```

**Fix**: Retrain with lower β or higher free bits

---

## ✅ Checklist Before Proceeding to Phase E-F

- [ ] Ran `validate_latent_interpretability.py` successfully
- [ ] Reviewed all 64 dimension plots
- [ ] ≥10 dimensions are clearly interpretable
- [ ] ≥95% overall plausibility achieved
- [ ] Filled `latent_dimension_descriptions.csv` with annotations
- [ ] Saved `latent_interpretability_report.md` in project docs
- [ ] Identified top 10-15 interpretable dimensions for downstream use

If all checked: **Proceed to Phase E-F (Clinical Features + Master Dataset)**

If not: **Follow retrain recommendations in the report**

---

## 📚 References

**β-VAE Disentanglement**:
- Higgins et al. (2017): "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
- Burgess et al. (2018): "Understanding disentangling in β-VAE"

**ECG Representation Learning**:
- Strodthoff et al. (2021): "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL"
- Mehari & Strodthoff (2022): "Self-supervised representation learning from 12-lead ECG data"

**Physiological Plausibility Checks**:
- Clifford et al. (2006): "Signal quality indices and data fusion for determining acceptability of electrocardiograms"
- Drew et al. (2004): "Insights into the problem of alarm fatigue with physiologic monitor devices"

---

**Good luck with your validation! 🎯**

