# ECG-Based MI Risk Prediction Using Deep Learning & Causal Analysis

**Capstone Project - MIMIC-IV-ECG Analysis**  
**Date:** November 2, 2025  
**Status:** 🔄 Phase D.3 Training (Epoch 15/160)

---

## 📋 Project Overview

This project develops a deep learning pipeline to analyze 12-lead ECG signals from MIMIC-IV for myocardial infarction (MI) risk prediction using:
- **β-VAE** (Variational Autoencoder) for unsupervised ECG representation learning
- **Causal Analysis** (CATE estimation) to identify high-risk patient subgroups
- **Clinical Validation** with adjudication and interpretability checks

### Key Objectives
1. Extract meaningful latent representations from 12-lead ECG signals
2. Identify pre-incident MI signatures in asymptomatic patients
3. Estimate conditional average treatment effects (CATE) for targeted interventions
4. Validate findings through clinical adjudication and negative controls

---

## 🏗️ Project Structure

```
Capstone/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── requirements.yml                   # Conda environment
├── install_vae_requirements.ps1       # Installation script
│
├── src/                               # Source code
│   ├── models/
│   │   ├── vae_conv1d.py             # Conv1D β-VAE (82.4M params)
│   │   └── train_vae.py              # Training pipeline
│   ├── data/
│   │   ├── ecg_dataset.py            # PyTorch Dataset
│   │   ├── phase_b_ingestion.py      # Data ingestion
│   │   └── phase_c_cohort_labeling.py # Cohort creation
│   └── features/
│       └── phase_d_feature_extraction.py # ECG feature extraction
│
├── scripts/                           # Utility scripts
│   ├── extract_latent_embeddings.py  # Phase D.4
│   ├── validate_latent_interpretability.py # Phase D.5
│   ├── preflight_vae_training.py     # Pre-training checks
│   └── analyze_adjudication.py       # Clinical validation
│
├── data/                              # Datasets
│   ├── raw/                          # Original MIMIC-IV-ECG data
│   ├── processed/                    # Processed parquet files
│   └── interim/                      # Intermediate outputs
│
├── models/checkpoints/                # VAE training checkpoints
├── notebooks/                         # Analysis notebooks
├── reports/                           # Reports and figures
└── cuda_env/                         # Python virtual environment
```

---

## 🚀 Quick Start

### Prerequisites
- Windows OS with PowerShell
- NVIDIA GPU (RTX 4050, 6GB VRAM tested)
- Python 3.12+ with CUDA support
- MIMIC-IV-ECG dataset access

### Installation

```powershell
# 1. Clone/navigate to project directory
cd C:\Users\Acer\Desktop\Capstone\Capstone

# 2. Activate virtual environment
.\cuda_env\Scripts\activate

# 3. Install required packages (if not already installed)
.\install_vae_requirements.ps1

# 4. Verify installation
python scripts/preflight_vae_training.py
```

### Current Training Status

**VAE Training (Phase D.3):**
- Model: Conv1D β-VAE with 82.4M parameters
- Training: Epoch 15/160 (β-annealing cycle 1/4)
- Dataset: 44,830 ECGs (Control_Symptomatic + MI_Pre-Incident)
- Hardware: RTX 4050 GPU (~70% utilization, batch_size=256)

**Latest Metrics (Epoch 13 - Best):**
```
Val Loss:   34,857 (Recon: 34,282, KL_raw: 272)
Train Loss: 31,651 (Recon: 31,456, KL_raw: 290)
Learning Rate: 1e-4, β: 1.20
```

---

## 📊 Dataset Information

### MIMIC-IV-ECG Dataset
- **Total ECGs**: 47,852 unique studies
- **Patients**: ~1,000 unique subjects
- **Format**: WFDB 12-lead (500 Hz, 10 seconds)
- **Location**: `data/raw/MIMIC-IV-ECG-1.0/files/`

### Cohort Labels
| Label | Count | Purpose |
|-------|-------|---------|
| Control_Symptomatic | 41,894 | Training (normal physiology) |
| MI_Pre-Incident | 2,936 | Training (pre-MI signatures) |
| MI_Acute_Presentation | 3,022 | Hold-out (CATE analysis) |

### Processed Data
- **Features**: `data/processed/ecg_features_with_demographics.parquet` (47,852 × 29)
- **Cohorts**: `cohort_master.parquet`, `cohort_strict.parquet`, `cohort_broad.parquet`
- **Database**: `data/mimic_database.duckdb` (DuckDB indexed)

---

## 🔧 Phase-by-Phase Workflow

### Phase A: Project Planning ✅
- [x] Define research question
- [x] Select datasets (MIMIC-IV-ECG, PTB-XL)
- [x] Plan analysis pipeline

### Phase B: Data Ingestion ✅
- [x] Download MIMIC-IV-ECG (1.0)
- [x] Create DuckDB database
- [x] Index record metadata
- [x] Verify file integrity

### Phase C: Cohort Definition ✅
- [x] Define MI labels (Acute, Pre-Incident)
- [x] Apply troponin criteria
- [x] Create control groups
- [x] Generate adjudication sample (100 cases)

### Phase D: Feature Engineering & VAE 🔄
- [x] **D.1-D.2**: Extract clinical features (QRS, QT, HR)
- [x] **D.3**: Train Conv1D β-VAE (Currently: Epoch 15/160)
- [ ] **D.4**: Extract latent embeddings (μ vectors)
- [ ] **D.5**: Validate interpretability (≥10 dims, ≥95% plausible)

### Phase E: CATE Analysis (Pending)
- [ ] Baseline propensity score models
- [ ] Causal forest / S-learner
- [ ] Estimate heterogeneous treatment effects
- [ ] Identify high-risk subgroups

### Phase F: Validation (Pending)
- [ ] Clinical adjudication (inter-rater reliability)
- [ ] Negative control outcomes
- [ ] Sensitivity analyses

---

## 🧠 VAE Architecture

### Conv1D β-VAE Specifications
- **Input**: (batch, 12 leads, 5000 timesteps) at 500 Hz
- **Latent Dimension**: z_dim = 64
- **β Parameter**: β = 4.0 (cyclical annealing: 0 → 4 → 0, 4 cycles)
- **Free-Bits**: 2.0 (prevents posterior collapse)
- **Architecture**: 
  - Encoder: Conv1D (12→32→64→128) + FC (80k→512→256→64)
  - Decoder: FC (64→256→512→80k) + ConvTranspose1D (128→64→32→12)
- **Parameters**: 82,424,076 total
- **Loss**: L = MSE(reconstruction) + β × max(KL - free_bits, 0)

### Key Design Choices
1. **Cyclical β-Annealing**: 4 cycles × 40 epochs prevents KL collapse
2. **Free-Bits Constraint**: Enforces minimum KL=2.0 per dimension
3. **Increased Capacity**: 82M params (up from 41M) for better reconstruction
4. **Batch Size 256**: Optimized for RTX 4050 (~70% GPU utilization)

---

## 📈 Training Commands

### Train VAE (Phase D.3)
```powershell
# Current training configuration
python src/models/train_vae.py `
    --batch_size 256 `
    --epochs 160 `
    --lr 1e-4 `
    --num_workers 0

# Monitor progress in real-time (terminal output shows epoch metrics)
```

### Extract Latent Embeddings (Phase D.4)
```powershell
# After training completes (best_model.pt)
python scripts/extract_latent_embeddings.py `
    --checkpoint models/checkpoints/vae_zdim64_beta4.0_*/best_model.pt `
    --batch_size 64
```

### Validate Interpretability (Phase D.5)
```powershell
# Dimension traversal analysis
python scripts/validate_latent_interpretability.py `
    --checkpoint models/checkpoints/vae_zdim64_beta4.0_*/best_model.pt `
    --save_plots
```

---

## 🎯 Success Criteria

### Phase D.3 (VAE Training)
**Epoch 100 Targets:**
- KL_raw: 150-400 (healthy latent usage)
- Recon: 5,000-15,000 (good reconstruction)
- Learning Rate: 1e-5 to 5e-5 (properly reduced)

**Epoch 160 Targets (Final):**
- KL_raw: 200-350 (excellent), >150 (acceptable)
- Recon: 3,000-8,000 (excellent), 8,000-12,000 (good)
- Val/Train Gap: <2,000 (perfect), <5,000 (acceptable)

### Phase D.5 (Interpretability)
**Go/No-Go Criteria:**
- ✅ **PROCEED**: ≥10 interpretable dimensions AND ≥95% plausible ECGs
- ❌ **RETRAIN**: <10 interpretable OR <95% plausible

**Interpretable Dimension = One of:**
- Morphology: P/Q/R/S/T wave shape
- Intervals: PR, QRS, QT duration
- Rate: Heart rate changes
- Axis: Lead orientation shifts
- Noise: Baseline wander, artifacts

---

## 📁 Output Files

### Training Outputs
```
models/checkpoints/vae_zdim64_beta4.0_20251102_073456/
├── best_model.pt              # Best checkpoint (lowest val loss)
├── checkpoint_epoch_10.pt     # Periodic checkpoint
└── config.json                # Training hyperparameters
```

### Latent Embeddings
```
data/processed/ecg_z_embeddings.parquet
Columns: [subject_id, study_id, z_ecg_1, ..., z_ecg_64]
Shape: (47,852 ECGs × 66 columns)
```

### Interpretability Results
```
results/latent_interpretability/
├── dimension_plots/
│   ├── dimension_001_traversal.png
│   ├── dimension_002_traversal.png
│   └── ... (64 total)
├── interpretability_results.csv
└── latent_interpretability_report.md  # Go/No-Go decision
```

---

## 🔍 Clinical Adjudication

### Adjudication Package
- **Sample**: 100 cases in `data/processed/adjudication_sample.csv`
- **Instructions**: `data/processed/ADJUDICATION_INSTRUCTIONS.md`
- **Quick Reference**: `data/processed/ADJUDICATION_QUICK_REFERENCE.md`
- **Email Template**: `data/processed/EMAIL_TEMPLATE.txt`

### Analysis Script
```powershell
# After receiving reviewed file
python scripts/analyze_adjudication.py
```

**Outputs:**
- Agreement rate & Cohen's κ
- Confusion matrix
- Disagreement cases CSV

---

## ⚙️ Hardware Requirements

### Minimum Specifications
- GPU: NVIDIA RTX 3060 (6GB VRAM)
- RAM: 16GB
- Storage: 100GB free (for MIMIC-IV-ECG dataset)
- OS: Windows 10/11 with CUDA 12.1+

### Tested Configuration
- GPU: RTX 4050 Laptop (6.44 GB VRAM)
- RAM: 32GB
- CPU: 24 cores
- Training Speed: ~11s/batch (batch_size=256)

---

## 📚 Key Dependencies

```
PyTorch:    2.5.1+cu121  # Deep learning framework
WFDB:       4.3.0        # ECG file I/O
NeuroKit2:  0.2.12       # ECG signal processing
PyArrow:    21.0.0       # Parquet file handling
NumPy:      2.2.6        # Numerical computing
Pandas:     2.3.3        # Data manipulation
Matplotlib: 3.10.7       # Visualization
```

Full list: See `requirements.txt`

---

## 🐛 Troubleshooting

### Issue: OOM (Out of Memory) Error
```powershell
# Reduce batch size
python src/models/train_vae.py --batch_size 128 --epochs 160 --lr 1e-4 --num_workers 0
```

### Issue: KL Divergence → 0 (Posterior Collapse)
**Current solution implemented:**
- Free-bits constraint (2.0)
- Cyclical β-annealing (4 cycles)
- Increased model capacity (82M params)

If still occurring, increase β:
```powershell
python src/models/train_vae.py --beta 8.0 --epochs 160 --num_workers 0
```

### Issue: Poor Reconstruction Quality
Reduce β for better reconstruction:
```powershell
python src/models/train_vae.py --beta 2.0 --epochs 160 --num_workers 0
```

### Issue: Slow Training
- Close background applications
- Check GPU temperature (thermal throttling)
- Reduce `--batch_size` if disk I/O is bottleneck

---

## 📖 Documentation

- **DIRECTORY_AUDIT_REPORT.md**: Complete file structure analysis
- **data/processed/ADJUDICATION_*.md**: Clinical validation guides
- **reports/**: Phase-specific reports and figures

---

## 🙏 Acknowledgments

- **Dataset**: MIMIC-IV-ECG (Johnson et al., PhysioNet)
- **Validation**: PTB-XL & PTB-XL+ (Strodthoff et al., PhysioNet)
- **Tools**: PyTorch, WFDB, NeuroKit2

---

## 📞 Contact

For questions or issues:
1. Review `DIRECTORY_AUDIT_REPORT.md` for file organization
2. Check training logs in terminal output
3. Verify environment: `python scripts/preflight_vae_training.py`

---

**Last Updated:** November 2, 2025  
**Current Phase:** D.3 (VAE Training - Epoch 15/160)  
**Next Milestone:** Complete training → Extract embeddings (D.4) → Validate interpretability (D.5)
