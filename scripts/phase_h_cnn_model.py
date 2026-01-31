"""
Phase H: CNN Model Training (Raw Waveform + Clinical Features)
================================================================
This script trains a CNN model on raw ECG waveforms + clinical features.
Separated from main baseline models script to avoid retraining XGBoost/MLP.

Author: Capstone Team
Date: December 2025
"""

import sys
import numpy as np
import pandas as pd

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wfdb
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.metrics import brier_score_loss, confusion_matrix
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ECG_BASE_PATH = Path("F:/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0")
OUTPUT_DIR = BASE_DIR / "models" / "baseline"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("\n" + "=" * 80)
print("Phase H: CNN Model Training (Raw Waveform + Clinical)")
print("=" * 80)

# ============================================================================
# Load Data and Existing Splits
# ============================================================================

print("\n[1] Loading master dataset and existing splits...")
df = pd.read_parquet(DATA_DIR / "processed" / "master_dataset.parquet")
print(f"✓ Loaded {len(df)} records")

# Load existing splits (from phase_h_baseline_models.py)
splits = pd.read_parquet(OUTPUT_DIR / "split_assignments.parquet")
df = df.merge(splits[['subject_id', 'h_split']], on='subject_id', how='left')

# Use the new split assignments from phase_h_baseline_models.py
df['split'] = df['h_split']

train_idx = df[df['split'] == 'train'].index
val_idx = df[df['split'] == 'val'].index
test_idx = df[df['split'] == 'test'].index

print(f"\nUsing existing splits:")
print(f"  Train: {len(train_idx)} records")
print(f"  Val:   {len(val_idx)} records")
print(f"  Test:  {len(test_idx)} records")

# Extract features
# Target: MI_Acute_Presentation (binary classification)
df['target'] = (df['Label'] == 'MI_Acute_Presentation').astype(int)

# Clinical features (same as XGBoost)
all_features = [c for c in df.columns if c not in ['record_id', 'subject_id', 'hadm_id', 'study_id', 
                                                     'Label', 'file_path', 'ecg_time', 'target', 
                                                     'split', 'split_new']]
clinical_features = [f for f in all_features if pd.api.types.is_numeric_dtype(df[f])]
clinical_features = [f for f in clinical_features if not f.startswith('latent_')]

print(f"\nUsing {len(clinical_features)} clinical features")
print(f"Sample features: {clinical_features[:5]}")

# Prepare clinical data
X_clinical_train = df.loc[train_idx, clinical_features].values.astype(np.float32)
X_clinical_val = df.loc[val_idx, clinical_features].values.astype(np.float32)
X_clinical_test = df.loc[test_idx, clinical_features].values.astype(np.float32)

y_train = df.loc[train_idx, 'target'].values.astype(np.float32)
y_val = df.loc[val_idx, 'target'].values.astype(np.float32)
y_test = df.loc[test_idx, 'target'].values.astype(np.float32)

print(f"\nTarget distribution:")
print(f"  Train MI_Acute: {y_train.sum()}/{len(y_train)} ({100*y_train.mean():.2f}%)")
print(f"  Val MI_Acute: {y_val.sum()}/{len(y_val)} ({100*y_val.mean():.2f}%)")
print(f"  Test MI_Acute: {y_test.sum()}/{len(y_test)} ({100*y_test.mean():.2f}%)")

# ============================================================================
# Load ECG Waveforms
# ============================================================================

print("\n[2] Loading ECG waveforms...")
print("⚠ This will take 5-10 minutes for ~48k waveforms\n")

def load_ecg_waveform(file_path, base_path):
    """Load 12-lead ECG waveform from WFDB file"""
    try:
        full_path = base_path / file_path
        full_path = full_path.with_suffix('')
        record = wfdb.rdrecord(str(full_path))
        
        # Resample to 5000 samples if needed
        signal = record.p_signal.T  # Shape: (12, N)
        
        if signal.shape[1] != 5000:
            # Simple interpolation to 5000 samples
            x_old = np.linspace(0, 1, signal.shape[1])
            x_new = np.linspace(0, 1, 5000)
            signal_resampled = np.zeros((12, 5000))
            for lead in range(12):
                f = interp1d(x_old, signal[lead], kind='linear')
                signal_resampled[lead] = f(x_new)
            signal = signal_resampled
        
        # Normalize per lead
        signal = (signal - signal.mean(axis=1, keepdims=True)) / (signal.std(axis=1, keepdims=True) + 1e-8)
        
        return signal.astype(np.float32)
    except Exception as e:
        # Return zeros for failed loads
        return None

def load_waveforms_batch(indices, desc):
    """Load waveforms with progress bar"""
    waveforms = []
    failed_count = 0
    for idx in tqdm(indices, desc=desc):
        waveform = load_ecg_waveform(df.loc[idx, 'file_path'], ECG_BASE_PATH)
        if waveform is not None:
            waveforms.append(waveform)
        else:
            # Use zeros for failed loads
            waveforms.append(np.zeros((12, 5000), dtype=np.float32))
            failed_count += 1
    if failed_count > 0:
        print(f"  ⚠ {failed_count} waveforms failed to load (using zeros)")
    return waveforms

train_waveforms = load_waveforms_batch(train_idx, "Loading train waveforms")
val_waveforms = load_waveforms_batch(val_idx, "Loading val waveforms")
test_waveforms = load_waveforms_batch(test_idx, "Loading test waveforms")

print(f"\n✓ Loaded {len(train_waveforms)} train, {len(val_waveforms)} val, {len(test_waveforms)} test waveforms")

# ============================================================================
# CNN Dataset and DataLoader
# ============================================================================

print("\n[3] Creating CNN datasets...")

class CNNDataset(Dataset):
    def __init__(self, waveforms, clinical, targets):
        self.waveforms = torch.FloatTensor(np.array(waveforms))
        self.clinical = torch.FloatTensor(clinical)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.waveforms)
    
    def __getitem__(self, idx):
        return self.waveforms[idx], self.clinical[idx], self.targets[idx]

train_dataset = CNNDataset(train_waveforms, X_clinical_train, y_train)
val_dataset = CNNDataset(val_waveforms, X_clinical_val, y_val)
test_dataset = CNNDataset(test_waveforms, X_clinical_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

print(f"✓ Created dataloaders (batch_size: train=32, val/test=64)")

# ============================================================================
# CNN Model Architecture
# ============================================================================

print("\n[4] Initializing CNN model...")

class ECGCNNClassifier(nn.Module):
    def __init__(self, num_clinical_features):
        super().__init__()
        
        # ECG pathway - processes raw 12-lead waveform
        self.ecg_pathway = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=10, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Clinical pathway
        self.clinical_pathway = nn.Sequential(
            nn.Linear(num_clinical_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Merged pathway
        self.classifier = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, ecg, clinical):
        # ECG features
        ecg_features = self.ecg_pathway(ecg)
        ecg_features = self.gap(ecg_features).squeeze(-1)
        
        # Clinical features
        clinical_features = self.clinical_pathway(clinical)
        
        # Merge and classify
        merged = torch.cat([ecg_features, clinical_features], dim=1)
        output = self.classifier(merged)
        
        return output.squeeze()

model = ECGCNNClassifier(num_clinical_features=len(clinical_features)).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

print("\nCNN Architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# Training Loop
# ============================================================================

print("\n[5] Training CNN model...")
print("Expected time: 30-60 minutes\n")

num_epochs = 50
best_val_auc = 0
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    train_preds = []
    train_targets = []
    
    for ecg_batch, clinical_batch, target_batch in train_loader:
        ecg_batch = ecg_batch.to(device)
        clinical_batch = clinical_batch.to(device)
        target_batch = target_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(ecg_batch, clinical_batch)
        loss = criterion(outputs, target_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_preds.extend(outputs.detach().cpu().numpy())
        train_targets.extend(target_batch.cpu().numpy())
    
    train_loss /= len(train_loader)
    train_auc = roc_auc_score(train_targets, train_preds)
    
    # Validation
    model.eval()
    val_preds = []
    val_targets = []
    val_loss = 0
    
    with torch.no_grad():
        for ecg_batch, clinical_batch, target_batch in val_loader:
            ecg_batch = ecg_batch.to(device)
            clinical_batch = clinical_batch.to(device)
            target_batch = target_batch.to(device)
            
            outputs = model(ecg_batch, clinical_batch)
            loss = criterion(outputs, target_batch)
            val_loss += loss.item()
            
            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(target_batch.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_auc = roc_auc_score(val_targets, val_preds)
    
    # Print progress
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
    
    # Early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), OUTPUT_DIR / 'cnn_model_best.pt')
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print(f"\n✓ Training complete!")
print(f"  Best validation AUC: {best_val_auc:.4f}")

# ============================================================================
# Evaluation
# ============================================================================

print("\n[6] Evaluating CNN model...")

# Load best model
model.load_state_dict(torch.load(OUTPUT_DIR / 'cnn_model_best.pt'))
model.eval()

def get_predictions(loader):
    """Get predictions from model"""
    preds = []
    with torch.no_grad():
        for ecg_batch, clinical_batch, target_batch in loader:
            ecg_batch = ecg_batch.to(device)
            clinical_batch = clinical_batch.to(device)
            outputs = model(ecg_batch, clinical_batch)
            preds.extend(outputs.cpu().numpy())
    return np.array(preds)

y_pred_train = get_predictions(train_loader)
y_pred_val = get_predictions(val_loader)
y_pred_test = get_predictions(test_loader)

# Calculate metrics
auc_train = roc_auc_score(y_train, y_pred_train)
auc_val = roc_auc_score(y_val, y_pred_val)
auc_test = roc_auc_score(y_test, y_pred_test)

auprc_train = average_precision_score(y_train, y_pred_train)
auprc_val = average_precision_score(y_val, y_pred_val)
auprc_test = average_precision_score(y_test, y_pred_test)

brier_train = brier_score_loss(y_train, y_pred_train)
brier_val = brier_score_loss(y_val, y_pred_val)
brier_test = brier_score_loss(y_test, y_pred_test)

print("\n✓ CNN Performance:")
print(f"\n  Train AUC-ROC: {auc_train:.4f} | AUPRC: {auprc_train:.4f} | Brier: {brier_train:.4f}")
print(f"  Val   AUC-ROC: {auc_val:.4f} | AUPRC: {auprc_val:.4f} | Brier: {brier_val:.4f}")
print(f"  Test  AUC-ROC: {auc_test:.4f} | AUPRC: {auprc_test:.4f} | Brier: {brier_test:.4f}")

# ============================================================================
# Save Results
# ============================================================================

print("\n[7] Saving results...")

# Save predictions
cnn_results = pd.DataFrame({
    'subject_id': df.loc[test_idx, 'subject_id'].values,
    'y_true': y_test,
    'y_pred_cnn': y_pred_test
})
cnn_results.to_csv(OUTPUT_DIR / 'cnn_predictions.csv', index=False)

# Append to existing metrics file
metrics_file = OUTPUT_DIR / 'model_evaluation_metrics.csv'
if metrics_file.exists():
    existing_metrics = pd.read_csv(metrics_file)
    
    # Add CNN metrics
    cnn_metrics = pd.DataFrame([
        {'Model': 'CNN', 'Split': 'Train', 'AUROC': auc_train, 'AUPRC': auprc_train, 'Brier Score': brier_train},
        {'Model': 'CNN', 'Split': 'Val', 'AUROC': auc_val, 'AUPRC': auprc_val, 'Brier Score': brier_val},
        {'Model': 'CNN', 'Split': 'Test', 'AUROC': auc_test, 'AUPRC': auprc_test, 'Brier Score': brier_test}
    ])
    
    # Remove old CNN entries if they exist
    existing_metrics = existing_metrics[existing_metrics['Model'] != 'CNN']
    
    # Append new CNN metrics
    updated_metrics = pd.concat([existing_metrics, cnn_metrics], ignore_index=True)
    updated_metrics.to_csv(metrics_file, index=False)
    print(f"✓ Updated metrics saved to {metrics_file}")
else:
    print("⚠ Warning: model_evaluation_metrics.csv not found, creating new file")
    cnn_metrics = pd.DataFrame([
        {'Model': 'CNN', 'Split': 'Train', 'AUROC': auc_train, 'AUPRC': auprc_train, 'Brier Score': brier_train},
        {'Model': 'CNN', 'Split': 'Val', 'AUROC': auc_val, 'AUPRC': auprc_val, 'Brier Score': brier_val},
        {'Model': 'CNN', 'Split': 'Test', 'AUROC': auc_test, 'AUPRC': auprc_test, 'Brier Score': brier_test}
    ])
    cnn_metrics.to_csv(metrics_file, index=False)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_test)
axes[0].plot(fpr, tpr, label=f'CNN (AUC = {auc_test:.3f})', linewidth=2)
axes[0].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('CNN ROC Curve', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Calibration Plot
prob_true, prob_pred = calibration_curve(y_test, y_pred_test, n_bins=5, strategy='quantile')
axes[1].plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8, label='CNN')
axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)
axes[1].set_xlabel('Mean Predicted Probability', fontsize=12)
axes[1].set_ylabel('Fraction of Positives', fontsize=12)
axes[1].set_title('CNN Calibration Plot', fontsize=14)
axes[1].legend(fontsize=10, loc='upper left')
axes[1].grid(alpha=0.3)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'cnn_performance.png', dpi=300, bbox_inches='tight')
print(f"✓ Plots saved to {PLOTS_DIR / 'cnn_performance.png'}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("✓ Phase H: CNN Model - COMPLETE")
print("=" * 80)

summary = f"""
CNN Model Summary
=================

Architecture:
- ECG Pathway: 4 Conv1D layers (12→32→64→128→256) + Global Average Pooling
- Clinical Pathway: 2 FC layers (16→64→32)
- Merged Classifier: 3 FC layers (288→128→64→1)
- Total parameters: {total_params:,}

Training:
- Epochs: {epoch+1} (early stopped)
- Best validation AUC: {best_val_auc:.4f}
- Optimizer: Adam (lr=0.0001, weight_decay=1e-5)
- Loss: Binary Cross-Entropy

Performance:
- Train AUC-ROC: {auc_train:.4f}
- Val   AUC-ROC: {auc_val:.4f}
- Test  AUC-ROC: {auc_test:.4f}

Artifacts Saved:
- models/baseline/cnn_model_best.pt
- models/baseline/cnn_predictions.csv
- models/baseline/model_evaluation_metrics.csv (updated)
- models/baseline/plots/cnn_performance.png

Next Steps:
- Compare CNN with XGBoost and MLP results
- Analyze if raw waveforms provide better performance than VAE embeddings
- Proceed to Phase I: Propensity Score Modeling
"""

print(summary)

with open(OUTPUT_DIR / "cnn_summary.txt", 'w') as f:
    f.write(summary)

print("\n✓ All done! CNN model training complete.")
