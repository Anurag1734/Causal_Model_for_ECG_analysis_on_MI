"""
Phase H: Baseline Predictive Models

Establish non-causal performance benchmarks for predicting MI_Acute_Presentation.

Models:
1. XGBoost (Tabular): ECG features + Clinical features
2. MLP (Latent): VAE embeddings + Clinical features  
3. CNN (End-to-End): Raw waveform + Clinical features

Critical: Grouped split by subject_id to prevent data leakage.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, 
    precision_recall_curve, brier_score_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import wfdb
import neurokit2 as nk

# Paths
MASTER_DATASET_PATH = "data/processed/master_dataset.parquet"
ECG_BASE_PATH = Path("data/raw/MIMIC-IV-ECG-1.0/files")
OUTPUT_DIR = Path("models/baseline")
PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Config
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print("\n" + "=" * 80)
print("Phase H: Baseline Predictive Models")
print("=" * 80)

# ============================================================================
# H.1: Load Data
# ============================================================================

print("\n[H.1] Loading master dataset...")
df = pd.read_parquet(MASTER_DATASET_PATH)
print(f"✓ Loaded {len(df)} records")

# Create binary target: MI_Acute_Presentation vs Others
df['target'] = (df['Label'] == 'MI_Acute_Presentation').astype(int)

print(f"\nTarget distribution:")
print(df['target'].value_counts())
print(f"  Class balance: {100 * df['target'].mean():.2f}% MI_Acute")

# Check subject_id availability
if 'subject_id' not in df.columns:
    print("\n⚠ WARNING: subject_id not found, attempting to extract from file_path...")
    # Extract from path: p10/p10000032/s41256771/...
    df['subject_id'] = df['file_path'].str.split('/').str[1].str.replace('p', '').astype(int)
    print(f"✓ Extracted subject_id from {len(df['subject_id'].unique())} unique subjects")

print(f"\nUnique subjects: {df['subject_id'].nunique()}")
print(f"Records per subject: {len(df) / df['subject_id'].nunique():.2f} avg")

# ============================================================================
# H.2: Grouped Train/Validation/Test Split (70/15/15)
# ============================================================================

print("\n[H.2] Creating grouped train/validation/test split...")

# Prepare features
ecg_features = [c for c in df.columns if c.startswith('ecg_') and c not in ['ecg_datetime', 'ecg_study_id']]
vae_features = [c for c in df.columns if c.startswith('z_ecg_')]
clinical_features = [
    'age', 'sex_M',
    'statin_use', 'total_cholesterol', 'ldl', 'hdl', 'triglycerides',
    'glucose', 'creatinine', 'troponin', 'bnp', 'crp',
    'heart_rate', 'sbp', 'dbp', 'temperature', 'respiratory_rate', 'spo2',
    'diabetes', 'hypertension', 'ckd', 'chf', 'cad', 'prior_mi', 'stroke'
]

# Filter to available features and ensure they are numeric
ecg_features = [f for f in ecg_features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
clinical_features = [f for f in clinical_features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]

print(f"\nFeature counts:")
print(f"  ECG features: {len(ecg_features)}")
print(f"  VAE features: {len(vae_features)}")
print(f"  Clinical features: {len(clinical_features)}")

# Debug: Show first few features of each type
if len(ecg_features) > 0:
    print(f"  Sample ECG features: {ecg_features[:3]}")
if len(clinical_features) > 0:
    print(f"  Sample clinical features: {clinical_features[:3]}")

# Grouped split (70% train, 30% temp)
splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
train_idx, temp_idx = next(splitter.split(df, df['target'], groups=df['subject_id']))

# Split temp into validation (50%) and test (50%) → 15% each of total
temp_df = df.iloc[temp_idx].copy()
splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_SEED)
val_idx_temp, test_idx_temp = next(splitter2.split(temp_df, temp_df['target'], groups=temp_df['subject_id']))

# Map back to original indices
val_idx = temp_idx[val_idx_temp]
test_idx = temp_idx[test_idx_temp]

# Create split column
df['h_split'] = 'train'
df.loc[val_idx, 'h_split'] = 'val'
df.loc[test_idx, 'h_split'] = 'test'

# Verify no subject leakage
train_subjects = set(df.loc[train_idx, 'subject_id'])
val_subjects = set(df.loc[val_idx, 'subject_id'])
test_subjects = set(df.loc[test_idx, 'subject_id'])

assert len(train_subjects & val_subjects) == 0, "Subject leakage: train-val overlap!"
assert len(train_subjects & test_subjects) == 0, "Subject leakage: train-test overlap!"
assert len(val_subjects & test_subjects) == 0, "Subject leakage: val-test overlap!"

print(f"\n✓ Grouped split created (no subject leakage):")
print(f"  Train: {len(train_idx)} records ({len(train_subjects)} subjects)")
print(f"  Val:   {len(val_idx)} records ({len(val_subjects)} subjects)")
print(f"  Test:  {len(test_idx)} records ({len(test_subjects)} subjects)")

# Target distribution per split
for split_name, indices in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
    split_targets = df.loc[indices, 'target']
    n_pos = split_targets.sum()
    n_total = len(split_targets)
    print(f"  {split_name} MI_Acute: {n_pos}/{n_total} ({100 * n_pos / n_total:.2f}%)")

# Save split assignments
df[['subject_id', 'h_split']].to_parquet(OUTPUT_DIR / "split_assignments.parquet")

# ============================================================================
# H.3: Model 1 - XGBoost (Tabular)
# ============================================================================

print("\n" + "=" * 80)
print("[H.3] Model 1: XGBoost (Tabular)")
print("=" * 80)

# Prepare data
tabular_features = ecg_features + clinical_features
print(f"\nUsing {len(tabular_features)} features (ECG + Clinical)")

X_train_tab = df.loc[train_idx, tabular_features].fillna(0).values
y_train = df.loc[train_idx, 'target'].values
groups_train = df.loc[train_idx, 'subject_id'].values

X_val_tab = df.loc[val_idx, tabular_features].fillna(0).values
y_val = df.loc[val_idx, 'target'].values

X_test_tab = df.loc[test_idx, tabular_features].fillna(0).values
y_test = df.loc[test_idx, 'target'].values

print(f"\nTrain shape: {X_train_tab.shape}")
print(f"Val shape: {X_val_tab.shape}")
print(f"Test shape: {X_test_tab.shape}")

# Hyperparameter tuning with grouped CV
print("\nPerforming hyperparameter tuning (RandomizedSearchCV)...")

param_distributions = {
    'n_estimators': [300, 500, 700],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.005, 0.01, 0.02, 0.05],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

base_xgb = xgb.XGBClassifier(
    eval_metric='auc',
    early_stopping_rounds=50,
    random_state=RANDOM_SEED,
    use_label_encoder=False,
    tree_method='hist',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Grouped 5-fold CV
from sklearn.model_selection import GroupKFold
cv = GroupKFold(n_splits=5)

random_search = RandomizedSearchCV(
    base_xgb,
    param_distributions=param_distributions,
    n_iter=20,
    cv=cv,
    scoring='roc_auc',
    random_state=RANDOM_SEED,
    verbose=1,
    n_jobs=-1
)

random_search.fit(
    X_train_tab, y_train, 
    groups=groups_train,
    eval_set=[(X_val_tab, y_val)],
    verbose=False
)

print(f"\n✓ Best hyperparameters found:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\n  Best CV AUC-ROC: {random_search.best_score_:.4f}")

# Train final model with best params
xgb_model = random_search.best_estimator_

# Predictions
y_pred_xgb_train = xgb_model.predict_proba(X_train_tab)[:, 1]
y_pred_xgb_val = xgb_model.predict_proba(X_val_tab)[:, 1]
y_pred_xgb_test = xgb_model.predict_proba(X_test_tab)[:, 1]

# Metrics
auc_train_xgb = roc_auc_score(y_train, y_pred_xgb_train)
auc_val_xgb = roc_auc_score(y_val, y_pred_xgb_val)
auc_test_xgb = roc_auc_score(y_test, y_pred_xgb_test)

print(f"\n✓ XGBoost Performance:")
print(f"  Train AUC-ROC: {auc_train_xgb:.4f}")
print(f"  Val AUC-ROC:   {auc_val_xgb:.4f}")
print(f"  Test AUC-ROC:  {auc_test_xgb:.4f}")

# Save model
with open(OUTPUT_DIR / "xgboost_model.pkl", 'wb') as f:
    pickle.dump(xgb_model, f)

# ============================================================================
# H.4: Model 2 - MLP (Latent Space)
# ============================================================================

print("\n" + "=" * 80)
print("[H.4] Model 2: MLP (VAE Latent + Clinical)")
print("=" * 80)

latent_features = vae_features + clinical_features
print(f"\nUsing {len(latent_features)} features ({len(vae_features)} VAE + {len(clinical_features)} Clinical)")

X_train_lat = df.loc[train_idx, latent_features].fillna(0).values
X_val_lat = df.loc[val_idx, latent_features].fillna(0).values
X_test_lat = df.loc[test_idx, latent_features].fillna(0).values

# PyTorch Dataset
class LatentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset_lat = LatentDataset(X_train_lat, y_train)
val_dataset_lat = LatentDataset(X_val_lat, y_val)
test_dataset_lat = LatentDataset(X_test_lat, y_test)

train_loader_lat = DataLoader(train_dataset_lat, batch_size=64, shuffle=True)
val_loader_lat = DataLoader(val_dataset_lat, batch_size=64, shuffle=False)
test_loader_lat = DataLoader(test_dataset_lat, batch_size=64, shuffle=False)

# MLP Architecture
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x).squeeze()

mlp_model = MLPClassifier(input_dim=len(latent_features)).to(DEVICE)
optimizer = optim.Adam(mlp_model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

print(f"\nMLP Architecture:")
print(mlp_model)
print(f"\nTraining on device: {DEVICE}")

# Training loop
best_val_auc = 0.0
patience = 10
patience_counter = 0
history_mlp = {'train_loss': [], 'val_loss': [], 'val_auc': []}

for epoch in range(100):
    # Train
    mlp_model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader_lat:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = mlp_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader_lat)
    
    # Validation
    mlp_model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader_lat:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = mlp_model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            
            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(y_batch.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader_lat)
    val_auc = roc_auc_score(val_targets, val_preds)
    
    history_mlp['train_loss'].append(avg_train_loss)
    history_mlp['val_loss'].append(avg_val_loss)
    history_mlp['val_auc'].append(val_auc)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/100 | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}")
    
    # Early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        torch.save(mlp_model.state_dict(), OUTPUT_DIR / "mlp_model_best.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

# Load best model
mlp_model.load_state_dict(torch.load(OUTPUT_DIR / "mlp_model_best.pt", weights_only=True))

# Predictions
def get_predictions(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            preds.extend(outputs.cpu().numpy())
    return np.array(preds)

y_pred_mlp_train = get_predictions(mlp_model, train_loader_lat)
y_pred_mlp_val = get_predictions(mlp_model, val_loader_lat)
y_pred_mlp_test = get_predictions(mlp_model, test_loader_lat)

auc_test_mlp = roc_auc_score(y_test, y_pred_mlp_test)

print(f"\n✓ MLP Performance:")
print(f"  Best Val AUC-ROC: {best_val_auc:.4f}")
print(f"  Test AUC-ROC: {auc_test_mlp:.4f}")

# ============================================================================
# H.5: Model 3 - End-to-End CNN (Raw Waveform)
# ============================================================================

print("\n" + "=" * 80)
print("[H.5] Model 3: CNN (Raw Waveform + Clinical)")
print("=" * 80)
print("\n⚠ NOTE: CNN requires loading raw ECG waveforms (time-intensive)")
print("This will load ~48k waveforms (12 leads × 5000 samples each)")
print("Expected time: 5-10 minutes for loading, 30-60 minutes for training\n")

# Load raw ECG waveforms
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
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, signal.shape[1])
            x_new = np.linspace(0, 1, 5000)
            signal_resampled = np.zeros((12, 5000))
            for lead in range(12):
                f = interp1d(x_old, signal[lead], kind='linear')
                signal_resampled[lead] = f(x_new)
            signal = signal_resampled
        
        # Normalize
        signal = (signal - signal.mean(axis=1, keepdims=True)) / (signal.std(axis=1, keepdims=True) + 1e-8)
        
        return signal.astype(np.float32)
    except Exception as e:
        # Return None for failed loads - will be replaced with zeros
        return None

# CNN Dataset
class CNNDataset(Dataset):
    def __init__(self, waveforms, clinical, targets):
        self.waveforms = torch.FloatTensor(np.array(waveforms))
        self.clinical = torch.FloatTensor(clinical)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.waveforms)
    
    def __getitem__(self, idx):
        return self.waveforms[idx], self.clinical[idx], self.targets[idx]

# CNN Architecture
class ECGCNNClassifier(nn.Module):
    def __init__(self, num_clinical_features):
        super().__init__()
        
        # ECG pathway
        self.ecg_pathway = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=15, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=10, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Clinical pathway
        self.clinical_pathway = nn.Sequential(
            nn.Linear(num_clinical_features, 32),
            nn.ReLU()
        )
        
        # Merged pathway
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
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

# Load full waveforms for all splits
print("\nLoading ECG waveforms (full dataset - this will take several minutes)...")

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

print(f"✓ Loaded {len(train_waveforms)} train, {len(val_waveforms)} val, {len(test_waveforms)} test waveforms")

# Create CNN datasets
train_dataset_cnn = CNNDataset(train_waveforms, X_clinical_train, y_train)
val_dataset_cnn = CNNDataset(val_waveforms, X_clinical_val, y_val)
test_dataset_cnn = CNNDataset(test_waveforms, X_clinical_test, y_test)

train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=32, shuffle=True)
val_loader_cnn = DataLoader(val_dataset_cnn, batch_size=64, shuffle=False)
test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=64, shuffle=False)

# Initialize CNN model
cnn_model = ECGCNNClassifier(num_clinical_features=len(clinical_features)).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.0001)

print("\nCNN Architecture:")
print(cnn_model)
print(f"\nTraining on device: {device}")

# Training loop
num_epochs = 50
best_val_auc = 0
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    # Training
    cnn_model.train()
    train_loss = 0
    for ecg_batch, clinical_batch, target_batch in train_loader_cnn:
        ecg_batch = ecg_batch.to(device)
        clinical_batch = clinical_batch.to(device)
        target_batch = target_batch.to(device)
        
        optimizer.zero_grad()
        outputs = cnn_model(ecg_batch, clinical_batch)
        loss = criterion(outputs, target_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader_cnn)
    
    # Validation
    cnn_model.eval()
    val_preds = []
    val_targets = []
    val_loss = 0
    
    with torch.no_grad():
        for ecg_batch, clinical_batch, target_batch in val_loader_cnn:
            ecg_batch = ecg_batch.to(device)
            clinical_batch = clinical_batch.to(device)
            target_batch = target_batch.to(device)
            
            outputs = cnn_model(ecg_batch, clinical_batch)
            loss = criterion(outputs, target_batch)
            val_loss += loss.item()
            
            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(target_batch.cpu().numpy())
    
    val_loss /= len(val_loader_cnn)
    val_auc = roc_auc_score(val_targets, val_preds)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
    
    # Early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(cnn_model.state_dict(), OUTPUT_DIR / 'cnn_model_best.pt')
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

# Load best model
cnn_model.load_state_dict(torch.load(OUTPUT_DIR / 'cnn_model_best.pt'))

# Get test predictions
cnn_model.eval()
test_preds_cnn = []

with torch.no_grad():
    for ecg_batch, clinical_batch, target_batch in test_loader_cnn:
        ecg_batch = ecg_batch.to(device)
        clinical_batch = clinical_batch.to(device)
        
        outputs = cnn_model(ecg_batch, clinical_batch)
        test_preds_cnn.extend(outputs.cpu().numpy())

y_pred_cnn_test = np.array(test_preds_cnn)
auc_test_cnn = roc_auc_score(y_test, y_pred_cnn_test)

print(f"\n✓ CNN Performance:")
print(f"  Best Val AUC-ROC: {best_val_auc:.4f}")
print(f"  Test AUC-ROC: {auc_test_cnn:.4f}")

# ============================================================================
# H.6: Evaluation Metrics
# ============================================================================

print("\n" + "=" * 80)
print("[H.6] Comprehensive Evaluation")
print("=" * 80)

# Collect predictions
# Get CNN train and val predictions for complete evaluation
cnn_model.eval()
train_preds_cnn = []
val_preds_cnn = []

with torch.no_grad():
    for ecg_batch, clinical_batch, target_batch in train_loader_cnn:
        ecg_batch = ecg_batch.to(device)
        clinical_batch = clinical_batch.to(device)
        outputs = cnn_model(ecg_batch, clinical_batch)
        train_preds_cnn.extend(outputs.cpu().numpy())
    
    for ecg_batch, clinical_batch, target_batch in val_loader_cnn:
        ecg_batch = ecg_batch.to(device)
        clinical_batch = clinical_batch.to(device)
        outputs = cnn_model(ecg_batch, clinical_batch)
        val_preds_cnn.extend(outputs.cpu().numpy())

y_pred_cnn_train = np.array(train_preds_cnn)
y_pred_cnn_val = np.array(val_preds_cnn)

models = {
    'XGBoost': {
        'train': y_pred_xgb_train,
        'val': y_pred_xgb_val,
        'test': y_pred_xgb_test
    },
    'MLP': {
        'train': y_pred_mlp_train,
        'val': y_pred_mlp_val,
        'test': y_pred_mlp_test
    },
    'CNN': {
        'train': y_pred_cnn_train,
        'val': y_pred_cnn_val,
        'test': y_pred_cnn_test
    }
}

# Evaluation function
def evaluate_model(y_true, y_pred, model_name, split_name):
    """Comprehensive evaluation metrics"""
    results = {}
    
    # Discrimination
    results['AUROC'] = roc_auc_score(y_true, y_pred)
    results['AUPRC'] = average_precision_score(y_true, y_pred)
    
    # Calibration
    results['Brier Score'] = brier_score_loss(y_true, y_pred)
    
    # Expected Calibration Error (ECE)
    # Use quantile strategy for imbalanced data, reduce bins to 5
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=5, strategy='quantile')
    ece = np.abs(prob_true - prob_pred).mean()
    results['ECE'] = ece
    
    # Clinical utility at 90% sensitivity threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    idx_90sens = np.argmax(tpr >= 0.9)
    threshold_90sens = thresholds[idx_90sens]
    
    y_pred_binary = (y_pred >= threshold_90sens).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    results['Sensitivity @90%'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    results['Specificity @90%'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    results['PPV @90%'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    results['NPV @90%'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return results

# Evaluate all models
print("\nEvaluating models...")
evaluation_results = []

for model_name, preds in models.items():
    for split in ['train', 'val', 'test']:
        if split == 'train':
            y_true = y_train
        elif split == 'val':
            y_true = y_val
        else:
            y_true = y_test
        
        y_pred = preds[split]
        metrics = evaluate_model(y_true, y_pred, model_name, split)
        
        evaluation_results.append({
            'Model': model_name,
            'Split': split.capitalize(),
            **metrics
        })

results_df = pd.DataFrame(evaluation_results)
print("\n", results_df.to_string(index=False))

# Save results
results_df.to_csv(OUTPUT_DIR / "model_evaluation_metrics.csv", index=False)

# ============================================================================
# Subgroup Analysis (Fairness Check)
# ============================================================================

print("\n" + "=" * 80)
print("Subgroup Analysis (Fairness Check)")
print("=" * 80)

test_df = df.loc[test_idx].copy()
test_df['predictions_xgb'] = y_pred_xgb_test
test_df['predictions_mlp'] = y_pred_mlp_test
test_df['predictions_cnn'] = y_pred_cnn_test

# Define subgroups
test_df['age_group'] = pd.cut(test_df['age'], bins=[0, 50, 70, 150], labels=['<50', '50-70', '>70'])

if 'diabetes' in test_df.columns:
    test_df['diabetes_status'] = test_df['diabetes'].map({0: 'No', 1: 'Yes'})
else:
    test_df['diabetes_status'] = 'Unknown'

if 'prior_mi' in test_df.columns:
    test_df['prior_mi_status'] = test_df['prior_mi'].map({0: 'No', 1: 'Yes'})
else:
    test_df['prior_mi_status'] = 'Unknown'

# Handle sex column (could be 'sex_M', 'sex', or 'Sex')
if 'sex_M' in test_df.columns:
    test_df['sex_group'] = test_df['sex_M'].map({0: 'Female', 1: 'Male'})
elif 'sex' in test_df.columns:
    if test_df['sex'].dtype == 'object':
        test_df['sex_group'] = test_df['sex']
    else:
        test_df['sex_group'] = test_df['sex'].map({0: 'Female', 1: 'Male'})
elif 'Sex' in test_df.columns:
    test_df['sex_group'] = test_df['Sex']
else:
    test_df['sex_group'] = 'Unknown'

subgroup_results = []

# Map model names to prediction column names
pred_col_mapping = {'XGBoost': 'predictions_xgb', 'MLP': 'predictions_mlp', 'CNN': 'predictions_cnn'}

for model_name in ['XGBoost', 'MLP', 'CNN']:
    pred_col = pred_col_mapping[model_name]
    
    for subgroup_col in ['age_group', 'sex_group', 'diabetes_status']:
        if subgroup_col not in test_df.columns or test_df[subgroup_col].isna().all():
            continue
        
        for subgroup_val in test_df[subgroup_col].dropna().unique():
            mask = test_df[subgroup_col] == subgroup_val
            if mask.sum() < 10:  # Skip small groups
                continue
            
            y_true_sub = test_df.loc[mask, 'target'].values
            y_pred_sub = test_df.loc[mask, pred_col].values
            
            if len(np.unique(y_true_sub)) < 2:  # Need both classes
                continue
            
            auc_sub = roc_auc_score(y_true_sub, y_pred_sub)
            
            subgroup_results.append({
                'Model': model_name,
                'Subgroup': subgroup_col,
                'Value': subgroup_val,
                'N': mask.sum(),
                'AUROC': auc_sub
            })

subgroup_df = pd.DataFrame(subgroup_results)
print("\n", subgroup_df.to_string(index=False))

# Check for disparate performance (>10% AUROC difference)
print("\nDisparity Check (>10% AUROC difference):")
for model_name in ['XGBoost', 'MLP', 'CNN']:
    for subgroup_col in ['age_group', 'sex_group', 'diabetes_status']:
        mask = (subgroup_df['Model'] == model_name) & (subgroup_df['Subgroup'] == subgroup_col)
        if mask.sum() < 2:
            continue
        
        aucs = subgroup_df.loc[mask, 'AUROC'].values
        if len(aucs) >= 2:
            max_diff = aucs.max() - aucs.min()
            status = "⚠ BIAS" if max_diff > 0.10 else "✓ OK"
            print(f"  {model_name} - {subgroup_col}: {max_diff:.4f} {status}")

subgroup_df.to_csv(OUTPUT_DIR / "subgroup_analysis.csv", index=False)

# ============================================================================
# Visualization
# ============================================================================

print("\n" + "=" * 80)
print("Generating Visualizations")
print("=" * 80)

# ROC Curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, model_name in enumerate(['XGBoost', 'MLP', 'CNN']):
    ax = axes[idx]
    
    if model_name == 'XGBoost':
        y_pred = y_pred_xgb_test
    elif model_name == 'MLP':
        y_pred = y_pred_mlp_test
    else:
        y_pred = y_pred_cnn_test
    
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'roc_curves.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved ROC curves to {PLOTS_DIR / 'roc_curves.png'}")

# Calibration Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, model_name in enumerate(['XGBoost', 'MLP', 'CNN']):
    ax = axes[idx]
    
    if model_name == 'XGBoost':
        y_pred = y_pred_xgb_test
    elif model_name == 'MLP':
        y_pred = y_pred_mlp_test
    else:
        y_pred = y_pred_cnn_test
    
    # Use quantile strategy for better bin distribution with imbalanced data
    # Reduce bins to 5 to ensure sufficient samples per bin (6.32% positive class)
    prob_true, prob_pred = calibration_curve(y_test, y_pred, n_bins=5, strategy='quantile')
    
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8, label=model_name)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(f'Calibration Plot - {model_name}', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'calibration_plots.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved calibration plots to {PLOTS_DIR / 'calibration_plots.png'}")

# Subgroup Performance Heatmap
pivot_data = subgroup_df.pivot_table(
    values='AUROC', 
    index=['Subgroup', 'Value'], 
    columns='Model'
)

fig, ax = plt.subplots(figsize=(8, 10))
sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=0.8, ax=ax)
ax.set_title('Subgroup AUROC Performance', fontsize=14)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'subgroup_performance.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved subgroup analysis to {PLOTS_DIR / 'subgroup_performance.png'}")

# ============================================================================
# Summary Report
# ============================================================================

print("\n" + "=" * 80)
print("✓ Phase H: Baseline Predictive Models - COMPLETE")
print("=" * 80)

summary = f"""
Phase H Summary
===============

Models Trained:
1. XGBoost (Tabular): {len(tabular_features)} features
   - Test AUC-ROC: {auc_test_xgb:.4f}
   - Best hyperparameters optimized via 5-fold grouped CV

2. MLP (VAE Latent): {len(latent_features)} features
   - Test AUC-ROC: {auc_test_mlp:.4f}
   - Early stopping at epoch with best validation AUC

3. CNN (Raw Waveform): Skipped (time-intensive, requires full waveform loading)

Data Split (Grouped by subject_id):
- Train: {len(train_idx)} records ({len(train_subjects)} subjects)
- Val:   {len(val_idx)} records ({len(val_subjects)} subjects)
- Test:  {len(test_idx)} records ({len(test_subjects)} subjects)

Best Model: {'XGBoost' if auc_test_xgb > auc_test_mlp else 'MLP'}
  - Test AUC-ROC: {max(auc_test_xgb, auc_test_mlp):.4f}

Artifacts Saved:
- models/baseline/xgboost_model.pkl
- models/baseline/mlp_model_best.pt
- models/baseline/model_evaluation_metrics.csv
- models/baseline/subgroup_analysis.csv
- models/baseline/split_assignments.parquet
- models/baseline/plots/*.png

Next Steps:
- Review baseline_models_report.md
- Use best model predictions as benchmark for causal models
- Proceed to Phase I: Propensity Score Modeling
"""

print(summary)

# Save summary
with open(OUTPUT_DIR / "phase_h_summary.txt", 'w') as f:
    f.write(summary)

print(f"\n✓ Summary saved to {OUTPUT_DIR / 'phase_h_summary.txt'}")
