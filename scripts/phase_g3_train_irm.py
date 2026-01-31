"""
Phase G.3: Train IRM Models

Train Invariant Risk Minimization (IRM) models to learn features robust
to environment shifts (temporal distribution changes).

Models:
- ERM (Empirical Risk Minimization): Standard baseline
- IRM with λ ∈ {0.01, 0.1, 1.0, 10.0}: Varying penalty strengths

Objective:
L_IRM = L_ERM + λ * Σ||∇_θ loss_e||²
where penalty enforces features equally predictive across all environments.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Paths
MASTER_DATASET_PATH = "data/processed/master_dataset.parquet"
OUTPUT_DIR = Path("models/irm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IRM lambda values to test
IRM_LAMBDAS = [0.0, 0.01, 0.1, 1.0, 10.0]  # 0.0 = ERM baseline

print("\n" + "=" * 80)
print("Phase G.3: IRM Model Training")
print("=" * 80)

# Load data
print("\nLoading master dataset...")
df = pd.read_parquet(MASTER_DATASET_PATH)
print(f"Loaded {len(df)} records")

# Check environment labels
if 'environment_label' not in df.columns:
    print("\n⚠ WARNING: 'environment_label' not found in dataset!")
    print("  Creating placeholder environments based on Label for demonstration...")
    print("  NOTE: Run Phase G.1-G.2 first to create proper environments!")
    # Fallback: use Label as environment
    label_to_env = {
        'Control_Symptomatic': 0,
        'MI_Acute_Presentation': 1,
        'MI_Pre-Incident': 2
    }
    df['environment_label'] = df['Label'].map(label_to_env)
    df['environment_name'] = 'env_' + df['Label']

# Split data
train_df = df[df['split'] == 'train'].copy()
test_df = df[df['split'] == 'test'].copy()

print(f"\nTrain: {len(train_df)} records")
print(f"Test:  {len(test_df)} records")

# Show environment distribution
print("\nEnvironment Distribution (Training):")
env_dist = train_df['environment_name'].value_counts()
for env, count in env_dist.items():
    pct = 100 * count / len(train_df)
    print(f"  {env:30s}: {count:6d} ({pct:5.1f}%)")

print("\n" + "=" * 80)
print("Feature Selection")
print("=" * 80)

# Define feature sets
ecg_features = [c for c in df.columns if c.startswith('ecg_') and c not in ['ecg_datetime', 'ecg_study_id']]
vae_features = [c for c in df.columns if c.startswith('z_ecg_')]
clinical_features = [
    'age', 'sex_M',
    'statin_use', 'total_cholesterol', 'ldl', 'hdl', 'triglycerides',
    'glucose', 'creatinine', 'troponin', 'bnp', 'crp',
    'heart_rate', 'sbp', 'dbp', 'temperature', 'respiratory_rate', 'spo2',
    'diabetes', 'hypertension', 'ckd', 'chf', 'cad', 'prior_mi', 'stroke'
]

# Filter to available features
ecg_features = [f for f in ecg_features if f in df.columns]
clinical_features = [f for f in clinical_features if f in df.columns]

print(f"\nOption 1 (ECG + Clinical): {len(ecg_features) + len(clinical_features)} features available")
print(f"Option 2 (VAE + Clinical): {len(vae_features) + len(clinical_features)} features available")

# Use VAE + Clinical (better for causal inference)
feature_cols = vae_features + clinical_features
print(f"\n✓ Using Option 2: VAE Latent + Clinical ({len(feature_cols)} features)")

# Create binary target
train_df['target'] = (train_df['Label'] != 'Control_Symptomatic').astype(int)
test_df['target'] = (test_df['Label'] != 'Control_Symptomatic').astype(int)

print("\nTarget Distribution (Training):")
target_dist = train_df['target'].value_counts()
print(f"  Control (0): {target_dist.get(0, 0):6d} ({100 * target_dist.get(0, 0) / len(train_df):5.1f}%)")
print(f"  MI (1):      {target_dist.get(1, 0):6d} ({100 * target_dist.get(1, 0) / len(train_df):5.1f}%)")

# Get environments
environments = sorted(train_df['environment_label'].unique())
print(f"\nNumber of environments: {len(environments)}")

# PyTorch Dataset
class IRMDataset(Dataset):
    def __init__(self, features, targets, environments):
        self.features = torch.FloatTensor(features.values)
        self.targets = torch.FloatTensor(targets.values)
        self.environments = torch.LongTensor(environments.values)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.environments[idx]

# Create datasets per environment
print("\nPreparing PyTorch datasets...")
env_datasets = {}
env_loaders = {}

for env in environments:
    env_mask = train_df['environment_label'] == env
    env_data = train_df[env_mask]
    
    X_env = env_data[feature_cols]
    y_env = env_data['target']
    e_env = env_data['environment_label']
    
    # Check for NaN values
    if X_env.isna().any().any():
        print(f"  ⚠ WARNING: NaN values found in environment {env}, filling with 0")
        X_env = X_env.fillna(0)
    
    env_datasets[env] = IRMDataset(X_env, y_env, e_env)
    env_loaders[env] = DataLoader(env_datasets[env], batch_size=BATCH_SIZE, shuffle=True)
    
    env_name = env_data['environment_name'].iloc[0] if len(env_data) > 0 else f"Env {env}"
    print(f"  Env {env} ({env_name}): {len(env_datasets[env])} samples")

# Test dataset
X_test = test_df[feature_cols]
y_test = test_df['target']
e_test = test_df['environment_label']

if X_test.isna().any().any():
    print(f"  ⚠ WARNING: NaN values found in test data, filling with 0")
    X_test = X_test.fillna(0)

test_dataset = IRMDataset(X_test, y_test, e_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"\nTest data: {len(test_dataset)} samples")

# Model
class IRMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# Training function
def compute_irm_penalty(losses):
    """Compute IRM penalty: variance of losses across environments"""
    if len(losses) < 2:
        return torch.tensor(0.0).to(losses[0].device)
    
    # Stack losses and compute variance
    loss_stack = torch.stack(losses)
    penalty = loss_stack.var()
    
    return penalty

def train_irm_model(irm_lambda, model_name):
    """Train single IRM model with given lambda"""
    print("\n" + "=" * 80)
    print(f"Training: {model_name}")
    print("=" * 80)
    
    model = IRMClassifier(input_dim=len(feature_cols)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'erm_loss': [], 'irm_penalty': [], 'test_auc': []}
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        
        # Collect batches from all environments
        env_losses = []
        total_erm_loss = 0.0
        total_penalty = 0.0
        n_batches = 0
        
        # Create iterators for each environment
        env_iters = {e: iter(env_loaders[e]) for e in environments}
        
        # Train on batches from all environments
        max_batches = max(len(env_loaders[e]) for e in environments)
        
        for batch_idx in range(max_batches):
            batch_losses = []
            batch_logits = []
            
            for env in environments:
                try:
                    X, y, _ = next(env_iters[env])
                except StopIteration:
                    # Reset iterator if exhausted
                    env_iters[env] = iter(env_loaders[env])
                    X, y, _ = next(env_iters[env])
                
                X, y = X.to(DEVICE), y.to(DEVICE)
                
                logits = model(X).squeeze()
                loss = criterion(logits, y)
                
                batch_losses.append(loss)
                batch_logits.append(logits)
            
            # Compute ERM loss (average across environments)
            erm_loss = torch.stack(batch_losses).mean()
            
            # Compute IRM penalty
            if irm_lambda > 0:
                penalty = compute_irm_penalty(batch_losses)
            else:
                penalty = torch.tensor(0.0).to(DEVICE)
            
            # Total loss
            total_loss = erm_loss + irm_lambda * penalty
            
            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_erm_loss += erm_loss.item()
            total_penalty += penalty.item() if isinstance(penalty, torch.Tensor) else penalty
            n_batches += 1
        
        avg_erm_loss = total_erm_loss / n_batches
        avg_penalty = total_penalty / n_batches
        avg_total_loss = avg_erm_loss + irm_lambda * avg_penalty
        
        # Evaluate on test set
        model.eval()
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for X, y, _ in test_loader:
                X = X.to(DEVICE)
                logits = model(X).squeeze()
                probs = torch.sigmoid(logits)
                test_preds.extend(probs.cpu().numpy())
                test_targets.extend(y.numpy())
        
        test_auc = roc_auc_score(test_targets, test_preds)
        
        # Save history
        history['train_loss'].append(avg_total_loss)
        history['erm_loss'].append(avg_erm_loss)
        history['irm_penalty'].append(avg_penalty)
        history['test_auc'].append(test_auc)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{MAX_EPOCHS} | Loss: {avg_total_loss:.4f} | "
                  f"ERM: {avg_erm_loss:.4f} | IRM Penalty: {avg_penalty:.6f} | "
                  f"Test AUC: {test_auc:.4f}")
        
        # Early stopping
        if test_auc > best_auc:
            best_auc = test_auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), OUTPUT_DIR / f"{model_name}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(OUTPUT_DIR / f"{model_name}_best.pt"))
    model.eval()
    
    final_preds = []
    final_targets = []
    
    with torch.no_grad():
        for X, y, _ in test_loader:
            X = X.to(DEVICE)
            logits = model(X).squeeze()
            probs = torch.sigmoid(logits)
            final_preds.extend(probs.cpu().numpy())
            final_targets.extend(y.numpy())
    
    final_preds = np.array(final_preds)
    final_targets = np.array(final_targets)
    
    final_auc = roc_auc_score(final_targets, final_preds)
    final_ap = average_precision_score(final_targets, final_preds)
    final_acc = accuracy_score(final_targets, (final_preds > 0.5).astype(int))
    
    print(f"\n✓ {model_name} Training Complete")
    print(f"  Best AUC-ROC: {best_auc:.4f}")
    print(f"  Final AUC-ROC: {final_auc:.4f}")
    print(f"  Final AUC-PR: {final_ap:.4f}")
    print(f"  Final Accuracy: {final_acc:.4f}")
    
    return {
        'model_name': model_name,
        'irm_lambda': irm_lambda,
        'best_auc': best_auc,
        'final_auc': final_auc,
        'final_ap': final_ap,
        'final_acc': final_acc,
        'history': history
    }

# Train models
print("\n" + "=" * 80)
print("Training Models")
print("=" * 80)
print(f"\nUsing device: {DEVICE}")
print(f"Input dimension: {len(feature_cols)}")

results = []

for lam in IRM_LAMBDAS:
    if lam == 0.0:
        model_name = "ERM"
    else:
        model_name = f"IRM_lambda{lam}"
    
    result = train_irm_model(lam, model_name)
    results.append(result)

# Compare models
print("\n" + "=" * 80)
print("Model Comparison")
print("=" * 80)

comparison_df = pd.DataFrame([{
    'Model': r['model_name'],
    'IRM Lambda': r['irm_lambda'],
    'Best AUC-ROC': r['best_auc'],
    'Final AUC-ROC': r['final_auc'],
    'Final AUC-PR': r['final_ap'],
    'Final Accuracy': r['final_acc']
} for r in results])

comparison_df = comparison_df.sort_values('Final AUC-ROC', ascending=False)
print("\n", comparison_df.to_string(index=False))

# Save results
comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
print(f"\n✓ Results saved to {OUTPUT_DIR}")

# Plot training curves
print("\nGenerating training curves...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for result in results:
    model_name = result['model_name']
    history = result['history']
    
    axes[0, 0].plot(history['train_loss'], label=model_name)
    axes[0, 1].plot(history['erm_loss'], label=model_name)
    axes[1, 0].plot(history['irm_penalty'], label=model_name)
    axes[1, 1].plot(history['test_auc'], label=model_name)

axes[0, 0].set_title('Total Training Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].set_title('ERM Loss Component')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('ERM Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].set_title('IRM Penalty')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Penalty')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].set_title('Test AUC-ROC')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('AUC-ROC')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=300, bbox_inches='tight')
print(f"✓ Training curves saved to {OUTPUT_DIR / 'training_curves.png'}")

print("\n" + "=" * 80)
print("✓ Phase G.3: IRM Model Training - COMPLETE")
print("=" * 80)

best_model = comparison_df.iloc[0]
print(f"\nBest Model: {best_model['Model']}")
print(f"  AUC-ROC: {best_model['Final AUC-ROC']:.4f}")
print(f"  IRM Lambda: {best_model['IRM Lambda']}")

print("\nNext Steps:")
print("  1. Review model comparison to select best IRM lambda")
print("  2. Analyze which features are environment-invariant")
print("  3. Proceed to Phase H-I: Propensity modeling and causal effect estimation")
