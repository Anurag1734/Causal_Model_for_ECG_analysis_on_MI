# Model Improvement Recommendations

Based on the evaluation results, here are specific code improvements to address the identified issues:

## Priority 1: Address Age Bias (Critical)

### Issue
- 11.1% AUROC disparity between age groups (<50: 0.798 vs >70: 0.687)
- Elderly patients (highest risk group) have worst performance

### Recommended Solutions

#### 1. Age-Stratified Training (Immediate)
```python
# Add to phase_h_baseline_models.py after data loading

from sklearn.model_selection import StratifiedGroupKFold

# Create age-stratified groups for balanced sampling
df['age_strata'] = pd.cut(df['age'], bins=[0, 50, 70, 150], labels=['<50', '50-70', '>70'])
df['stratify_target'] = df['age_strata'].astype(str) + '_' + df['target'].astype(str)

# Use stratified split
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

for train_idx, test_idx in sgkf.split(df, df['stratify_target'], groups=df['subject_id']):
    # Ensures balanced age distribution across splits
    break
```

#### 2. Age-Weighted Loss Function
```python
# For XGBoost - add age-based sample weights
def compute_age_weights(ages, targets):
    """Higher weight for elderly patients to improve their performance"""
    weights = np.ones(len(ages))
    
    # Increase weight for elderly (>70) and MI positive
    mask_elderly_mi = (ages > 70) & (targets == 1)
    weights[mask_elderly_mi] *= 2.0
    
    # Standard class imbalance weight
    pos_weight = (targets == 0).sum() / (targets == 1).sum()
    weights[targets == 1] *= pos_weight
    
    return weights

# In XGBoost training:
age_weights = compute_age_weights(df.loc[train_idx, 'age'].values, y_train)

xgb_model.fit(
    X_train_tab, y_train,
    sample_weight=age_weights,  # Add this
    eval_set=[(X_val_tab, y_val)],
    verbose=False
)
```

#### 3. For MLP - Focal Loss with Age Reweighting
```python
import torch.nn.functional as F

class AgeFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, age_boost=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.age_boost = age_boost
    
    def forward(self, inputs, targets, ages):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Focal loss component
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # Age-based weighting (boost elderly patients)
        age_weight = torch.ones_like(ages)
        age_weight[ages > 70] *= self.age_boost
        
        return (focal_loss * age_weight).mean()

# Usage in training loop:
criterion = AgeFocalLoss(alpha=0.25, gamma=2.0, age_boost=2.0)

for epoch in range(100):
    for X_batch, y_batch in train_loader_lat:
        # Extract ages from batch (requires modification to dataset)
        ages = ages_batch.to(DEVICE)
        
        outputs = mlp_model(X_batch)
        loss = criterion(outputs, y_batch, ages)
        # ... rest of training
```

---

## Priority 2: Improve Precision / Reduce False Positives

### Issue
- PPV = 8.5% (only 1 in 12 positive predictions is correct)
- Specificity = 27.4% at 90% sensitivity

### Recommended Solutions

#### 1. Cost-Sensitive Learning
```python
# XGBoost with custom objective
def weighted_logloss(y_pred, dtrain):
    y_true = dtrain.get_label()
    
    # FP penalty: 5x worse than FN (clinical preference may vary)
    fp_weight = 5.0
    fn_weight = 1.0
    
    # Gradient
    grad = np.where(
        y_true == 1,
        -fn_weight * (y_true - y_pred) / (y_pred + 1e-8),
        fp_weight * (y_true - y_pred) / (1 - y_pred + 1e-8)
    )
    
    # Hessian
    hess = np.where(
        y_true == 1,
        fn_weight * y_pred * (1 - y_pred),
        fp_weight * y_pred * (1 - y_pred)
    )
    
    return grad, hess

# In XGBoost training:
xgb_model = xgb.XGBClassifier(
    objective=weighted_logloss,  # Custom objective
    # ... other params
)
```

#### 2. Ensemble with Stacking
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Base estimators
base_estimators = [
    ('xgb', xgb_model),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
    ('lgbm', LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42))
]

# Meta-learner with higher precision focus
meta_learner = LogisticRegression(
    C=0.1,  # Higher regularization
    class_weight='balanced',
    random_state=42
)

# Stacking ensemble
stacking_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=5,
    stack_method='predict_proba'
)

stacking_model.fit(X_train_tab, y_train)
y_pred_stack = stacking_model.predict_proba(X_test_tab)[:, 1]

# Expected improvement: +3-5% AUROC, +5-10% precision
```

#### 3. Threshold Optimization for Clinical Use
```python
from sklearn.metrics import precision_recall_curve

def optimize_threshold_f_beta(y_true, y_pred_proba, beta=2.0):
    """
    Optimize threshold for F-beta score
    beta > 1: favor recall (catch more MI cases)
    beta < 1: favor precision (reduce false alarms)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    f_beta = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall + 1e-8)
    
    best_idx = np.argmax(f_beta)
    best_threshold = thresholds[best_idx]
    
    return best_threshold, f_beta[best_idx]

# Find optimal threshold
optimal_threshold, optimal_f2 = optimize_threshold_f_beta(
    y_test, y_pred_xgb_test, beta=2.0
)

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"F2-score: {optimal_f2:.3f}")

# Use this threshold instead of 0.5 default
y_pred_binary = (y_pred_xgb_test >= optimal_threshold).astype(int)
```

---

## Priority 3: Enhance Feature Engineering

### Issue
- Moderate AUROC (0.702) suggests missed predictive patterns
- ECG fiducial features may miss morphological details

### Recommended Solutions

#### 1. Add Time-Domain Variability Features
```python
import neurokit2 as nk

def extract_hrv_features(ecg_signal, sampling_rate=500):
    """Extract heart rate variability features"""
    
    # Detect R-peaks
    peaks, info = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
    
    # HRV metrics
    hrv_time = nk.hrv_time(peaks, sampling_rate=sampling_rate)
    hrv_freq = nk.hrv_frequency(peaks, sampling_rate=sampling_rate)
    
    features = {
        'hrv_rmssd': hrv_time['HRV_RMSSD'].values[0],  # Root mean square of successive differences
        'hrv_sdnn': hrv_time['HRV_SDNN'].values[0],    # Standard deviation of NN intervals
        'hrv_lf_hf_ratio': hrv_freq['HRV_LFHF'].values[0],  # Autonomic balance
        'hrv_total_power': hrv_freq['HRV_TP'].values[0]
    }
    
    return features

# Add to ECG processing pipeline
for idx, row in df.iterrows():
    ecg_signal = load_ecg_waveform(row['file_path'], ECG_BASE_PATH)
    hrv_features = extract_hrv_features(ecg_signal[0])  # Lead I
    
    df.loc[idx, 'hrv_rmssd'] = hrv_features['hrv_rmssd']
    df.loc[idx, 'hrv_sdnn'] = hrv_features['hrv_sdnn']
    # ... etc
```

#### 2. ECG Morphology Features
```python
def extract_morphology_features(ecg_signal):
    """Extract waveform shape features"""
    
    features = {}
    
    for lead_idx, lead_name in enumerate(['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                                           'V1', 'V2', 'V3', 'V4', 'V5', 'V6']):
        signal = ecg_signal[lead_idx]
        
        # Statistical features
        features[f'{lead_name}_skewness'] = stats.skew(signal)
        features[f'{lead_name}_kurtosis'] = stats.kurtosis(signal)
        
        # Waveform complexity
        features[f'{lead_name}_zero_crossings'] = np.sum(np.diff(np.sign(signal)) != 0)
        
        # Spectral features
        fft = np.fft.fft(signal)
        features[f'{lead_name}_spectral_entropy'] = -np.sum(
            np.abs(fft)**2 * np.log(np.abs(fft)**2 + 1e-8)
        )
    
    return features
```

#### 3. Clinical Feature Interactions
```python
# Create interaction terms
df['troponin_x_age'] = df['troponin'] * df['age']
df['glucose_x_diabetes'] = df['glucose'] * df['diabetes']
df['hr_x_age'] = df['heart_rate'] * df['age']
df['troponin_x_ckd'] = df['troponin'] * df['ckd']

# Polynomial features for key biomarkers
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)

biomarkers = ['troponin', 'bnp', 'crp', 'age', 'heart_rate']
poly_features = poly.fit_transform(df[biomarkers])

poly_feature_names = poly.get_feature_names_out(biomarkers)
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

df = pd.concat([df, poly_df], axis=1)
```

---

## Priority 4: Address Class Imbalance

### Issue
- Only 6.3% positive class
- AUPRC remains low (0.184)

### Recommended Solutions

#### 1. SMOTE Oversampling (Synthetic Minority Oversampling)
```python
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# SMOTE + Random undersampling
over = SMOTE(sampling_strategy=0.3, random_state=42)  # 30% positive class
under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)  # 50% positive

# Pipeline
resampler = ImbPipeline([
    ('over', over),
    ('under', under)
])

X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_tab, y_train)

print(f"Original: {len(y_train)} samples, {y_train.sum()} positive ({100*y_train.mean():.1f}%)")
print(f"Resampled: {len(y_train_resampled)} samples, {y_train_resampled.sum()} positive ({100*y_train_resampled.mean():.1f}%)")

# Train on resampled data
xgb_model.fit(X_train_resampled, y_train_resampled)
```

#### 2. Focal Loss for Neural Networks
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Replace BCE loss in MLP training
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

#### 3. Class-Balanced Batch Sampling
```python
from torch.utils.data import WeightedRandomSampler

# Compute sample weights (inverse class frequency)
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
sample_weights = class_weights[y_train]

# Create weighted sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Use in DataLoader
train_loader_lat = DataLoader(
    train_dataset_lat,
    batch_size=64,
    sampler=sampler  # Replace shuffle=True
)
```

---

## Priority 5: Implement CNN for Raw Waveform Learning

### Issue
- Current models rely on hand-crafted features
- May miss complex morphological patterns

### Recommended Solution: ResNet-Style 1D CNN

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=15, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class ImprovedECGCNN(nn.Module):
    def __init__(self, num_clinical_features):
        super().__init__()
        
        # ECG pathway (12 leads, 5000 samples)
        self.ecg_pathway = nn.Sequential(
            # Initial convolution
            nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Residual blocks
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Clinical pathway
        self.clinical_pathway = nn.Sequential(
            nn.Linear(num_clinical_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
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
        
        # Fusion
        merged = torch.cat([ecg_features, clinical_features], dim=1)
        output = self.classifier(merged)
        
        return output.squeeze()

# Expected improvement: AUROC 0.75-0.82 (based on literature)
```

---

## Summary of Expected Improvements

| Improvement | Expected AUROC Gain | Expected Precision Gain | Reduces Age Bias? |
|-------------|---------------------|-------------------------|-------------------|
| Age-stratified training | +0.02-0.03 | +1-2% | ✓ Yes (primary) |
| Age-weighted loss | +0.03-0.05 | +2-3% | ✓ Yes (primary) |
| Cost-sensitive learning | +0.01-0.02 | +5-8% | ✗ No |
| Stacking ensemble | +0.03-0.05 | +3-5% | ~ Slight |
| HRV features | +0.01-0.02 | +1-2% | ✗ No |
| SMOTE oversampling | +0.02-0.03 | +2-4% | ✗ No |
| Focal loss | +0.01-0.02 | +2-3% | ✗ No |
| ResNet CNN | +0.05-0.10 | +5-10% | ~ Depends |

**Recommended Implementation Order:**
1. Age-weighted loss (addresses critical bias)
2. Cost-sensitive learning (improves clinical utility)
3. Enhanced features (HRV, interactions)
4. Stacking ensemble (robust performance boost)
5. ResNet CNN (long-term, highest potential)

---

## Implementation Checklist

- [ ] Implement age-stratified training
- [ ] Add age-weighted loss function
- [ ] Test cost-sensitive XGBoost
- [ ] Build stacking ensemble
- [ ] Extract HRV features
- [ ] Create clinical interaction terms
- [ ] Implement SMOTE oversampling
- [ ] Replace BCE with Focal Loss
- [ ] Build ResNet CNN architecture
- [ ] Re-run fairness analysis
- [ ] Update baseline_models_report.md
- [ ] Validate on held-out test set

**Estimated Development Time:** 2-3 weeks for full implementation
**Expected Final AUROC:** 0.75-0.80 (vs current 0.702)
