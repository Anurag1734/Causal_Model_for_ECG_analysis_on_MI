"""
Pre-flight check for Phase D.3-D.5 (VAE training pipeline).

This script validates:
1. All required files are present
2. Python environment has all dependencies
3. GPU is available and sufficient
4. Data files are readable and have expected structure
5. Model architecture can be instantiated
6. Dataset loading works correctly

Run this before starting VAE training to catch issues early.
"""

import sys
import os
from pathlib import Path
import importlib.util

def check_import(package_name, display_name=None):
    """Check if a package can be imported."""
    if display_name is None:
        display_name = package_name
    
    try:
        importlib.import_module(package_name)
        print(f"  ✓ {display_name}")
        return True
    except ImportError:
        print(f"  ✗ {display_name} - NOT INSTALLED")
        return False


def main():
    print("\n" + "=" * 80)
    print("Phase D.3-D.5 Pre-Flight Check")
    print("=" * 80)
    
    all_checks_passed = True
    
    # ========== Check 1: Python Version ==========
    print("\n[1/9] Checking Python version...")
    import sys
    py_version = sys.version_info
    print(f"  ✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print(f"  ⚠ Warning: Python 3.8+ recommended, you have {py_version.major}.{py_version.minor}")
    
    # ========== Check 2: Dependencies ==========
    print("\n[2/9] Checking Python dependencies...")
    required_packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('torch', 'PyTorch'),
        ('wfdb', 'WFDB'),
        ('neurokit2', 'NeuroKit2'),
        ('tqdm', 'tqdm'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'scikit-learn'),
    ]
    
    for package, display_name in required_packages:
        if not check_import(package, display_name):
            all_checks_passed = False
    
    # ========== Check 3: GPU Availability ==========
    print("\n[3/9] Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ VRAM: {vram_gb:.2f} GB")
            if vram_gb < 4:
                print(f"  ⚠ Warning: Less than 4 GB VRAM, may need to reduce batch size")
        else:
            print(f"  ✗ CUDA not available - will use CPU (VERY SLOW)")
            all_checks_passed = False
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}")
        all_checks_passed = False
    
    # ========== Check 4: Required Files ==========
    print("\n[4/9] Checking required files...")
    required_files = [
        'data/processed/ecg_features_with_demographics.parquet',
        'src/models/vae_conv1d.py',
        'src/models/train_vae.py',
        'src/data/ecg_dataset.py',
        'scripts/extract_latent_embeddings.py',
        'scripts/validate_latent_interpretability.py',
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - NOT FOUND")
            all_checks_passed = False
    
    # ========== Check 5: Data Directory ==========
    print("\n[5/9] Checking data directory...")
    data_dir = Path('data/raw/MIMIC-IV-ECG-1.0/files')
    if data_dir.exists():
        print(f"  ✓ {data_dir}")
        # Check for sample patient directory
        patient_dirs = list(data_dir.glob('p*'))
        if len(patient_dirs) > 0:
            print(f"  ✓ Found {len(patient_dirs)} patient directories")
        else:
            print(f"  ⚠ Warning: No patient directories found in {data_dir}")
    else:
        print(f"  ✗ {data_dir} - NOT FOUND")
        print(f"  → Please ensure MIMIC-IV-ECG data is extracted to this location")
        all_checks_passed = False
    
    # ========== Check 6: Metadata File ==========
    print("\n[6/9] Checking metadata file...")
    try:
        import pandas as pd
        metadata_path = 'data/processed/ecg_features_with_demographics.parquet'
        if os.path.exists(metadata_path):
            df = pd.read_parquet(metadata_path)
            print(f"  ✓ Loaded {len(df)} records")
            
            # Check required columns
            required_cols = ['subject_id', 'study_id', 'file_path', 'Label']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"  ✗ Missing columns: {missing_cols}")
                all_checks_passed = False
            else:
                print(f"  ✓ All required columns present")
            
            # Check label distribution
            print(f"  ✓ Label distribution:")
            for label, count in df['Label'].value_counts().items():
                print(f"      {label}: {count}")
            
            # Check for training labels
            train_labels = ['Control_Symptomatic', 'MI_Pre-Incident']
            n_train = len(df[df['Label'].isin(train_labels)])
            print(f"  ✓ Training set size (Control + MI_Pre): {n_train}")
            
            if n_train < 100:
                print(f"  ⚠ Warning: Very small training set ({n_train})")
        else:
            print(f"  ✗ {metadata_path} - NOT FOUND")
            all_checks_passed = False
    except Exception as e:
        print(f"  ✗ Error loading metadata: {e}")
        all_checks_passed = False
    
    # ========== Check 7: Model Architecture ==========
    print("\n[7/9] Checking model architecture...")
    try:
        sys.path.insert(0, 'src')
        from models.vae_conv1d import Conv1DVAE, count_parameters
        
        model = Conv1DVAE(z_dim=64, beta=4.0)
        n_params = count_parameters(model)
        print(f"  ✓ Model instantiated successfully")
        print(f"  ✓ Total parameters: {n_params:,}")
        
        # Test forward pass
        import torch
        x = torch.randn(2, 12, 5000)
        x_recon, mu, logvar = model(x)
        
        if x_recon.shape == (2, 12, 5000):
            print(f"  ✓ Forward pass successful")
        else:
            print(f"  ✗ Unexpected output shape: {x_recon.shape}")
            all_checks_passed = False
    except Exception as e:
        print(f"  ✗ Error testing model: {e}")
        all_checks_passed = False
    
    # ========== Check 8: Dataset Loading ==========
    print("\n[8/9] Checking dataset loading...")
    try:
        from data.ecg_dataset import ECGDataset
        import pandas as pd
        
        # Load small sample
        df = pd.read_parquet('data/processed/ecg_features_with_demographics.parquet')
        df_sample = df.head(2)
        
        dataset = ECGDataset(
            df_sample,
            base_path='data/raw/MIMIC-IV-ECG-1.0/files',
            normalize=True
        )
        
        print(f"  ✓ Dataset created with {len(dataset)} samples")
        
        # Try loading one sample
        try:
            signal, metadata = dataset[0]
            print(f"  ✓ Sample loaded successfully")
            print(f"      Signal shape: {signal.shape}")
            print(f"      Signal range: [{signal.min():.4f}, {signal.max():.4f}]")
            
            if signal.shape != (12, 5000):
                print(f"  ⚠ Warning: Unexpected signal shape: {signal.shape}")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load sample (file may not exist): {e}")
            print(f"      This is OK if you haven't extracted all ECGs yet")
    except Exception as e:
        print(f"  ✗ Error testing dataset: {e}")
        all_checks_passed = False
    
    # ========== Check 9: Disk Space ==========
    print("\n[9/9] Checking disk space...")
    try:
        import shutil
        total, used, free = shutil.disk_usage(os.getcwd())
        free_gb = free / (2**30)
        print(f"  ✓ Free disk space: {free_gb:.2f} GB")
        
        if free_gb < 10:
            print(f"  ⚠ Warning: Less than 10 GB free, ensure enough space for checkpoints")
    except Exception as e:
        print(f"  ⚠ Could not check disk space: {e}")
    
    # ========== Summary ==========
    print("\n" + "=" * 80)
    if all_checks_passed:
        print("✓ All checks passed! Ready to start VAE training.")
        print("\nNext step:")
        print("  python src/models/train_vae.py --batch_size 32 --epochs 100")
    else:
        print("✗ Some checks failed. Please fix the issues above before training.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install torch torchvision wfdb neurokit2")
        print("  - Ensure MIMIC-IV-ECG data is in data/raw/MIMIC-IV-ECG-1.0/files/")
        print("  - Run from project root directory")
    print("=" * 80 + "\n")
    
    return all_checks_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
