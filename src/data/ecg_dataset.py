"""
PyTorch Dataset for loading raw ECG signals from WFDB format.

Handles:
- Loading 12-lead ECGs from MIMIC-IV-ECG-1.0 WFDB files
- Per-lead normalization (zero mean, unit variance)
- Quality control filtering
- Train/val/test splits with stratification
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import wfdb
from typing import Tuple, Optional, List
from pathlib import Path
from tqdm import tqdm


class ECGDataset(Dataset):
    """
    Dataset for loading raw 12-lead ECG signals.
    
    Parameters
    ----------
    metadata_df : pd.DataFrame
        DataFrame with columns: [subject_id, study_id, file_path, Label, ...]
    base_path : str
        Base directory containing WFDB files (e.g., "data/raw/MIMIC-IV-ECG-1.0/files")
    normalize : bool, default=True
        Whether to normalize each lead to zero mean, unit variance
    expected_length : int, default=5000
        Expected number of samples (at 500 Hz, 5000 = 10 seconds)
    """
    
    def __init__(
        self, 
        metadata_df: pd.DataFrame,
        base_path: str,
        normalize: bool = True,
        expected_length: int = 5000
    ):
        self.metadata = metadata_df.reset_index(drop=True)
        self.base_path = Path(base_path)
        self.normalize = normalize
        self.expected_length = expected_length
        
        # Extract file paths
        if 'file_path' in self.metadata.columns:
            self.file_paths = self.metadata['file_path'].tolist()
        else:
            raise ValueError("metadata_df must contain 'file_path' column")
        
        # Extract labels if available
        if 'Label' in self.metadata.columns:
            self.labels = self.metadata['Label'].tolist()
        else:
            self.labels = [None] * len(self.metadata)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Load and return ECG signal with metadata.
        
        Returns
        -------
        signal : torch.Tensor
            ECG signal, shape (12, 5000)
        metadata : dict
            Dictionary with keys: subject_id, study_id, file_path, label
        """
        # Get file path
        file_path = self.file_paths[idx]
        
        # Load WFDB record
        try:
            signal = self._load_wfdb_signal(file_path)
            
            # Validate signal doesn't contain NaN/Inf
            if np.isnan(signal).any() or np.isinf(signal).any():
                raise ValueError(f"Signal contains NaN/Inf values")
                
        except Exception as e:
            # Skip this sample by returning None - DataLoader will handle it
            # (We'll add collate_fn to filter out None values)
            return None
        
        # Normalize per-lead if requested
        if self.normalize:
            signal = self._normalize_signal(signal)
            
            # Double-check normalization didn't create NaN
            if np.isnan(signal).any() or np.isinf(signal).any():
                return None
        
        # Convert to tensor
        signal = torch.from_numpy(signal).float()
        
        # Prepare metadata
        metadata = {
            'idx': idx,
            'subject_id': self.metadata.iloc[idx]['subject_id'],
            'study_id': self.metadata.iloc[idx]['study_id'],
            'file_path': file_path,
            'label': self.labels[idx]
        }
        
        return signal, metadata
    
    def _load_wfdb_signal(self, file_path: str) -> np.ndarray:
        """
        Load WFDB record and extract 12-lead signal.
        
        Parameters
        ----------
        file_path : str
            Relative path to WFDB file (e.g., "p10/p10020306/s41256771/41256771")
        
        Returns
        -------
        signal : np.ndarray
            12-lead ECG signal, shape (12, length)
        """
        # Construct full path (remove extension if present)
        full_path = self.base_path / file_path
        if full_path.suffix:
            full_path = full_path.with_suffix('')
        
        # Read WFDB record
        record = wfdb.rdrecord(str(full_path))
        
        # Extract signal: (n_samples, n_leads) → (n_leads, n_samples)
        signal = record.p_signal.T  # Shape: (12, length)
        
        # Ensure exactly 12 leads
        if signal.shape[0] != 12:
            raise ValueError(f"Expected 12 leads, got {signal.shape[0]}")
        
        # Pad or truncate to expected length
        if signal.shape[1] < self.expected_length:
            # Pad with zeros
            pad_width = ((0, 0), (0, self.expected_length - signal.shape[1]))
            signal = np.pad(signal, pad_width, mode='constant', constant_values=0)
        elif signal.shape[1] > self.expected_length:
            # Truncate
            signal = signal[:, :self.expected_length]
        
        return signal.astype(np.float32)
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Normalize each lead to zero mean, unit variance.
        
        Parameters
        ----------
        signal : np.ndarray
            ECG signal, shape (12, length)
        
        Returns
        -------
        normalized_signal : np.ndarray
            Normalized signal, shape (12, length)
        """
        normalized = np.zeros_like(signal)
        for i in range(12):
            lead = signal[i, :]
            mean = np.mean(lead)
            std = np.std(lead)
            if std > 0:
                normalized[i, :] = (lead - mean) / std
            else:
                normalized[i, :] = lead - mean  # Avoid division by zero
        return normalized


def create_train_val_test_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify_col: str = 'Label',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    train_ratio : float
        Proportion for training set
    val_ratio : float
        Proportion for validation set
    test_ratio : float
        Proportion for test set
    stratify_col : str
        Column to stratify by (e.g., 'Label')
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    train_df : pd.DataFrame
        Training set
    val_df : pd.DataFrame
        Validation set
    test_df : pd.DataFrame
        Test set
    """
    from sklearn.model_selection import train_test_split
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df[stratify_col],
        random_state=random_state
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        stratify=temp_df[stratify_col],
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def collate_fn_skip_none(batch):
    """
    Custom collate function that skips None samples (corrupted files).
    
    Parameters
    ----------
    batch : list
        List of (signal, metadata) tuples or None values
    
    Returns
    -------
    signals : torch.Tensor
        Batched signals, shape (batch_size, 12, 5000)
    metadata_list : list
        List of metadata dicts
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    # If all samples failed, return empty batch
    if len(batch) == 0:
        return None, None
    
    # Separate signals and metadata
    signals = torch.stack([item[0] for item in batch])
    metadata_list = [item[1] for item in batch]
    
    return signals, metadata_list


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    normalize: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train/val/test sets.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training metadata
    val_df : pd.DataFrame
        Validation metadata
    test_df : pd.DataFrame
        Test metadata
    base_path : str
        Base directory containing WFDB files
    batch_size : int
        Batch size for DataLoader
    num_workers : int
        Number of worker processes (0 = main process only)
    normalize : bool
        Whether to normalize signals
    
    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    test_loader : DataLoader
    """
    # Create datasets
    train_dataset = ECGDataset(train_df, base_path, normalize=normalize)
    val_dataset = ECGDataset(val_df, base_path, normalize=normalize)
    test_dataset = ECGDataset(test_df, base_path, normalize=normalize)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_skip_none
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_skip_none
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_skip_none
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("=" * 80)
    print("ECG Dataset Test")
    print("=" * 80)
    
    # Load sample metadata
    metadata_path = "data/processed/ecg_features_with_demographics.parquet"
    if os.path.exists(metadata_path):
        df = pd.read_parquet(metadata_path)
        print(f"\n✓ Loaded {len(df)} records from {metadata_path}")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Test dataset
        base_path = "data/raw/MIMIC-IV-ECG-1.0/files"
        dataset = ECGDataset(df.head(10), base_path, normalize=True)
        print(f"\n✓ Created dataset with {len(dataset)} samples")
        
        # Test loading one sample
        signal, metadata = dataset[0]
        print(f"\n✓ Sample signal shape: {tuple(signal.shape)}")
        print(f"✓ Signal dtype: {signal.dtype}")
        print(f"✓ Signal range: [{signal.min():.4f}, {signal.max():.4f}]")
        print(f"✓ Metadata keys: {list(metadata.keys())}")
        
        print("\n" + "=" * 80)
        print("✓ Dataset test passed!")
        print("=" * 80)
    else:
        print(f"\n⚠ Metadata file not found: {metadata_path}")
        print("  Please run this from the project root directory.")
