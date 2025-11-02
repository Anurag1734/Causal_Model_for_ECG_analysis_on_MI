"""
Extract latent embeddings from trained VAE for all ECGs.

This script:
1. Loads the trained VAE encoder
2. Processes ALL ECGs from cohort_master (including MI_Acute_Presentation)
3. Extracts latent mean vectors (μ, not sampled)
4. Saves to ecg_z_embeddings.parquet with columns: [record_id, z_ecg_1, ..., z_ecg_64]

Usage:
    python extract_latent_embeddings.py --checkpoint models/checkpoints/.../best_model.pt
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.vae_conv1d import Conv1DVAE
from data.ecg_dataset import ECGDataset


def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    z_dim: int
) -> tuple:
    """
    Extract latent mean embeddings for all ECGs.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained VAE model
    dataloader : DataLoader
        DataLoader with ECG data
    device : torch.device
        Device to run on
    z_dim : int
        Latent dimension size
    
    Returns
    -------
    embeddings : np.ndarray
        Latent embeddings, shape (n_samples, z_dim)
    subject_ids : list
        Subject IDs for each embedding
    study_ids : list
        Study IDs for each embedding
    """
    model.eval()
    
    all_embeddings = []
    all_subject_ids = []
    all_study_ids = []
    
    with torch.no_grad():
        for signals, metadata in tqdm(dataloader, desc="Extracting embeddings"):
            # Move to device
            signals = signals.to(device)
            
            # Extract latent mean (no sampling)
            mu = model.get_latent_embeddings(signals)
            
            # Move to CPU and store
            all_embeddings.append(mu.cpu().numpy())
            all_subject_ids.extend(metadata['subject_id'].tolist())
            all_study_ids.extend(metadata['study_id'].tolist())
    
    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)
    
    return embeddings, all_subject_ids, all_study_ids


def main(args):
    """Main extraction function."""
    print("\n" + "=" * 80)
    print("Latent Embedding Extraction")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Load checkpoint
    print(f"\n✓ Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Extract config from checkpoint if available
    if 'history' in checkpoint:
        print(f"✓ Checkpoint from epoch {checkpoint['epoch'] + 1}")
        print(f"✓ Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    # Initialize model
    print(f"\n✓ Initializing Conv1D β-VAE (z_dim={args.z_dim})")
    model = Conv1DVAE(z_dim=args.z_dim, beta=args.beta).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded successfully")
    
    # Load metadata for ALL ECGs (including MI_Acute_Presentation)
    print(f"\n✓ Loading metadata from {args.metadata_path}")
    df = pd.read_parquet(args.metadata_path)
    print(f"✓ Loaded {len(df)} records")
    print(f"✓ Label distribution:\n{df['Label'].value_counts()}")
    
    # Create dataset (no train/val/test split - process all)
    print(f"\n✓ Creating dataset for all ECGs")
    dataset = ECGDataset(df, args.base_path, normalize=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"✓ Total batches: {len(dataloader)}")
    
    # Extract embeddings
    print(f"\n✓ Extracting latent embeddings...")
    embeddings, subject_ids, study_ids = extract_embeddings(
        model, dataloader, device, args.z_dim
    )
    print(f"✓ Extracted {len(embeddings)} embeddings")
    print(f"✓ Embedding shape: {embeddings.shape}")
    
    # Create DataFrame
    print(f"\n✓ Creating DataFrame...")
    
    # Create column names: z_ecg_1, z_ecg_2, ..., z_ecg_64
    z_columns = [f'z_ecg_{i+1}' for i in range(args.z_dim)]
    
    # Create DataFrame
    df_embeddings = pd.DataFrame(embeddings, columns=z_columns)
    df_embeddings.insert(0, 'subject_id', subject_ids)
    df_embeddings.insert(1, 'study_id', study_ids)
    
    print(f"✓ DataFrame shape: {df_embeddings.shape}")
    print(f"✓ Columns: {list(df_embeddings.columns[:5])} ... {list(df_embeddings.columns[-3:])}")
    
    # Save to parquet
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Saving embeddings to {output_path}")
    df_embeddings.to_parquet(output_path, index=False)
    
    # Statistics
    print(f"\n✓ Embedding statistics:")
    print(f"  Shape: {df_embeddings.shape}")
    print(f"  Memory: {df_embeddings.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"  Mean: {embeddings.mean():.6f}")
    print(f"  Std: {embeddings.std():.6f}")
    print(f"  Min: {embeddings.min():.6f}")
    print(f"  Max: {embeddings.max():.6f}")
    
    # Save summary statistics
    summary_path = output_path.parent / 'embedding_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Latent Embedding Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"z_dim: {args.z_dim}\n")
        f.write(f"beta: {args.beta}\n")
        f.write(f"Total embeddings: {len(embeddings)}\n")
        f.write(f"Embedding shape: {embeddings.shape}\n\n")
        f.write(f"Statistics:\n")
        f.write(f"  Mean: {embeddings.mean():.6f}\n")
        f.write(f"  Std: {embeddings.std():.6f}\n")
        f.write(f"  Min: {embeddings.min():.6f}\n")
        f.write(f"  Max: {embeddings.max():.6f}\n\n")
        f.write(f"Per-dimension statistics:\n")
        for i in range(args.z_dim):
            dim_data = embeddings[:, i]
            f.write(f"  z_ecg_{i+1}: mean={dim_data.mean():.6f}, std={dim_data.std():.6f}, "
                   f"min={dim_data.min():.6f}, max={dim_data.max():.6f}\n")
    
    print(f"\n✓ Summary saved to {summary_path}")
    
    print("\n" + "=" * 80)
    print("✓ Embedding extraction completed!")
    print(f"✓ Output: {output_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract latent embeddings from trained VAE")
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--z_dim', type=int, default=64,
                       help='Latent dimension size (must match training)')
    parser.add_argument('--beta', type=float, default=4.0,
                       help='β-VAE parameter (must match training)')
    
    # Data
    parser.add_argument('--metadata_path', type=str, default='data/processed/ecg_features_with_demographics.parquet',
                       help='Path to metadata parquet file')
    parser.add_argument('--base_path', type=str, default='data/raw/MIMIC-IV-ECG-1.0/files',
                       help='Base path to WFDB files')
    
    # System
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of DataLoader workers')
    
    # Output
    parser.add_argument('--output_path', type=str, default='data/processed/ecg_z_embeddings.parquet',
                       help='Output path for embeddings parquet file')
    
    args = parser.parse_args()
    main(args)
