"""
Test script for evaluating the best VAE model on the test set.

Usage:
    python scripts/test_vae_model.py --checkpoint_path models/checkpoints/vae_zdim64_beta4.0_20251102_073456/best_model.pt
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.vae_conv1d import Conv1DVAE, count_parameters
from src.data.ecg_dataset import create_train_val_test_splits, create_dataloaders


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    beta: float = 1.0
) -> dict:
    """
    Evaluate the model on the test set.
    
    Parameters
    ----------
    model : nn.Module
        Trained VAE model
    test_loader : DataLoader
        Test data loader
    device : torch.device
        Device to run evaluation on
    beta : float
        Beta value for KL weighting (use 1.0 for fair comparison)
    
    Returns
    -------
    results : dict
        Dictionary containing test metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_kl_raw = 0.0
    n_batches = 0
    n_samples = 0
    
    print(f"\n{'='*80}")
    print(f"Testing model with β={beta}")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", leave=True)
        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch
            if batch_data is None or batch_data[0] is None:
                print(f"Warning: Batch {batch_idx} returned None - all samples failed to load")
                continue
            
            signals, metadata = batch_data
            signals = signals.to(device)
            batch_size = signals.size(0)
            
            # Forward pass
            recon, mu, logvar = model(signals)
            
            # Calculate losses using model's loss_function
            loss_dict = model.loss_function(signals, recon, mu, logvar)
            
            # Extract individual losses
            total_loss_batch = loss_dict['loss']
            recon_loss = loss_dict['recon_loss']
            kl_loss = loss_dict['kl_loss']
            kl_raw = loss_dict['kl_raw']
            
            # Accumulate
            total_loss += total_loss_batch.item() * batch_size
            total_recon_loss += recon_loss.item() * batch_size
            total_kl_loss += kl_loss.item() * batch_size
            total_kl_raw += kl_raw.item() * batch_size
            n_samples += batch_size
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss_batch.item():.2f}',
                'recon': f'{recon_loss.item():.2f}',
                'kl': f'{kl_loss.item():.2f}'
            })
    
    # Calculate averages
    if n_samples == 0:
        raise ValueError("No samples were processed! Check your data loader.")
    
    results = {
        'test_loss': total_loss / n_samples,
        'test_recon_loss': total_recon_loss / n_samples,
        'test_kl_loss': total_kl_loss / n_samples,
        'test_kl_raw': total_kl_raw / n_samples,
        'n_samples': n_samples,
        'n_batches': n_batches
    }
    
    return results


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    if device.type == 'cuda':
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load metadata
    print(f"\n✓ Loading metadata from {args.metadata_path}")
    df = pd.read_parquet(args.metadata_path)
    print(f"✓ Loaded {len(df)} records")
    
    # Filter for training/testing
    train_labels = ['Control_Symptomatic', 'MI_Pre-Incident']
    df_train = df[df['Label'].isin(train_labels)].copy()
    print(f"✓ Filtered to {len(df_train)} ECGs for testing")
    
    # Create splits (same seed as training for reproducibility)
    print(f"\n✓ Creating train/val/test splits (80/10/10)")
    train_df, val_df, test_df = create_train_val_test_splits(
        df_train,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        stratify_col='Label',
        random_state=42  # Same as training
    )
    print(f"✓ Test set: {len(test_df)} ECGs")
    print(f"✓ Test label distribution:\n{test_df['Label'].value_counts()}")
    
    # Create test dataloader
    print(f"\n✓ Creating test dataloader (batch_size={args.batch_size})")
    
    # First, let's test if we can load a single sample
    from src.data.ecg_dataset import ECGDataset
    test_dataset = ECGDataset(
        metadata_df=test_df,
        base_path=args.base_path,
        normalize=True
    )
    print(f"✓ Test dataset created: {len(test_dataset)} samples")
    
    # Try loading first sample
    print(f"✓ Testing first sample load...")
    sample = test_dataset[0]
    if sample is None:
        print(f"ERROR: First sample failed to load!")
        print(f"  Base path: {args.base_path}")
        print(f"  First file: {test_df.iloc[0]['file_path']}")
        import sys
        sys.exit(1)
    else:
        print(f"✓ First sample loaded successfully: shape {sample[0].shape}")
    
    _, _, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        base_path=args.base_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=True
    )
    print(f"✓ Test batches: {len(test_loader)}")
    
    # Initialize model
    print(f"\n✓ Initializing Conv1D β-VAE (z_dim={args.z_dim})")
    model = Conv1DVAE(z_dim=args.z_dim, beta=args.beta, free_bits=2.0).to(device)
    print(f"✓ Total parameters: {count_parameters(model):,}")
    
    # Load checkpoint
    print(f"\n✓ Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get checkpoint info
    checkpoint_epoch = checkpoint.get('epoch', 'Unknown')
    checkpoint_val_loss = checkpoint.get('best_val_loss', 'Unknown')
    print(f"✓ Checkpoint loaded successfully!")
    print(f"  Epoch: {checkpoint_epoch}")
    print(f"  Best val loss: {checkpoint_val_loss}")
    
    # Test the model
    test_results = test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        beta=args.test_beta
    )
    
    # Print results
    print(f"\n{'='*80}")
    print(f"TEST RESULTS (β={args.test_beta})")
    print(f"{'='*80}")
    print(f"  Test Loss:       {test_results['test_loss']:.4f}")
    print(f"  Recon Loss:      {test_results['test_recon_loss']:.4f}")
    print(f"  KL Loss:         {test_results['test_kl_loss']:.4f}")
    print(f"  KL Raw:          {test_results['test_kl_raw']:.4f}")
    print(f"  Samples tested:  {test_results['n_samples']}")
    print(f"{'='*80}\n")
    
    # Save results to JSON
    if args.output_path:
        import json
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_to_save = {
            'checkpoint_path': str(args.checkpoint_path),
            'checkpoint_epoch': int(checkpoint_epoch) if checkpoint_epoch != 'Unknown' else None,
            'checkpoint_val_loss': float(checkpoint_val_loss) if checkpoint_val_loss != 'Unknown' else None,
            'test_beta': args.test_beta,
            'test_results': test_results,
            'model_config': {
                'z_dim': args.z_dim,
                'beta': args.beta,
                'total_parameters': count_parameters(model)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"✓ Results saved to {output_path}\n")
    
    return test_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test VAE model on test set')
    
    # Data
    parser.add_argument('--metadata_path', type=str, 
                       default='data/processed/ecg_features_with_demographics.parquet',
                       help='Path to metadata parquet file')
    parser.add_argument('--base_path', type=str, 
                       default='data/raw/MIMIC-IV-ECG-1.0/files',
                       help='Base path for WFDB records')
    
    # Model
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--z_dim', type=int, default=64,
                       help='Latent dimension')
    parser.add_argument('--beta', type=float, default=4.0,
                       help='Beta value used during training')
    parser.add_argument('--test_beta', type=float, default=1.0,
                       help='Beta value for testing (default: 1.0 for fair comparison)')
    
    # Testing
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of DataLoader workers')
    parser.add_argument('--output_path', type=str, 
                       default='reports/test_results.json',
                       help='Path to save test results (JSON)')
    
    args = parser.parse_args()
    main(args)
