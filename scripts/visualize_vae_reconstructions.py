"""
Visualize VAE reconstructions and latent space manipulations.

This script allows you to:
1. Load a trained VAE model
2. Reconstruct ECGs from the test set
3. Manipulate latent features and generate new ECGs
4. Visualize original vs reconstructed vs manipulated ECGs

Usage:
    python scripts/visualize_vae_reconstructions.py \
        --checkpoint_path models/checkpoints/vae_zdim64_beta4.0_20251102_073456/best_model.pt \
        --num_samples 5 \
        --output_dir reports/figures/vae_reconstructions
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.vae_conv1d import Conv1DVAE, count_parameters
from src.data.ecg_dataset import ECGDataset, create_train_val_test_splits, create_dataloaders


def plot_12_lead_ecg(signal, title="12-Lead ECG", figsize=(15, 10), save_path=None):
    """
    Plot a 12-lead ECG signal.
    
    Parameters
    ----------
    signal : np.ndarray or torch.Tensor
        ECG signal, shape (12, 5000)
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()
    
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    fig, axes = plt.subplots(12, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Time axis (assuming 500 Hz, 10 seconds)
    time = np.linspace(0, 10, signal.shape[1])
    
    for i, (ax, lead_name) in enumerate(zip(axes, lead_names)):
        ax.plot(time, signal[i], linewidth=0.8, color='black')
        ax.set_ylabel(lead_name, fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(signal[i].min() - 0.5, signal[i].max() + 0.5)
        
        if i == 11:  # Last subplot
            ax.set_xlabel('Time (seconds)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    return fig


def plot_reconstruction_comparison(original, reconstructed, sample_idx, label=None, 
                                   figsize=(20, 10), save_path=None):
    """
    Plot original vs reconstructed ECG side by side.
    
    Parameters
    ----------
    original : torch.Tensor
        Original ECG, shape (12, 5000)
    reconstructed : torch.Tensor
        Reconstructed ECG, shape (12, 5000)
    sample_idx : int
        Sample index for title
    label : str, optional
        Sample label
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.cpu().numpy()
    
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(12, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    title_text = f"Sample {sample_idx}"
    if label:
        title_text += f" ({label})"
    fig.suptitle(title_text, fontsize=16, fontweight='bold')
    
    # Time axis
    time = np.linspace(0, 10, original.shape[1])
    
    # Calculate MSE per lead
    mse_per_lead = np.mean((original - reconstructed) ** 2, axis=1)
    
    for i, lead_name in enumerate(lead_names):
        # Original
        ax_orig = fig.add_subplot(gs[i, 0])
        ax_orig.plot(time, original[i], linewidth=0.8, color='blue')
        if i == 0:
            ax_orig.set_title('Original', fontsize=14, fontweight='bold')
        ax_orig.set_ylabel(lead_name, fontweight='bold', fontsize=10)
        ax_orig.grid(True, alpha=0.3)
        
        # Reconstructed
        ax_recon = fig.add_subplot(gs[i, 1])
        ax_recon.plot(time, reconstructed[i], linewidth=0.8, color='red')
        if i == 0:
            ax_recon.set_title(f'Reconstructed (MSE: {mse_per_lead.mean():.4f})', 
                             fontsize=14, fontweight='bold')
        ax_recon.set_ylabel(f'MSE: {mse_per_lead[i]:.4f}', fontsize=9)
        ax_recon.grid(True, alpha=0.3)
        
        if i == 11:  # Last row
            ax_orig.set_xlabel('Time (s)', fontsize=12)
            ax_recon.set_xlabel('Time (s)', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison to {save_path}")
    
    return fig


def manipulate_latent_feature(model, original_signal, feature_idx, delta_values, device):
    """
    Manipulate a specific latent feature and generate ECGs.
    
    Parameters
    ----------
    model : Conv1DVAE
        Trained VAE model
    original_signal : torch.Tensor
        Original ECG signal, shape (12, 5000)
    feature_idx : int
        Index of latent dimension to manipulate
    delta_values : list
        List of values to add to the latent feature
    device : torch.device
        Device
    
    Returns
    -------
    manipulated_ecgs : list of np.ndarray
        List of manipulated ECGs
    original_z : np.ndarray
        Original latent vector
    """
    model.eval()
    
    with torch.no_grad():
        # Encode
        signal_batch = original_signal.unsqueeze(0).to(device)
        mu, logvar = model.encode(signal_batch)
        z = mu  # Use mean for deterministic reconstruction
        
        original_z = z.cpu().numpy().squeeze()
        manipulated_ecgs = []
        
        for delta in delta_values:
            # Modify latent feature
            z_modified = z.clone()
            z_modified[0, feature_idx] += delta
            
            # Decode
            recon = model.decode(z_modified)
            manipulated_ecgs.append(recon.cpu().numpy().squeeze())
    
    return manipulated_ecgs, original_z


def plot_latent_manipulation(original, manipulated_ecgs, delta_values, feature_idx,
                            original_z, lead_idx=1, figsize=(20, 12), save_path=None):
    """
    Plot the effect of manipulating a latent feature.
    
    Parameters
    ----------
    original : np.ndarray
        Original ECG, shape (12, 5000)
    manipulated_ecgs : list of np.ndarray
        List of manipulated ECGs
    delta_values : list
        Delta values used
    feature_idx : int
        Latent dimension manipulated
    original_z : np.ndarray
        Original latent vector
    lead_idx : int
        Which lead to plot (default: Lead II)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save
    """
    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()
    
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.3)
    
    # Time axis
    time = np.linspace(0, 10, original.shape[1])
    
    # Plot manipulated ECGs for one lead
    ax1 = fig.add_subplot(gs[0])
    colors = plt.cm.RdYlBu(np.linspace(0, 1, len(delta_values)))
    
    for i, (ecg, delta, color) in enumerate(zip(manipulated_ecgs, delta_values, colors)):
        label = f'z[{feature_idx}] + {delta:.2f}'
        alpha = 0.5 if delta != 0 else 1.0
        linewidth = 2 if delta == 0 else 1
        ax1.plot(time, ecg[lead_idx], label=label, color=color, alpha=alpha, linewidth=linewidth)
    
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel(f'Lead {lead_names[lead_idx]} Amplitude', fontsize=12)
    ax1.set_title(f'Latent Feature z[{feature_idx}] Manipulation (Original value: {original_z[feature_idx]:.4f})',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Plot latent vector
    ax2 = fig.add_subplot(gs[1])
    z_dim = len(original_z)
    ax2.bar(range(z_dim), original_z, color='steelblue', alpha=0.7)
    ax2.axvline(feature_idx, color='red', linestyle='--', linewidth=2, label=f'Manipulated: z[{feature_idx}]')
    ax2.set_xlabel('Latent Dimension', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title(f'Original Latent Vector (z_dim={z_dim})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved manipulation plot to {save_path}")
    
    return fig


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    # Load metadata
    print(f"\n✓ Loading metadata from {args.metadata_path}")
    df = pd.read_parquet(args.metadata_path)
    
    # Filter for training
    train_labels = ['Control_Symptomatic', 'MI_Pre-Incident']
    df_train = df[df['Label'].isin(train_labels)].copy()
    
    # Create splits
    print(f"✓ Creating train/val/test splits (80/10/10)")
    train_df, val_df, test_df = create_train_val_test_splits(
        df_train, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        stratify_col='Label', random_state=42
    )
    print(f"✓ Test set: {len(test_df)} ECGs")
    
    # Create test dataset
    test_dataset = ECGDataset(
        metadata_df=test_df,
        base_path=args.base_path,
        normalize=True
    )
    
    # Load model
    print(f"\n✓ Initializing Conv1D β-VAE (z_dim={args.z_dim})")
    model = Conv1DVAE(z_dim=args.z_dim, beta=args.beta, free_bits=2.0).to(device)
    
    print(f"✓ Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded (epoch {checkpoint.get('epoch', 'Unknown')})")
    
    # Select random samples
    print(f"\n✓ Selecting {args.num_samples} random test samples...")
    indices = np.random.choice(len(test_dataset), size=args.num_samples, replace=False)
    
    # Generate reconstructions
    print(f"\n{'='*80}")
    print(f"Generating Reconstructions")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        for i, idx in enumerate(tqdm(indices, desc="Processing samples")):
            sample = test_dataset[idx]
            if sample is None:
                print(f"Warning: Sample {idx} failed to load, skipping...")
                continue
            
            signal, metadata = sample
            label = metadata.get('label', 'Unknown')
            
            # Reconstruct
            signal_batch = signal.unsqueeze(0).to(device)
            recon, mu, logvar = model(signal_batch)
            recon = recon.squeeze(0)
            
            # Plot comparison
            save_path = output_dir / f"sample_{i+1}_reconstruction.png"
            plot_reconstruction_comparison(
                signal, recon, sample_idx=i+1, label=label, save_path=save_path
            )
            plt.close()
    
    # Latent feature manipulation
    if args.manipulate_features:
        print(f"\n{'='*80}")
        print(f"Latent Feature Manipulation")
        print(f"{'='*80}\n")
        
        # Use first valid sample
        sample = test_dataset[indices[0]]
        if sample is not None:
            signal, metadata = sample
            label = metadata.get('label', 'Unknown')
            
            # Manipulate multiple features
            features_to_manipulate = args.feature_indices or [0, 1, 2, 5, 10]
            delta_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
            
            for feat_idx in features_to_manipulate:
                if feat_idx >= args.z_dim:
                    continue
                
                print(f"  Manipulating latent feature z[{feat_idx}]...")
                manipulated_ecgs, original_z = manipulate_latent_feature(
                    model, signal, feat_idx, delta_values, device
                )
                
                # Plot for Lead II (most informative)
                save_path = output_dir / f"latent_manipulation_z{feat_idx}_lead2.png"
                plot_latent_manipulation(
                    signal, manipulated_ecgs, delta_values, feat_idx,
                    original_z, lead_idx=1, save_path=save_path
                )
                plt.close()
    
    print(f"\n{'='*80}")
    print(f"✓ All visualizations saved to {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize VAE reconstructions and manipulations')
    
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
                       help='Beta value')
    
    # Visualization
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--manipulate_features', action='store_true',
                       help='Generate latent feature manipulation plots')
    parser.add_argument('--feature_indices', type=int, nargs='+',
                       help='Specific latent features to manipulate (e.g., 0 1 2 5 10)')
    parser.add_argument('--output_dir', type=str,
                       default='reports/figures/vae_reconstructions',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    main(args)
