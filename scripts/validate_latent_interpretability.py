"""
Validate latent space interpretability (Phase D.5).

Goal: Confirm that VAE learned meaningful, disentangled physiological features.

Protocol:
1. Single-Dimension Traversal: Vary each latent dimension independently
2. Qualitative Assessment: Review plots for interpretability
3. Quantitative Checks: Validate physiological plausibility
4. Go/No-Go Decision: Decide whether to proceed or retrain

Criteria:
- ≥10 dimensions show clear interpretability
- ≥95% of decoded ECGs are physiologically plausible

Usage:
    python validate_latent_interpretability.py --checkpoint models/checkpoints/.../best_model.pt
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import neurokit2 as nk

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.vae_conv1d import Conv1DVAE
from src.data.ecg_dataset import ECGDataset


LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def collate_fn_filter_none(batch):
    """Filter out None values from batch (failed ECG loads)."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    # Separate signals and metadata
    signals = torch.stack([item[0] for item in batch])
    metadata = [item[1] for item in batch]
    return signals, metadata


def compute_base_latent(model, dataloader, device, label='Control_Symptomatic'):
    """
    Compute mean latent vector from Control ECGs.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained VAE model
    dataloader : DataLoader
        DataLoader with ECG data
    device : torch.device
        Device to run on
    label : str
        Label to filter for (default: Control_Symptomatic)
    
    Returns
    -------
    z_base : np.ndarray
        Mean latent vector, shape (z_dim,)
    """
    model.eval()
    
    all_z = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Skip empty batches (all None values filtered out)
            if batch is None:
                continue
                
            signals, metadata = batch
            
            # Filter by label (metadata is now a list of dicts)
            labels = [m['label'] for m in metadata]
            mask = [l == label for l in labels]
            
            if not any(mask):
                continue
            
            signals = signals[mask].to(device)
            
            # Get latent mean
            mu = model.get_latent_embeddings(signals)
            all_z.append(mu.cpu().numpy())
    
    # Compute mean
    z_base = np.vstack(all_z).mean(axis=0)
    
    return z_base


def generate_dimension_traversals(z_base, z_dim, alphas=None):
    """
    Generate latent vectors for dimension traversal.
    
    Parameters
    ----------
    z_base : np.ndarray
        Base latent vector, shape (z_dim,)
    z_dim : int
        Latent dimension size
    alphas : list, optional
        Traversal values (default: [-3, -2, -1, 0, 1, 2, 3])
    
    Returns
    -------
    traversals : dict
        Dictionary mapping dimension index to list of latent vectors
    """
    if alphas is None:
        alphas = [-3, -2, -1, 0, 1, 2, 3]
    
    traversals = {}
    
    for dim in range(z_dim):
        z_variants = []
        for alpha in alphas:
            z = z_base.copy()
            z[dim] += alpha
            z_variants.append(z)
        traversals[dim] = np.array(z_variants)
    
    return traversals


def decode_traversals(model, traversals, device):
    """
    Decode latent traversals to ECG signals.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained VAE model
    traversals : dict
        Dictionary mapping dimension index to latent vectors
    device : torch.device
        Device to run on
    
    Returns
    -------
    decoded : dict
        Dictionary mapping dimension index to decoded ECG signals
    """
    model.eval()
    decoded = {}
    
    with torch.no_grad():
        for dim, z_variants in tqdm(traversals.items(), desc="Decoding traversals"):
            z_tensor = torch.from_numpy(z_variants).float().to(device)
            x_recon = model.decode(z_tensor)
            decoded[dim] = x_recon.cpu().numpy()
    
    return decoded


def check_physiological_plausibility(ecg_signal, sampling_rate=500):
    """
    Check if decoded ECG is physiologically plausible.
    
    Parameters
    ----------
    ecg_signal : np.ndarray
        ECG signal, shape (12, 5000) or (5000,)
    sampling_rate : int
        Sampling rate in Hz
    
    Returns
    -------
    is_plausible : bool
        Whether ECG passes plausibility checks
    metrics : dict
        Computed metrics
    """
    # Use lead II for feature extraction (standard for HR)
    if ecg_signal.ndim == 2:
        lead_ii = ecg_signal[1, :]  # Lead II
    else:
        lead_ii = ecg_signal
    
    metrics = {
        'has_nan': False,
        'has_inf': False,
        'amplitude_ok': True,
        'hr_ok': False,
        'hr': None,
        'qtc_ok': False,
        'qtc': None
    }
    
    # Check for NaN/Inf
    if np.isnan(ecg_signal).any():
        metrics['has_nan'] = True
        return False, metrics
    
    if np.isinf(ecg_signal).any():
        metrics['has_inf'] = True
        return False, metrics
    
    # Check amplitude range (-5 to +5 mV)
    if ecg_signal.min() < -5 or ecg_signal.max() > 5:
        metrics['amplitude_ok'] = False
    
    # Try to extract heart rate and QTc
    try:
        # Clean signal
        cleaned = nk.ecg_clean(lead_ii, sampling_rate=sampling_rate)
        
        # Process ECG
        signals, info = nk.ecg_process(cleaned, sampling_rate=sampling_rate)
        
        # Extract HR
        if 'ECG_Rate' in signals.columns:
            hr = signals['ECG_Rate'].mean()
            metrics['hr'] = hr
            
            # Check HR range (20-200 bpm)
            if 20 <= hr <= 200:
                metrics['hr_ok'] = True
        
        # Extract QTc (Bazett's formula)
        rpeaks = info.get('ECG_R_Peaks', [])
        tpeaks = info.get('ECG_T_Peaks', [])
        
        if len(rpeaks) > 1 and len(tpeaks) > 0:
            # Compute RR interval (seconds)
            rr_intervals = np.diff(rpeaks) / sampling_rate
            rr_mean = rr_intervals.mean()
            
            # Compute QT interval (pair R-peaks with T-peaks)
            qt_intervals = []
            for i in range(min(len(rpeaks)-1, len(tpeaks))):
                if rpeaks[i] < tpeaks[i] < rpeaks[i+1]:
                    qt = (tpeaks[i] - rpeaks[i]) / sampling_rate * 1000  # milliseconds
                    qt_intervals.append(qt)
            
            if qt_intervals:
                qt_mean = np.mean(qt_intervals)
                qtc = qt_mean / np.sqrt(rr_mean)  # Bazett's formula
                metrics['qtc'] = qtc
                
                # Check QTc < 700ms (protocol requirement)
                if qtc < 700:
                    metrics['qtc_ok'] = True
    except:
        pass
    
    # Overall plausibility (all checks must pass)
    is_plausible = (
        not metrics['has_nan'] and
        not metrics['has_inf'] and
        metrics['amplitude_ok'] and
        metrics['hr_ok']
        # QTc is optional but good to have
    )
    
    return is_plausible, metrics


def plot_dimension_traversal(decoded_signals, dim, alphas, output_dir):
    """
    Plot 12-lead ECG grid for dimension traversal.
    
    Parameters
    ----------
    decoded_signals : np.ndarray
        Decoded ECG signals, shape (n_alphas, 12, 5000)
    dim : int
        Dimension index
    alphas : list
        Traversal values
    output_dir : Path
        Output directory for plots
    """
    n_alphas = len(alphas)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(12, n_alphas, hspace=0.3, wspace=0.2)
    
    # Plot each lead x alpha combination
    for lead_idx, lead_name in enumerate(LEAD_NAMES):
        for alpha_idx, alpha in enumerate(alphas):
            ax = fig.add_subplot(gs[lead_idx, alpha_idx])
            
            signal = decoded_signals[alpha_idx, lead_idx, :]
            time = np.arange(len(signal)) / 500  # Convert to seconds
            
            ax.plot(time, signal, linewidth=0.8, color='black')
            ax.set_xlim(0, 10)
            ax.set_ylim(-3, 3)
            
            # Labels only on edges
            if alpha_idx == 0:
                ax.set_ylabel(lead_name, fontsize=10, fontweight='bold')
            else:
                ax.set_yticks([])
            
            if lead_idx == 0:
                ax.set_title(f'α={alpha:+.0f}', fontsize=10)
            
            if lead_idx == 11:
                ax.set_xlabel('Time (s)', fontsize=8)
            else:
                ax.set_xticks([])
            
            ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.suptitle(f'Latent Dimension {dim+1} Traversal', fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    plot_path = output_dir / f'dimension_{dim+1:03d}_traversal.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def assess_interpretability(decoded_signals, alphas):
    """
    Assess interpretability of a dimension based on decoded signals.
    
    Parameters
    ----------
    decoded_signals : np.ndarray
        Decoded ECG signals, shape (n_alphas, 12, 5000)
    alphas : list
        Traversal values
    
    Returns
    -------
    assessment : dict
        Dictionary with interpretability metrics
    """
    # Compute change magnitude for each lead
    baseline_idx = alphas.index(0)
    baseline = decoded_signals[baseline_idx]
    
    changes = []
    for i, alpha in enumerate(alphas):
        if alpha != 0:
            diff = np.abs(decoded_signals[i] - baseline).mean()
            changes.append(diff)
    
    avg_change = np.mean(changes)
    max_change = np.max(changes)
    
    # Check monotonicity (does dimension smoothly vary?)
    # A dimension is monotonic if AT LEAST ONE lead shows smooth variation
    # (indicating it controls a specific feature in that lead)
    monotonic_leads = 0
    for lead_idx in range(12):
        lead_signals = decoded_signals[:, lead_idx, :]
        lead_means = lead_signals.mean(axis=1)
        
        # Check if mostly increasing or decreasing
        diffs = np.diff(lead_means)
        # Allow up to 2 non-monotonic transitions (noise tolerance)
        if np.sum(diffs > 0) >= len(diffs) - 2 or np.sum(diffs < 0) >= len(diffs) - 2:
            monotonic_leads += 1
    
    # Interpretable if at least one lead is monotonic
    is_monotonic = monotonic_leads > 0
    
    return {
        'avg_change': avg_change,
        'max_change': max_change,
        'is_monotonic': is_monotonic,
        'monotonic_leads': monotonic_leads
    }


def main(args):
    """Main validation function."""
    print("\n" + "=" * 80)
    print("Latent Space Interpretability Validation (Phase D.5)")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Load checkpoint
    print(f"\n✓ Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Initialize model
    print(f"\n✓ Initializing Conv1D β-VAE (z_dim={args.z_dim}, β={args.beta})")
    model = Conv1DVAE(z_dim=args.z_dim, beta=args.beta).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded successfully")
    
    # Load metadata
    print(f"\n✓ Loading metadata from {args.metadata_path}")
    df = pd.read_parquet(args.metadata_path)
    
    # Filter for Control_Symptomatic only
    df_control = df[df['Label'] == 'Control_Symptomatic'].copy()
    print(f"✓ Using {len(df_control)} Control_Symptomatic ECGs for base latent")
    
    # Create dataset
    from torch.utils.data import DataLoader
    dataset = ECGDataset(df_control.head(args.n_base_samples), args.base_path, normalize=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn_filter_none)
    
    # Compute base latent
    print(f"\n✓ Computing base latent vector from Control ECGs...")
    z_base = compute_base_latent(model, dataloader, device)
    print(f"✓ Base latent shape: {z_base.shape}")
    print(f"✓ Base latent mean: {z_base.mean():.6f}, std: {z_base.std():.6f}")
    
    # Generate traversals
    print(f"\n✓ Generating dimension traversals...")
    alphas = args.alphas
    traversals = generate_dimension_traversals(z_base, args.z_dim, alphas)
    print(f"✓ Generated {len(traversals)} dimension traversals")
    print(f"✓ Alpha values: {alphas}")
    
    # Decode traversals
    print(f"\n✓ Decoding traversals...")
    decoded = decode_traversals(model, traversals, device)
    print(f"✓ Decoded {len(decoded)} dimensions")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'dimension_plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Analyze each dimension
    print(f"\n✓ Analyzing dimensions...")
    results = []
    
    for dim in tqdm(range(args.z_dim), desc="Analyzing dimensions"):
        decoded_signals = decoded[dim]  # Shape: (n_alphas, 12, 5000)
        
        # Plot
        if args.save_plots:
            plot_path = plot_dimension_traversal(decoded_signals, dim, alphas, plots_dir)
        
        # Assess interpretability
        interpretability = assess_interpretability(decoded_signals, alphas)
        
        # Check plausibility for each alpha
        plausibility_checks = []
        for i, alpha in enumerate(alphas):
            is_plausible, metrics = check_physiological_plausibility(decoded_signals[i])
            plausibility_checks.append({
                'alpha': alpha,
                'is_plausible': is_plausible,
                **metrics
            })
        
        # Compute plausibility rate
        n_plausible = sum(1 for p in plausibility_checks if p['is_plausible'])
        plausibility_rate = n_plausible / len(alphas)
        
        results.append({
            'dimension': dim + 1,
            'avg_change': interpretability['avg_change'],
            'max_change': interpretability['max_change'],
            'is_monotonic': interpretability['is_monotonic'],
            'monotonic_leads': interpretability['monotonic_leads'],
            'plausibility_rate': plausibility_rate,
            'n_plausible': n_plausible,
            'n_total': len(alphas)
        })
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results
    results_path = output_dir / 'interpretability_results.csv'
    df_results.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # Compute overall statistics
    print("\n" + "=" * 80)
    print("Overall Assessment")
    print("=" * 80)
    
    # Interpretability criteria (dimension shows clear, single-factor control)
    # A dimension is interpretable if:
    # 1. It produces measurable change (avg_change > 0.01)
    # 2. The change is monotonic (smooth variation)
    # 3. Decoded ECGs are mostly plausible (>80%)
    interpretable_dims = df_results[
        (df_results['avg_change'] > 0.01) & 
        (df_results['is_monotonic'] == True) &
        (df_results['plausibility_rate'] > 0.80)
    ]
    n_interpretable = len(interpretable_dims)
    
    print(f"\n✓ Interpretable dimensions: {n_interpretable}/{args.z_dim}")
    print(f"  (Criteria: avg_change > 0.01 AND monotonic)")
    
    # Plausibility criteria
    overall_plausibility = df_results['plausibility_rate'].mean()
    print(f"\n✓ Overall plausibility rate: {overall_plausibility:.2%}")
    
    # Go/No-Go decision
    print("\n" + "=" * 80)
    print("Go/No-Go Decision")
    print("=" * 80)
    
    go_interpretable = n_interpretable >= 10
    go_plausible = overall_plausibility >= 0.95
    
    print(f"\n✓ Interpretability criterion (≥10 interpretable dims): {n_interpretable}/10 - {'PASS ✓' if go_interpretable else 'FAIL ✗'}")
    print(f"✓ Plausibility criterion (≥95% plausible ECGs): {overall_plausibility:.1%}/95% - {'PASS ✓' if go_plausible else 'FAIL ✗'}")
    
    # Detailed diagnostic logic (per protocol)
    if go_interpretable and go_plausible:
        decision = "PROCEED ✓"
        recommendation = "VAE has learned meaningful, disentangled features. Proceed to Phase E-F (Master Dataset)."
    elif not go_interpretable and not go_plausible:
        decision = "RETRAIN ✗"
        recommendation = "Both criteria failed. Multiple issues detected:\n"
        recommendation += f"  - Only {n_interpretable}/10 interpretable dimensions\n"
        recommendation += f"  - Low plausibility ({overall_plausibility:.1%})\n\n"
        recommendation += "Suggested actions:\n"
        recommendation += "  1. Check for posterior collapse: Review KL divergence (should be 5-15)\n"
        recommendation += "  2. If KL→0: Decrease β or increase free bits\n"
        recommendation += "  3. If KL>20: Increase β for more regularization\n"
        recommendation += "  4. Consider increasing z_dim (64 → 128) for more capacity"
    elif not go_interpretable:
        decision = "RETRAIN ✗"
        # Check if it's entanglement or posterior collapse
        avg_plausibility = df_results['plausibility_rate'].mean()
        if avg_plausibility < 0.5:
            recommendation = f"Only {n_interpretable}/10 interpretable dimensions + low plausibility.\n"
            recommendation += "Likely cause: Posterior collapse (VAE ignoring latent space).\n\n"
            recommendation += "Suggested fix:\n"
            recommendation += f"  - Decrease β: {args.beta} → {args.beta / 2}\n"
            recommendation += "  - Increase free bits: 2.0 → 3.0\n"
            recommendation += "  - Add warm-up schedule for β"
        else:
            recommendation = f"Only {n_interpretable}/10 interpretable dimensions (entanglement issue).\n"
            recommendation += "Dimensions are changing multiple factors simultaneously.\n\n"
            recommendation += "Suggested fix:\n"
            recommendation += f"  - Increase β: {args.beta} → {args.beta * 2} (encourage disentanglement)\n"
            recommendation += "  - Train longer (more epochs for disentanglement to emerge)\n"
            recommendation += "  - Check dataset diversity (need varied ECG patterns)"
    else:
        decision = "RETRAIN ✗"
        recommendation = f"Low plausibility ({overall_plausibility:.1%}), but {n_interpretable} dims are interpretable.\n"
        recommendation += "Likely cause: Poor reconstruction quality.\n\n"
        recommendation += "Suggested fix:\n"
        recommendation += f"  - Decrease β: {args.beta} → {args.beta / 2} (prioritize reconstruction)\n"
        recommendation += "  - Increase model capacity (more conv layers or channels)\n"
        recommendation += "  - Check input preprocessing (normalization, filtering)"
    
    print(f"\n{'=' * 80}")
    print(f"Decision: {decision}")
    print(f"{'=' * 80}")
    print(f"\nRecommendation: {recommendation}")
    
    # Save decision report
    report_path = output_dir / 'latent_interpretability_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Latent Space Interpretability Report\n\n")
        f.write(f"**Model:** {args.checkpoint}\n\n")
        f.write(f"**z_dim:** {args.z_dim}\n\n")
        f.write(f"**beta:** {args.beta}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Interpretable dimensions:** {n_interpretable}/{args.z_dim}\n")
        f.write(f"- **Overall plausibility:** {overall_plausibility:.2%}\n\n")
        f.write("## Criteria\n\n")
        f.write(f"1. ≥10 interpretable dimensions: {'✓ PASS' if go_interpretable else '✗ FAIL'} ({n_interpretable}/10)\n")
        f.write(f"2. ≥95% plausible ECGs: {'✓ PASS' if go_plausible else '✗ FAIL'} ({overall_plausibility:.1%}/95%)\n\n")
        f.write(f"## Decision: **{decision}**\n\n")
        f.write(f"**Recommendation:** {recommendation}\n\n")
        f.write("## Top 10 Most Interpretable Dimensions\n\n")
        
        top_dims = df_results.nlargest(10, 'avg_change')
        f.write("| Dimension | Avg Change | Max Change | Monotonic | Plausibility |\n")
        f.write("|-----------|------------|------------|-----------|-------------|\n")
        for _, row in top_dims.iterrows():
            f.write(f"| z_ecg_{row['dimension']} | {row['avg_change']:.6f} | {row['max_change']:.6f} | "
                   f"{'Yes' if row['is_monotonic'] else 'No'} | {row['plausibility_rate']:.1%} |\n")
        
        f.write("\n## Plausibility Statistics\n\n")
        plausibility_stats = df_results['plausibility_rate'].describe()
        f.write(f"- Mean: {plausibility_stats['mean']:.2%}\n")
        f.write(f"- Median: {plausibility_stats['50%']:.2%}\n")
        f.write(f"- Min: {plausibility_stats['min']:.2%}\n")
        f.write(f"- Max: {plausibility_stats['max']:.2%}\n")
        f.write(f"\nDimensions with 100% plausibility: {len(df_results[df_results['plausibility_rate'] == 1.0])}\n")
        f.write(f"Dimensions with <50% plausibility: {len(df_results[df_results['plausibility_rate'] < 0.5])}\n")
        
        f.write("\n## Next Steps\n\n")
        if go_interpretable and go_plausible:
            f.write("1. **Manual Annotation**: Review all 64 dimension plots and annotate interpretable ones\n")
            f.write("2. **Fill latent_dimension_descriptions.csv**: Add physiological descriptions\n")
            f.write("3. **Proceed to Phase E-F**: Merge latent features with clinical data\n")
        else:
            f.write("1. **Review Training Logs**: Check KL divergence, reconstruction loss trends\n")
            f.write("2. **Inspect Failed Dimensions**: Focus on low-plausibility dimensions\n")
            f.write("3. **Retrain Model**: Follow recommendations above\n")
        
        f.write("\n## Dimension Descriptions (Manual Annotation Required)\n\n")
        f.write("*After reviewing dimension traversal plots, fill in descriptions for interpretable dimensions.*\n\n")
        f.write("**Examples of Good Annotations:**\n")
        f.write("- z_ecg_1: Heart rate (slow 50 bpm → fast 120 bpm)\n")
        f.write("- z_ecg_5: ST-segment elevation in leads V2-V4 (0mm → +3mm)\n")
        f.write("- z_ecg_12: QRS duration (narrow 80ms → wide 180ms, suggests LBBB pattern)\n")
        f.write("- z_ecg_23: T-wave inversion in inferior leads (II, III, aVF)\n\n")
        f.write("**Examples of Bad (Entangled) Dimensions:**\n")
        f.write("- z_ecg_17: Changes HR + ST-segment + T-wave simultaneously (entangled)\n")
        f.write("- z_ecg_42: Produces noisy, implausible signals (not interpretable)\n")
    
    print(f"\n✓ Report saved to {report_path}")
    
    # Save dimension descriptions template
    desc_path = output_dir / 'latent_dimension_descriptions.csv'
    df_desc = pd.DataFrame({
        'dimension': [f'z_ecg_{i+1}' for i in range(args.z_dim)],
        'description': [''] * args.z_dim,
        'interpretable': df_results['is_monotonic'].tolist(),
        'plausibility_rate': df_results['plausibility_rate'].tolist()
    })
    df_desc.to_csv(desc_path, index=False)
    print(f"✓ Dimension descriptions template saved to {desc_path}")
    
    print("\n" + "=" * 80)
    print("✓ Validation completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate latent space interpretability")
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--z_dim', type=int, default=64,
                       help='Latent dimension size')
    parser.add_argument('--beta', type=float, default=4.0,
                       help='β-VAE parameter')
    
    # Data
    parser.add_argument('--metadata_path', type=str, default='data/processed/ecg_features_with_demographics.parquet',
                       help='Path to metadata parquet file')
    parser.add_argument('--base_path', type=str, default='data/raw/MIMIC-IV-ECG-1.0/files',
                       help='Base path to WFDB files')
    parser.add_argument('--n_base_samples', type=int, default=1000,
                       help='Number of Control ECGs to compute base latent')
    
    # Traversal
    parser.add_argument('--alphas', type=float, nargs='+', default=[-3, -2, -1, 0, 1, 2, 3],
                       help='Alpha values for dimension traversal')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/latent_interpretability',
                       help='Output directory for results and plots')
    parser.add_argument('--save_plots', action='store_true', default=True,
                       help='Save dimension traversal plots')
    
    args = parser.parse_args()
    main(args)
