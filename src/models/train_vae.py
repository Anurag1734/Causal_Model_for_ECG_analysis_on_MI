"""
CORRECTED Training script for Conv1D β-VAE on 12-lead ECG signals.

Key fixes:
1. Cyclical β-annealing (4 cycles of 40 epochs)
2. Proper KL raw tracking
3. Updated history tracking

Run with:
    python src/models/train_vae.py --batch_size 64 --epochs 160 --lr 1e-4
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.vae_conv1d import Conv1DVAE, count_parameters
from data.ecg_dataset import ECGDataset, create_train_val_test_splits, create_dataloaders


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, epoch: int) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        
        return self.early_stop


class VAETrainer:
    """Trainer for Conv1D β-VAE with cyclical annealing."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        output_dir: str,
        early_stopping_patience: int = 10
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
        
        # Cyclical β-annealing parameters
        self.beta_max = self.model.beta
        self.beta_cycle_length = 40  # One cycle: 0→beta_max over 40 epochs
        self.n_cycles = 4  # Total of 4 cycles (160 epochs)
        
        # History tracking (FIXED: includes kl_raw)
        self.history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_kl_loss': [],
            'train_kl_raw': [],  # Raw KL (before free bits)
            'val_loss': [],
            'val_recon_loss': [],
            'val_kl_loss': [],
            'val_kl_raw': [],  # Raw KL (before free bits)
            'learning_rate': [],
            'beta': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def get_beta(self, epoch: int) -> float:
        """Get current β value with CYCLICAL annealing."""
        total_anneal_epochs = self.beta_cycle_length * self.n_cycles
        
        if epoch >= total_anneal_epochs:
            return self.beta_max
        
        # Which point in the current cycle?
        epoch_in_cycle = epoch % self.beta_cycle_length
        
        # Linear ramp within cycle: 0 → beta_max
        progress = epoch_in_cycle / self.beta_cycle_length
        return self.beta_max * progress
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        # Get current β value (cyclical annealing)
        current_beta = self.get_beta(epoch)
        
        # Initialize accumulators
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_kl_raw = 0.0  # FIXED: Initialize here
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train] (β={current_beta:.2f})")
        for batch_idx, (signals, metadata) in enumerate(pbar):
            # Skip empty batches (all corrupted files)
            if signals is None:
                continue
                
            # Move to device
            signals = signals.to(self.device)
            
            # Forward pass
            x_recon, mu, logvar = self.model(signals)
            
            # Compute loss with current β
            loss_dict = self.model.loss_function(signals, x_recon, mu, logvar)
            # Override β in loss calculation (for cyclical annealing)
            recon_loss = loss_dict['recon_loss']
            kl_loss = loss_dict['kl_loss']
            loss = recon_loss + current_beta * kl_loss
            
            # Check for NaN before backward
            if torch.isnan(loss):
                print(f"\nWARNING: NaN detected at batch {batch_idx}")
                print(f"  Signal stats: min={signals.min():.4f}, max={signals.max():.4f}, mean={signals.mean():.4f}")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # Track metrics (FIXED: track kl_raw inside loop)
            total_loss += loss.item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
            total_kl_raw += loss_dict['kl_raw'].item()  # FIXED: Track raw KL
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{loss_dict['recon_loss'].item():.4f}",
                'kl': f"{loss_dict['kl_loss'].item():.4f}",
                'kl_raw': f"{loss_dict['kl_raw'].item():.4f}"  # Show raw KL
            })
        
        # Average metrics
        avg_loss = total_loss / n_batches if n_batches > 0 else float('inf')
        avg_recon_loss = total_recon_loss / n_batches if n_batches > 0 else float('inf')
        avg_kl_loss = total_kl_loss / n_batches if n_batches > 0 else float('inf')
        avg_kl_raw = total_kl_raw / n_batches if n_batches > 0 else float('inf')
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss,
            'kl_raw': avg_kl_raw  # FIXED: Return raw KL
        }
    
    def validate(self, epoch: int) -> dict:
        """Validate on validation set."""
        self.model.eval()
        
        # Initialize accumulators
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_kl_raw = 0.0  # FIXED: Initialize here
        n_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
            for signals, metadata in pbar:
                # Skip empty batches
                if signals is None:
                    continue
                    
                # Move to device
                signals = signals.to(self.device)
                
                # Forward pass
                x_recon, mu, logvar = self.model(signals)
                
                # Compute loss
                loss_dict = self.model.loss_function(signals, x_recon, mu, logvar)
                
                # Track metrics (FIXED: track kl_raw inside loop)
                total_loss += loss_dict['loss'].item()
                total_recon_loss += loss_dict['recon_loss'].item()
                total_kl_loss += loss_dict['kl_loss'].item()
                total_kl_raw += loss_dict['kl_raw'].item()  # FIXED: Track raw KL
                n_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['loss'].item():.4f}",
                    'recon': f"{loss_dict['recon_loss'].item():.4f}",
                    'kl': f"{loss_dict['kl_loss'].item():.4f}",
                    'kl_raw': f"{loss_dict['kl_raw'].item():.4f}"
                })
        
        # Average metrics
        avg_loss = total_loss / n_batches
        avg_recon_loss = total_recon_loss / n_batches
        avg_kl_loss = total_kl_loss / n_batches
        avg_kl_raw = total_kl_raw / n_batches
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss,
            'kl_raw': avg_kl_raw  # FIXED: Return raw KL
        }
    
    def train(self, num_epochs: int, start_epoch: int = 0):
        """
        Train for multiple epochs.
        
        Args:
            num_epochs: Total number of epochs to train
            start_epoch: Epoch to start/resume from (0 for new training)
        """
        print("\n" + "=" * 80)
        print(f"Starting training for {num_epochs} epochs (from epoch {start_epoch+1})")
        print(f"Cyclical β-annealing: {self.n_cycles} cycles × {self.beta_cycle_length} epochs")
        print("=" * 80)
        
        for epoch in range(start_epoch, num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update learning rate scheduler
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            current_beta = self.get_beta(epoch)
            
            # Log metrics (FIXED: include kl_raw)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon_loss'].append(train_metrics['recon_loss'])
            self.history['train_kl_loss'].append(train_metrics['kl_loss'])
            self.history['train_kl_raw'].append(train_metrics['kl_raw'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_recon_loss'].append(val_metrics['recon_loss'])
            self.history['val_kl_loss'].append(val_metrics['kl_loss'])
            self.history['val_kl_raw'].append(val_metrics['kl_raw'])
            self.history['learning_rate'].append(current_lr)
            self.history['beta'].append(current_beta)
            
            # Print summary (FIXED: show raw KL for monitoring)
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(Recon: {train_metrics['recon_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f}, "
                  f"KL_raw: {train_metrics['kl_raw']:.4f})")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} "
                  f"(Recon: {val_metrics['recon_loss']:.4f}, KL: {val_metrics['kl_loss']:.4f}, "
                  f"KL_raw: {val_metrics['kl_raw']:.4f})")
            print(f"  Learning Rate: {current_lr:.2e}, β: {current_beta:.2f}")
            
            # Check for posterior collapse
            if train_metrics['kl_raw'] < 1.0:
                print(f"  ⚠️  WARNING: Possible posterior collapse (KL_raw < 1.0)")
            
            # Save checkpoint if best
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best model! (Val Loss: {val_metrics['loss']:.4f})")
            
            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.early_stopping(val_metrics['loss'], epoch):
                print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
                print(f"✓ Best epoch was {self.best_epoch+1} with val_loss={self.best_val_loss:.4f}")
                break
        
        # Save final checkpoint and history
        self.save_checkpoint(epoch, is_best=False, suffix='_final')
        self.save_history()
        self.plot_training_curves()
        
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best model saved at epoch {self.best_epoch+1} with val_loss={self.best_val_loss:.4f}")
        print("=" * 80)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, suffix: str = ''):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'patience_counter': self.patience_counter
        }
        
        if is_best:
            path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, path)
        else:
            path = self.output_dir / f'checkpoint_epoch_{epoch+1}{suffix}.pt'
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and restore training state."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', -1)
        self.history = checkpoint.get('history', {
            'train_loss': [], 'train_recon_loss': [], 'train_kl_loss': [], 'train_kl_raw': [],
            'val_loss': [], 'val_recon_loss': [], 'val_kl_loss': [], 'val_kl_raw': [],
            'learning_rate': [], 'beta': []
        })
        self.patience_counter = checkpoint.get('patience_counter', 0)
        
        print(f"✓ Checkpoint loaded successfully!")
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Best val loss: {self.best_val_loss:.4f} (epoch {self.best_epoch+1})")
        print(f"  Early stopping counter: {self.patience_counter}/{self.early_stopping_patience}")
        
        return start_epoch
    
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\n✓ Training history saved to {history_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Total loss
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        axes[0, 1].plot(epochs, self.history['train_recon_loss'], label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_recon_loss'], label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Reconstruction Loss (MSE)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # KL loss (with free bits)
        axes[0, 2].plot(epochs, self.history['train_kl_loss'], label='Train', linewidth=2)
        axes[0, 2].plot(epochs, self.history['val_kl_loss'], label='Val', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('KL Divergence (with Free Bits)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # KL raw (without free bits) - CRITICAL FOR MONITORING
        axes[1, 0].plot(epochs, self.history['train_kl_raw'], label='Train', linewidth=2, color='red')
        axes[1, 0].plot(epochs, self.history['val_kl_raw'], label='Val', linewidth=2, color='darkred')
        axes[1, 0].axhline(y=1.0, color='black', linestyle='--', label='Collapse threshold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('KL Divergence')
        axes[1, 0].set_title('KL Raw (should be > 1.0)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Beta schedule (should show 4 cycles)
        axes[1, 1].plot(epochs, self.history['beta'], linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('β')
        axes[1, 1].set_title('β Schedule (Cyclical)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 2].plot(epochs, self.history['learning_rate'], linewidth=2, color='green')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_title('Learning Rate Schedule')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {plot_path}")
        plt.close()


def main(args):
    """Main training function."""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
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
    print(f"✓ Label distribution:\n{df['Label'].value_counts()}")
    
    # Filter for training
    train_labels = ['Control_Symptomatic', 'MI_Pre-Incident']
    df_train = df[df['Label'].isin(train_labels)].copy()
    print(f"\n✓ Training set: {len(df_train)} ECGs")
    print(f"✓ Training label distribution:\n{df_train['Label'].value_counts()}")
    
    # Create splits
    print(f"\n✓ Creating train/val/test splits (80/10/10)")
    train_df, val_df, test_df = create_train_val_test_splits(
        df_train,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        stratify_col='Label',
        random_state=args.seed
    )
    print(f"✓ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Create dataloaders
    print(f"\n✓ Creating dataloaders (batch_size={args.batch_size})")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        base_path=args.base_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=True
    )
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # Initialize model
    print(f"\n✓ Initializing Conv1D β-VAE (z_dim={args.z_dim}, β={args.beta}, free_bits=2.0)")
    model = Conv1DVAE(z_dim=args.z_dim, beta=args.beta, free_bits=2.0).to(device)
    print(f"✓ Total parameters: {count_parameters(model):,}")
    
    # Optimizer and scheduler
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True
    )
    
    print(f"\n✓ Optimizer: Adam (lr={args.lr:.2e}, weight_decay={args.weight_decay:.2e})")
    print(f"✓ Scheduler: ReduceLROnPlateau (patience={args.lr_patience}, factor={args.lr_factor})")
    print(f"✓ Early stopping patience: {args.early_stopping_patience}")
    
    # Create or determine output directory
    if args.resume and args.checkpoint_path:
        # When resuming, use the checkpoint's directory
        checkpoint_path = Path(args.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        output_dir = checkpoint_path.parent
        print(f"\n✓ Resuming training from: {checkpoint_path}")
        print(f"✓ Output directory: {output_dir}")
    else:
        # New training - create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"vae_zdim{args.z_dim}_beta{args.beta}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Output directory: {output_dir}")
    
    # Save configuration (only for new training)
    if not args.resume:
        config = vars(args)
        config['device'] = str(device)
        config['num_train'] = len(train_df)
        config['num_val'] = len(val_df)
        config['num_test'] = len(test_df)
        config['model_parameters'] = count_parameters(model)
        
        config_path = output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Configuration saved to {config_path}")
    
    # Train
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume and args.checkpoint_path:
        start_epoch = trainer.load_checkpoint(args.checkpoint_path)
    
    trainer.train(num_epochs=args.epochs, start_epoch=start_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conv1D β-VAE on 12-lead ECG signals")
    
    # Data
    parser.add_argument('--metadata_path', type=str, default='data/processed/ecg_features_with_demographics.parquet',
                       help='Path to metadata parquet file')
    parser.add_argument('--base_path', type=str, default='data/raw/MIMIC-IV-ECG-1.0/files',
                       help='Base path to WFDB files')
    
    # Model
    parser.add_argument('--z_dim', type=int, default=64,
                       help='Latent dimension size')
    parser.add_argument('--beta', type=float, default=4.0,
                       help='β-VAE parameter for disentanglement')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (reduce to 32/16 if OOM)')
    parser.add_argument('--epochs', type=int, default=160,
                       help='Maximum number of epochs (4 cycles × 40)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--lr_patience', type=int, default=10,
                       help='ReduceLROnPlateau patience')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                       help='ReduceLROnPlateau factor')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='Early stopping patience')
    
    # System
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='models/checkpoints',
                       help='Output directory for checkpoints')
    
    # Resume
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from a checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to checkpoint file for resuming training')
    
    args = parser.parse_args()
    main(args)