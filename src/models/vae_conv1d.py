"""
FIXED Conv1D-based β-VAE for 12-lead ECG signals.

Key Fixes for Posterior Collapse:
1. Free bits constraint (minimum KL per dimension)
2. Proper reconstruction loss scaling
3. More gradual architecture compression
4. Improved numerical stability

Usage:
    Drop this file into src/models/vae_conv1d.py (replace existing)
    Then run training with: --beta 4.0 --epochs 100 --batch_size 64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class Conv1DVAE(nn.Module):
    """
    Fixed β-VAE with solutions for posterior collapse.
    
    Parameters
    ----------
    z_dim : int, default=64
        Latent dimension size
    beta : float, default=4.0
        β-VAE parameter for disentanglement
    free_bits : float, default=2.0
        Minimum KL divergence per latent dimension (prevents collapse)
    """
    
    def __init__(self, z_dim: int = 64, beta: float = 4.0, free_bits: float = 2.0):
        super(Conv1DVAE, self).__init__()
        self.z_dim = z_dim
        self.beta = beta
        self.free_bits = free_bits
        
        # ========== ENCODER (Improved gradual compression) ==========
        # Input: (batch, 12, 5000)
        self.encoder = nn.Sequential(
            # Layer 1: (batch, 12, 5000) → (batch, 32, 2500)
            nn.Conv1d(in_channels=12, out_channels=32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),  # LeakyReLU prevents dead neurons
            
            # Layer 2: (batch, 32, 2500) → (batch, 64, 1250)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=10, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            # Layer 3: (batch, 64, 1250) → (batch, 128, 625)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )
        
        # Flatten: (batch, 128, 625) → (batch, 80000)
        self.flatten = nn.Flatten()
        
        # More gradual FC compression: 80000 → 512 → 256
        self.fc_encoder = nn.Sequential(
            nn.Linear(in_features=128 * 625, out_features=512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),  # Regularization
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(0.2),
        )
        
        # Latent space projection: μ and log(σ²)
        self.fc_mu = nn.Linear(in_features=256, out_features=z_dim)
        self.fc_logvar = nn.Linear(in_features=256, out_features=z_dim)
        
        # ========== DECODER ==========
        self.fc_decoder = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=256),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=512, out_features=128 * 625),
            nn.LeakyReLU(0.2),
        )
        
        # Unflatten: (batch, 80000) → (batch, 128, 625)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 625))
        
        self.decoder = nn.Sequential(
            # Layer 1: (batch, 128, 625) → (batch, 64, 1250)
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=5, 
                              stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            # Layer 2: (batch, 64, 1250) → (batch, 32, 2500)
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=10,
                              stride=2, padding=4, output_padding=0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            # Layer 3: (batch, 32, 2500) → (batch, 12, 5000)
            nn.ConvTranspose1d(in_channels=32, out_channels=12, kernel_size=15,
                              stride=2, padding=7, output_padding=1),
            # No activation - reconstruct in original signal space
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input ECG to latent distribution parameters."""
        h = self.encoder(x)          # (batch, 128, 625)
        h = self.flatten(h)           # (batch, 80000)
        h = self.fc_encoder(h)        # (batch, 256)
        
        mu = self.fc_mu(h)            # (batch, z_dim)
        logvar = self.fc_logvar(h)    # (batch, z_dim)
        
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed ECG."""
        h = self.fc_decoder(z)        # (batch, 80000)
        h = self.unflatten(h)         # (batch, 128, 625)
        x_recon = self.decoder(h)     # (batch, 12, 5000)
        return x_recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def loss_function(self, x: torch.Tensor, x_recon: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Improved β-VAE loss with free bits constraint.
        
        Key improvements:
        1. Reconstruction loss scaled by signal variance
        2. Free bits: only penalize KL above threshold
        3. Track raw KL (before free bits) for monitoring
        """
        batch_size = x.size(0)
        
        # 1. Reconstruction loss (MSE scaled by signal variance)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size
        
        # Scale by empirical variance to balance with KL term
        signal_var = x.var() + 1e-8
        recon_loss = recon_loss / signal_var
        
        # 2. KL divergence per dimension
        # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        # Track raw KL (before free bits) for monitoring
        kl_raw = kl_per_dim.sum(dim=1).mean()
        
        # Apply free bits constraint: only penalize KL above threshold
        # This prevents posterior collapse by ensuring minimum information
        kl_per_dim = torch.clamp(kl_per_dim - self.free_bits, min=0.0)
        kl_loss = kl_per_dim.sum(dim=1).mean()
        
        # 3. Total loss with β weighting
        loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'kl_raw': kl_raw  # For monitoring - should be > 0!
        }
    
    def get_latent_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent mean vectors (deterministic, for saving embeddings)."""
        mu, _ = self.encode(x)
        return mu


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model architecture
    print("=" * 80)
    print("Fixed Conv1D β-VAE Architecture Test")
    print("=" * 80)
    
    # Initialize model
    model = Conv1DVAE(z_dim=64, beta=4.0, free_bits=2.0)
    print(f"\n✓ Model initialized with z_dim=64, β=4.0, free_bits=2.0")
    print(f"✓ Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 12, 5000)
    print(f"\n✓ Input shape: {tuple(x.shape)}")
    
    x_recon, mu, logvar = model(x)
    print(f"✓ Reconstructed shape: {tuple(x_recon.shape)}")
    print(f"✓ Latent mean shape: {tuple(mu.shape)}")
    print(f"✓ Latent logvar shape: {tuple(logvar.shape)}")
    
    # Test loss computation
    loss_dict = model.loss_function(x, x_recon, mu, logvar)
    print(f"\n✓ Loss: {loss_dict['loss'].item():.4f}")
    print(f"✓ Reconstruction loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"✓ KL loss (with free bits): {loss_dict['kl_loss'].item():.4f}")
    print(f"✓ KL raw (without free bits): {loss_dict['kl_raw'].item():.4f}")
    print(f"  → KL raw should be > 0 (not collapsed!)")
    
    # Test latent embedding extraction
    embeddings = model.get_latent_embeddings(x)
    print(f"\n✓ Latent embeddings shape: {tuple(embeddings.shape)}")
    
    print("\n" + "=" * 80)
    print("✓ All architecture tests passed!")
    print("=" * 80)