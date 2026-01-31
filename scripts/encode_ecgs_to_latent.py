"""
Encode all ECG signals to VAE latent embeddings.

This script:
1. Loads the trained VAE model
2. Encodes all ECG signals in the dataset to latent vectors (z_ecg_1 to z_ecg_64)
3. Saves embeddings to parquet file for Phase F

Output: models/vae_latent_embeddings.parquet
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.vae_conv1d import Conv1DVAE
from src.data.ecg_dataset import ECGDataset

# Paths
CHECKPOINT_PATH = "models/checkpoints/vae_zdim64_beta4.0_20251102_073456/best_model.pt"
METADATA_PATH = "data/processed/ecg_features_with_demographics.parquet"
BASE_PATH = "data/raw/MIMIC-IV-ECG-1.0/files"
OUTPUT_PATH = "models/vae_latent_embeddings.parquet"

# Model parameters
Z_DIM = 64
BETA = 4.0
BATCH_SIZE = 64

def collate_fn_filter_none(batch):
    """Filter out None values from batch (failed ECG loads)."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    # Separate signals and metadata
    signals = torch.stack([item[0] for item in batch])
    metadata = [item[1] for item in batch]
    return signals, metadata

def main():
    print("\n" + "=" * 80)
    print("VAE Latent Embedding Generation")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    print(f"\nLoading VAE checkpoint from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    
    model = Conv1DVAE(z_dim=Z_DIM, beta=BETA).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded")
    
    # Load metadata
    print(f"\nLoading metadata from {METADATA_PATH}...")
    df_metadata = pd.read_parquet(METADATA_PATH)
    print(f"✓ Loaded {len(df_metadata)} ECG records")
    
    # Create dataset and dataloader
    print(f"\nCreating dataset...")
    dataset = ECGDataset(df_metadata, BASE_PATH, normalize=True)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_filter_none
    )
    
    # Encode all ECGs
    print(f"\nEncoding ECGs to latent space (z_dim={Z_DIM})...")
    
    all_record_ids = []
    all_embeddings = []
    failed_records = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding batches"):
            if batch is None:
                continue
            
            signals, metadata = batch
            
            # Get record IDs from metadata indices
            indices = [m['idx'] for m in metadata]
            record_ids = df_metadata.iloc[indices]['record_id'].tolist()
            
            try:
                # Move to device
                signals = signals.to(device)
                
                # Get latent embeddings (mean of posterior)
                z = model.get_latent_embeddings(signals)
                
                # Store
                all_record_ids.extend(record_ids)
                all_embeddings.append(z.cpu().numpy())
                
            except Exception as e:
                print(f"\n⚠ Error encoding batch: {e}")
                failed_records.extend(record_ids)
                continue
    
    # Concatenate all embeddings
    if len(all_embeddings) > 0:
        all_embeddings = np.vstack(all_embeddings)
    else:
        print("\n✗ No embeddings generated!")
        return
    
    print(f"\n✓ Encoded {len(all_record_ids)} records")
    if failed_records:
        print(f"⚠ Failed to encode {len(failed_records)} records")
    
    # Create DataFrame
    print("\nCreating embeddings DataFrame...")
    
    # Create column names for latent dimensions
    z_columns = [f'z_ecg_{i+1}' for i in range(Z_DIM)]
    
    df_embeddings = pd.DataFrame(
        all_embeddings,
        columns=z_columns
    )
    
    # Add record_id
    df_embeddings.insert(0, 'record_id', all_record_ids)
    
    # Save
    print(f"\nSaving embeddings to {OUTPUT_PATH}...")
    df_embeddings.to_parquet(OUTPUT_PATH, index=False)
    
    print(f"\n✓ Embeddings saved!")
    print(f"  Records: {len(df_embeddings)}")
    print(f"  Latent dims: {Z_DIM}")
    print(f"  Shape: {df_embeddings.shape}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Embedding Statistics")
    print("=" * 80)
    
    print(f"\nLatent space statistics (first 10 dimensions):")
    for i in range(min(10, Z_DIM)):
        col = f'z_ecg_{i+1}'
        mean = df_embeddings[col].mean()
        std = df_embeddings[col].std()
        min_val = df_embeddings[col].min()
        max_val = df_embeddings[col].max()
        print(f"  {col:12s}: mean={mean:7.3f}, std={std:6.3f}, min={min_val:7.3f}, max={max_val:7.3f}")
    
    if Z_DIM > 10:
        print(f"  ... and {Z_DIM-10} more dimensions")
    
    print("\n" + "=" * 80)
    print("✓ Complete!")
    print("=" * 80)
    print(f"\nOutput: {OUTPUT_PATH}")
    print("Ready for Phase F: Master Dataset creation")

if __name__ == "__main__":
    main()
