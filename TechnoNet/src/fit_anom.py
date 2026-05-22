import pickle as pkl 
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pickle
from TechnoNet.src.metrics import AnomolyMetrics
import TechnoNet.src.models as models

TECHNONET_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = TECHNONET_DIR / "cached_lc_train"
CHECKPOINT_PATH = TECHNONET_DIR / "models" / "best_TAE_model.pth"
OUTPUT_DIR = TECHNONET_DIR / "fitted_metrics"

# Hyperparameters (must match training)
BATCH_SIZE = 128

# AnomalyMetrics configuration (Thill 2021 defaults)
COMBINE_WEIGHT = 0.5  # Weight between Mahalanobis (0.5) and Reconstruction (0.5)
REDUCTION = "mean"  # How to reduce reconstruction error
FLATTEN_MODE = "mean"  # How to flatten latent embedding: "mean" or "flatten"
NORMALIZE = "robust"  # Normalization: "robust" (median/MAD) or "zscore"

class TAE_TESS_Dataset(Dataset):
    """Dataset wrapper for TESS light curves."""
    
    def __init__(self, light_curves):
        self.light_curves = light_curves
        print(f"\nDataset Statistics:")
        print(f"  Total light curves: {len(light_curves)}")
        lengths = [len(lc) for lc in light_curves]
        print(f"  Length - min: {min(lengths)}, max: {max(lengths)}, mean: {np.mean(lengths):.1f}")
    
    def __len__(self):
        return len(self.light_curves)
    
    def __getitem__(self, idx):
        lc = torch.tensor(self.light_curves[idx], dtype=torch.float32)
        return lc.unsqueeze(0)  # Add channel dimension: [1, time_steps]

def collate_fn_pad(batch):
    """Pad sequences to the same length in a batch."""
    max_len = max(x.shape[-1] for x in batch)
    padded_batch = []
    for x in batch:
        if x.shape[-1] < max_len:
            padding = torch.zeros(x.shape[0], max_len - x.shape[-1])
            x = torch.cat([x, padding], dim=-1)
        padded_batch.append(x)
    return torch.stack(padded_batch, dim=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # ============================================
    # Load Training Data
    # ============================================

    print(f"\n{'='*60}")
    print("LOADING TRAINING DATA")
    print(f"{'='*60}\n")

    cache_files = [f for f in CACHE_DIR.iterdir() if f.suffix == ".npz"]

    if not cache_files:
        raise RuntimeError(
            f"No cached training data found in {CACHE_DIR}!\n"
            f"Run train.py first to generate training data."
        )

    normalized_light_curves = []

    for cache_file in sorted(cache_files):
        data = np.load(cache_file)
        sector_lcs = [data[key] for key in data.files]
        normalized_light_curves.extend(sector_lcs)
        print(f"  ✓ Loaded {len(sector_lcs)} LCs from {cache_file.name}")

    print(f"\nTotal loaded: {len(normalized_light_curves)} light curves\n")

    # Create dataset and loader
    dataset = TAE_TESS_Dataset(normalized_light_curves)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Don't shuffle for consistency
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn_pad,
    )

    # ============================================
    # Load Trained TAE Model
    # ============================================

    print(f"{'='*60}")
    print("LOADING TRAINED TAE MODEL")
    print(f"{'='*60}\n")

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            f"TAE checkpoint not found at {CHECKPOINT_PATH}!\n"
            f"Train the TAE model first using train.py with TAE_TRAIN=True"
        )

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model_config = checkpoint['model_config']

    print(f"Checkpoint info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint.get('val_loss', checkpoint.get('loss', 'N/A')):.6f}")
    print(f"  Latent time steps: {model_config['latent_time']}")
    print(f"  Bottleneck channels: {model_config['bottleneck_channels']}")

    # Initialize model with saved config
    TAE = models.TemporalAutoEncoder(
        in_channels=model_config['in_channels'],
        channels=model_config['channels'],
        kernel_size=model_config['kernel_size'],
        reduced_channels=model_config['reduced_channels'],
        bottleneck_channels=model_config['bottleneck_channels'],
        latent_time=model_config['latent_time'],
        dropout=model_config['dropout'],
    ).to(device)

    # Load trained weights
    TAE.load_state_dict(checkpoint['model_state_dict'])
    TAE.eval()  # Set to evaluation mode

    print(f"\n✓ TAE model loaded successfully\n")

    # ============================================
    # Extract Latent Embeddings & Reconstruction Errors
    # ============================================

    print(f"{'='*60}")
    print("EXTRACTING LATENT EMBEDDINGS")
    print(f"{'='*60}\n")

    all_embeddings = []
    all_recon_errors = []
    # Only collect examples from first batch
    originals_examples = None
    reconstructions_examples = None

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            
            # Forward pass through TAE
            recon, z, _, _ = TAE(batch)
            
            # Store latent embeddings - NEED ALL OF THESE
            all_embeddings.append(z.cpu().numpy())
            
            # Compute reconstruction error per sample - NEED ALL OF THESE
            recon_error = torch.abs(recon - batch).mean(dim=(1, 2))  # [B]
            all_recon_errors.append(recon_error.cpu().numpy())
            
            # Only save first batch for visual inspection examples
            if i == 0:
                originals_examples = batch.cpu().numpy()
                reconstructions_examples = recon.cpu().numpy()
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {(i+1)*BATCH_SIZE}/{len(dataset)} samples...")

    # Concatenate all batches - THIS IS WHAT YOU USE FOR FITTING
    embeddings = np.concatenate(all_embeddings, axis=0)  # [10000, 32, 4]
    recon_errors = np.concatenate(all_recon_errors, axis=0)  # [10000]

    print(f"\n✓ Extraction complete:")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Reconstruction errors shape: {recon_errors.shape}")
    print(f"  Mean reconstruction error: {recon_errors.mean():.6f}")
    print(f"  Std reconstruction error: {recon_errors.std():.6f}\n")

    # ============================================
    # Fit AnomalyMetrics to Normal Distribution
    # ============================================

    print(f"{'='*60}")
    print("FITTING ANOMALY METRICS")
    print(f"{'='*60}\n")

    # Initialize AnomalyMetrics
    metrics = AnomolyMetrics(
        combine_weight=COMBINE_WEIGHT,
        reduction=REDUCTION,
        flatten_mode=FLATTEN_MODE,
        normalize=NORMALIZE,
    )

    print(f"AnomalyMetrics configuration:")
    print(f"  Combine weight (Mahal/Recon): {COMBINE_WEIGHT}/{1-COMBINE_WEIGHT}")
    print(f"  Flatten mode: {FLATTEN_MODE}")
    print(f"  Normalization: {NORMALIZE}")
    print(f"  Reduction: {REDUCTION}\n")

    # Fit to normal distribution
    # This computes:
    # 1. Mean vector and covariance matrix of latent embeddings
    # 2. Statistics for Mahalanobis distances
    # 3. Statistics for reconstruction errors

    md_stats, recon_stats = metrics.fit(
        embedding=embeddings,
        recon_errors=recon_errors
    )

    print(f"✓ Fitting complete!\n")

    print(f"Mahalanobis Distance Statistics:")
    if NORMALIZE == "robust":
        print(f"  Median: {md_stats['median']:.6f}")
        print(f"  MAD: {md_stats['mean_abs_deviation']:.6f}")
    else:
        print(f"  Mean: {md_stats['mean']:.6f}")
        print(f"  Std: {md_stats['std']:.6f}")

    print(f"\nReconstruction Error Statistics:")
    if NORMALIZE == "robust":
        print(f"  Median: {recon_stats['median']:.6f}")
        print(f"  MAD: {recon_stats['mean_abs_deviation']:.6f}")
    else:
        print(f"  Mean: {recon_stats['mean']:.6f}")
        print(f"  Std: {recon_stats['std']:.6f}")

    # ============================================
    # Verify Fitting with Test Scores
    # ============================================

    print(f"\n{'='*60}")
    print("VERIFYING FITTED METRICS")
    print(f"{'='*60}\n")

    # Compute Mahalanobis distances for training data
    test_md = metrics.compute_mahalanobis_distance(embeddings)

    # Normalize scores
    norm_md, norm_recon = metrics.normalize_scores(test_md, recon_errors)

    # Combine into final anomaly scores
    score_metrics = metrics.combine_signals(norm_md, norm_recon)
    anomaly_scores = score_metrics['score']

    print(f"Anomaly Scores on Training Data (should be ~N(0,1)):")
    print(f"  Mean: {anomaly_scores.mean():.4f} (expect ~0)")
    print(f"  Std: {anomaly_scores.std():.4f} (expect ~1)")
    print(f"  Min: {anomaly_scores.min():.4f}")
    print(f"  Max: {anomaly_scores.max():.4f}")
    print(f"  Median: {np.median(anomaly_scores):.4f}")

    # Test adaptive thresholding
    k_values = [2, 3, 4, 5]
    print(f"\nAdaptive Thresholding Results:")
    for k in k_values:
        flags = metrics.adaptive_thresholding(k, score_metrics)
        pct_flagged = 100 * flags.sum() / len(flags)
        print(f"  k={k}: {flags.sum()}/{len(flags)} flagged ({pct_flagged:.2f}%)")

    # ============================================
    # Save Fitted Metrics
    # ============================================

    print(f"\n{'='*60}")
    print("SAVING FITTED METRICS")
    print(f"{'='*60}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the fitted AnomalyMetrics object
    metrics_path = OUTPUT_DIR / "fitted_anomaly_metrics.pkl"
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)

    print(f"✓ Saved fitted metrics to: {metrics_path}")

    # Also save as a dictionary for inspection
    metrics_dict = {
        'config': {
            'combine_weight': COMBINE_WEIGHT,
            'reduction': REDUCTION,
            'flatten_mode': FLATTEN_MODE,
            'normalize': NORMALIZE,
        },
        'fitted_params': {
            'mean_vec': metrics.mean_vec,
            'cov_inv': metrics.cov_inv,
            'md_stats': metrics.md_stats,
            'recon_errors_stats': metrics.recon_errors_stats,
        },
        'tae_config': model_config,
        'checkpoint_epoch': checkpoint['epoch'],
        'num_training_samples': len(embeddings),
    }

    dict_path = OUTPUT_DIR / "fitted_metrics_dict.npz"
    np.savez(
        dict_path,
        mean_vec=metrics.mean_vec,
        cov_inv=metrics.cov_inv,
        md_stats=np.array([md_stats]),
        recon_stats=np.array([recon_stats]),
        anomaly_scores_train=anomaly_scores,
    )

    print(f"✓ Saved metrics dictionary to: {dict_path}")

    # Save some example embeddings and scores for inspection
    examples_path = OUTPUT_DIR / "example_embeddings.npz"
    n_examples = min(100, len(embeddings))
    np.savez(
        examples_path,
        embeddings=embeddings[:n_examples],
        recon_errors=recon_errors[:n_examples],
        anomaly_scores=anomaly_scores[:n_examples],
        originals=originals_examples[:n_examples] if originals_examples is not None else None,
        reconstructions=reconstructions_examples[:n_examples] if reconstructions_examples is not None else None,
    )

    print(f"✓ Saved {n_examples} example embeddings to: {examples_path}")

    # ============================================
    # Summary
    # ============================================

    print(f"\n{'='*60}")
    print("FITTING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nFitted metrics saved in: {OUTPUT_DIR.relative_to(Path.cwd())}/")
    print(f"\nFiles created:")
    print(f"  1. fitted_anomaly_metrics.pkl    - Full fitted metrics object")
    print(f"  2. fitted_metrics_dict.npz       - Inspection/debugging")
    print(f"  3. example_embeddings.npz        - Sample data for testing")

    print(f"\nTo use at inference:")
    print(f"  import pickle")
    print(f"  with open('{metrics_path}', 'rb') as f:")
    print(f"      metrics = pickle.load(f)")
    print(f"  ")
    print(f"  # Then use metrics.compute_mahalanobis_distance(),")
    print(f"  # metrics.normalize_scores(), and metrics.combine_signals()")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
