"""
Quick sanity check: Do Dyson candidates score higher than normal stars?
Run this after fit_anom.py completes.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import TechnoNet.src.models as models
import matplotlib.pyplot as plt

TECHNONET_DIR = Path("TechnoNet")
DYSON_DIR = "Eleanor/sh_files/s0018" # Whatever directory
CHECKPOINT_PATH = TECHNONET_DIR / "models" / "best_TAE_model.pth"
METRICS_PATH = TECHNONET_DIR / "fitted_metrics" / "fitted_anomaly_metrics.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load fitted metrics
with open(METRICS_PATH, 'rb') as f:
    metrics = pickle.load(f)
print("✓ Loaded fitted anomaly metrics")

# Load TAE model
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model_config = checkpoint['model_config']

TAE = models.TemporalAutoEncoder(
    in_channels=model_config['in_channels'],
    channels=model_config['channels'],
    kernel_size=model_config['kernel_size'],
    reduced_channels=model_config['reduced_channels'],
    bottleneck_channels=model_config['bottleneck_channels'],
    latent_time=model_config['latent_time'],
    dropout=model_config['dropout'],
).to(device)

TAE.load_state_dict(checkpoint['model_state_dict'])
TAE.eval()
print("✓ Loaded TAE model")

# Load Dyson candidates (just first 50 for quick test)
dyson_files = sorted(DYSON_DIR.glob("*.npz"))
# Load Dyson candidates (just first 50 for quick test)
dyson_files = sorted(DYSON_DIR.glob("*.npz"))
dyson_lcs = []

for file_path in dyson_files[:2]:  # Just first 2 files
    data = np.load(file_path, allow_pickle=True)
    if 'light_curves' in data:
        for i in range(min(25, len(data['light_curves']))):
            lc = data['light_curves'][i]
            
            # Filter out invalid entries
            if isinstance(lc, np.ndarray):
                # Skip scalar arrays containing dicts/objects
                if lc.shape == () and lc.dtype == object:
                    continue
                # Skip object dtype arrays that aren't numeric
                if lc.dtype == object:
                    continue
                # Only keep 1D numeric arrays
                if lc.ndim == 1 and np.issubdtype(lc.dtype, np.number):
                    dyson_lcs.append(lc)
    else:
        for key in list(data.files)[:25]:
            lc = data[key]
            
            # Same filtering
            if isinstance(lc, np.ndarray):
                if lc.shape == () and lc.dtype == object:
                    continue
                if lc.dtype == object:
                    continue
                if lc.ndim == 1 and np.issubdtype(lc.dtype, np.number):
                    dyson_lcs.append(lc)

print(f"✓ Loaded {len(dyson_lcs)} valid Dyson candidates")

# Debug: Check what we actually loaded
print("\n" + "="*60)
print("DATA INSPECTION")
print("="*60)
for i in range(min(3, len(dyson_lcs))):
    lc = dyson_lcs[i]
    print(f"LC {i}: type={type(lc)}, dtype={lc.dtype if hasattr(lc, 'dtype') else 'N/A'}, shape={lc.shape if hasattr(lc, 'shape') else 'N/A'}")
    if hasattr(lc, 'dtype') and lc.dtype == object:
        print(f"  -> Object content type: {type(lc.flat[0])}")
print("="*60 + "\n")

class SimpleDataset(Dataset):
    def __init__(self, lcs):
        self.lcs = lcs
    def __len__(self):
        return len(self.lcs)
    def __getitem__(self, idx):
        lc = self.lcs[idx].astype(np.float32)
        return torch.from_numpy(lc).unsqueeze(0)
        
def collate_fn_pad(batch):
    max_len = max(x.shape[-1] for x in batch)
    padded = []
    for x in batch:
        if x.shape[-1] < max_len:
            padding = torch.zeros(x.shape[0], max_len - x.shape[-1])
            x = torch.cat([x, padding], dim=-1)
        padded.append(x)
    return torch.stack(padded, dim=0)

dyson_dataset = SimpleDataset(dyson_lcs)
dyson_loader = DataLoader(dyson_dataset, batch_size=32, collate_fn=collate_fn_pad)

# Compute anomaly scores
dyson_scores = []

with torch.no_grad():
    for batch in dyson_loader:
        batch = batch.to(device)
        recon, z, _, _ = TAE(batch)
        
        embeddings = z.cpu().numpy()   
        recon_errors = torch.abs(recon - batch).mean(dim=(1, 2)).cpu().numpy()
        
        md = metrics.compute_mahalanobis_distance(embeddings)
        norm_md, norm_recon = metrics.normalize_scores(md, recon_errors)
        score_metrics = metrics.combine_signals(norm_md, norm_recon)
        
        dyson_scores.extend(score_metrics['score'])

dyson_scores = np.array(dyson_scores)

# Results
print("\n" + "="*60)
print("ANOMALY SCORE COMPARISON")
print("="*60)
print(f"Normal stars (training): mean=0.00, std=1.00")
print(f"Dyson candidates:        mean={dyson_scores.mean():.2f}, std={dyson_scores.std():.2f}")
print(f"                         min={dyson_scores.min():.2f}, max={dyson_scores.max():.2f}")
print(f"\nDyson scores are {dyson_scores.mean():.1f}σ above normal!")
print("="*60)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(dyson_scores, bins=20, alpha=0.7, label='Dyson Candidates', color='red')
plt.axvline(0, color='blue', linestyle='--', label='Normal Mean (0.0)')
plt.axvline(3, color='orange', linestyle='--', label='3σ Threshold')
plt.xlabel('Anomaly Score')
plt.ylabel('Count')
plt.title('TAE Anomaly Scores: Dyson Candidates vs Normal Distribution')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('tae_sanity_check.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved plot to tae_sanity_check.png")