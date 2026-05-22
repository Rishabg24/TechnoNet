"""
Split cached TESS data into train/val sets.

Run this ONCE before training to create separate train and validation caches.
"""

import numpy as np
from pathlib import Path
import shutil

# Configuration
CACHE_DIR = Path("TechnoNet/cached_lc")
TRAIN_DIR = Path("TechnoNet/cached_lc_train")
VAL_DIR = Path("TechnoNet/cached_lc_val")

VAL_PER_SECTOR = 300  # Take 300 from each sector for validation
RANDOM_SEED = 42  # For reproducibility

# Create output directories
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("SPLITTING CACHED DATA INTO TRAIN/VAL SETS")
print("="*70)
print(f"Source: {CACHE_DIR}")
print(f"Train output: {TRAIN_DIR}")
print(f"Val output: {VAL_DIR}")
print(f"Validation samples per sector: {VAL_PER_SECTOR}")
print(f"Random seed: {RANDOM_SEED}\n")

np.random.seed(RANDOM_SEED)

total_train = 0
total_val = 0

# Process each sector file
cache_files = sorted(CACHE_DIR.glob("sector*_batch.npz"))

if len(cache_files) == 0:
    print(f"ERROR: No .npz files found in {CACHE_DIR}")
    exit(1)

print(f"Found {len(cache_files)} sector files\n")

for cache_file in cache_files:
    sector_name = cache_file.stem  # e.g., "sector5_batch"
    print(f"Processing {cache_file.name}...")
    
    # Load the data
    data = np.load(cache_file)
    all_lcs = [data[key] for key in data.files]
    print(f"  Total light curves: {len(all_lcs)}")
    
    # Shuffle indices
    indices = np.arange(len(all_lcs))
    np.random.shuffle(indices)
    
    # Split into val and train
    val_indices = indices[:VAL_PER_SECTOR]
    train_indices = indices[VAL_PER_SECTOR:]
    
    val_lcs = [all_lcs[i] for i in val_indices]
    train_lcs = [all_lcs[i] for i in train_indices]
    
    print(f"  Train: {len(train_lcs)} | Val: {len(val_lcs)}")
    
    # Save train set
    train_file = TRAIN_DIR / f"{sector_name}.npz"
    np.savez_compressed(
        train_file,
        **{f"lc_{j}": lc for j, lc in enumerate(train_lcs)}
    )
    
    # Save val set
    val_file = VAL_DIR / f"{sector_name}.npz"
    np.savez_compressed(
        val_file,
        **{f"lc_{j}": lc for j, lc in enumerate(val_lcs)}
    )
    
    total_train += len(train_lcs)
    total_val += len(val_lcs)
    
    print(f"  ✓ Saved to {train_file.name} and {val_file.name}\n")

print("="*70)
print("SPLIT COMPLETE")
print("="*70)
print(f"Total training samples: {total_train}")
print(f"Total validation samples: {total_val}")
print(f"Train/Val split: {total_train/(total_train+total_val)*100:.1f}% / {total_val/(total_train+total_val)*100:.1f}%")
print(f"\nNext steps:")
print(f"1. Update train.py to use TRAIN_DIR: CACHE_DIR = TECHNONET_DIR / 'cached_lc_train'")
print(f"2. Add validation loop using VAL_DIR: VAL_CACHE_DIR = TECHNONET_DIR / 'cached_lc_val'")
print("="*70)