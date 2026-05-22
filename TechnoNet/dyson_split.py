"""
Create train/val splits for TCN classifier training.

Combines:
- Normal light curves (label 0) from cached_lc_train/val
- Dyson positives (label 1) from dyson_positives/

Outputs:
- final_training_data_TCN_train/
- final_training_data_TCN_val/
"""

import numpy as np
from pathlib import Path
import random

# Configuration
TECHNONET_DIR = Path("TechnoNet")
NORMAL_TRAIN_DIR = TECHNONET_DIR / "cached_lc_train"
NORMAL_VAL_DIR = TECHNONET_DIR / "cached_lc_val"
DYSON_DIR = TECHNONET_DIR / "dyson_positives_new"

OUTPUT_TRAIN_DIR = TECHNONET_DIR / "final_training_data_TCN_train"
OUTPUT_VAL_DIR = TECHNONET_DIR / "final_training_data_TCN_val"

VAL_RATIO = 0.10  # 10% for validation
RANDOM_SEED = 42

# Downsampling targets
TARGET_NORMAL_TRAIN = 2250  # Aim for ~4-5:1 ratio with Dyson
TARGET_NORMAL_VAL = 500

# Create output directories
OUTPUT_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VAL_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("CREATING TCN TRAINING DATA (NORMAL + DYSON)")
print("="*70)
print(f"Normal train: {NORMAL_TRAIN_DIR}")
print(f"Normal val: {NORMAL_VAL_DIR}")
print(f"Dyson positives: {DYSON_DIR}")
print(f"\nOutput train: {OUTPUT_TRAIN_DIR}")
print(f"Output val: {OUTPUT_VAL_DIR}")
print(f"Validation ratio: {VAL_RATIO*100}%")
print(f"Random seed: {RANDOM_SEED}\n")

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ============================================
# 1. Load Normal Light Curves (Label 0)
# ============================================

print("="*70)
print("LOADING NORMAL LIGHT CURVES")
print("="*70)

normal_train_lcs = []
normal_val_lcs = []

# Load training normals
if NORMAL_TRAIN_DIR.exists():
    train_files = sorted(NORMAL_TRAIN_DIR.glob("*.npz"))
    print(f"\nFound {len(train_files)} files in {NORMAL_TRAIN_DIR.name}/")
    
    for data_file in train_files:
        print(f"  Loading {data_file.name}...")
        data = np.load(data_file)
        
        # Each .npz has multiple arrays (one per light curve)
        for key in data.files:
            lc = data[key]
            normal_train_lcs.append(lc)
        
        print(f"    Loaded {len(data.files)} light curves")
    
    print(f"\nTotal training normals: {len(normal_train_lcs)}")
else:
    print(f"WARNING: {NORMAL_TRAIN_DIR} not found!")

# Load validation normals
if NORMAL_VAL_DIR.exists():
    val_files = sorted(NORMAL_VAL_DIR.glob("*.npz"))
    print(f"\nFound {len(val_files)} files in {NORMAL_VAL_DIR.name}/")
    
    for data_file in val_files:
        print(f"  Loading {data_file.name}...")
        data = np.load(data_file)
        
        for key in data.files:
            lc = data[key]
            normal_val_lcs.append(lc)
        
        print(f"    Loaded {len(data.files)} light curves")
    
    print(f"\nTotal validation normals: {len(normal_val_lcs)}")
else:
    print(f"WARNING: {NORMAL_VAL_DIR} not found!")

# ============================================
# Downsample Normals for Better Class Balance
# ============================================

print(f"\n{'='*70}")
print("DOWNSAMPLING NORMALS")
print(f"{'='*70}")

print(f"\nOriginal normal data:")
print(f"  Train: {len(normal_train_lcs)}")
print(f"  Val: {len(normal_val_lcs)}")

if len(normal_train_lcs) > TARGET_NORMAL_TRAIN:
    print(f"\nDownsampling normal training data to {TARGET_NORMAL_TRAIN}...")
    normal_train_lcs = random.sample(normal_train_lcs, TARGET_NORMAL_TRAIN)

if len(normal_val_lcs) > TARGET_NORMAL_VAL:
    print(f"Downsampling normal validation data to {TARGET_NORMAL_VAL}...")
    normal_val_lcs = random.sample(normal_val_lcs, TARGET_NORMAL_VAL)

print(f"\nReduced normal data:")
print(f"  Train: {len(normal_train_lcs)}")
print(f"  Val: {len(normal_val_lcs)}")

# ============================================
# 2. Load Dyson Positives (Label 1)
# ============================================

print("\n" + "="*70)
print("LOADING DYSON POSITIVES")
print("="*70)

dyson_lcs = []

if DYSON_DIR.exists():
    dyson_files = sorted(DYSON_DIR.glob("*.npz"))
    print(f"\nFound {len(dyson_files)} files in {DYSON_DIR.name}/")
    
    for data_file in dyson_files:
        print(f"  Loading {data_file.name}...")
        
        # Always use allow_pickle=True for Dyson files since they may contain object arrays
        data = np.load(data_file, allow_pickle=True)
        
        # Check the structure of the file
        if 'flux' in data:
            # Direct flux key (standard Dyson file format)
            flux = data['flux']
            dyson_lcs.append(np.array(flux, dtype=np.float32))
            print(f"    Loaded 1 light curve (flux array, length {len(flux)})")
        elif 'light_curves' in data and 'labels' in data:
            # Already has structure - extract light curves only
            lcs = data['light_curves']
            # Handle both object arrays and regular arrays
            if lcs.dtype == object:
                for lc in lcs:
                    dyson_lcs.append(lc)
            else:
                for i in range(len(lcs)):
                    dyson_lcs.append(lcs[i])
            print(f"    Loaded {len(lcs)} light curves (pre-labeled)")
        else:
            print(f"    WARNING: Unknown file structure")
            print(f"    Available keys: {data.files}")
            print(f"    Skipping this file...")
    
    print(f"\nTotal Dyson positives: {len(dyson_lcs)}")
else:
    print(f"ERROR: {DYSON_DIR} not found!")
    print("Cannot proceed without Dyson positives.")
    exit(1)

# ============================================
# 3. Combine and Label
# ============================================

print("\n" + "="*70)
print("COMBINING DATA")
print("="*70)

# Combine all data
all_train_lcs = normal_train_lcs + dyson_lcs
all_train_labels = np.concatenate([
    np.zeros(len(normal_train_lcs), dtype=np.int32),  # Normal = 0
    np.ones(len(dyson_lcs), dtype=np.int32)           # Dyson = 1
])

print(f"\nCombined training pool:")
print(f"  Total samples: {len(all_train_lcs)}")
print(f"  Class 0 (normal): {np.sum(all_train_labels == 0)}")
print(f"  Class 1 (dyson): {np.sum(all_train_labels == 1)}")
print(f"  Class balance: {np.sum(all_train_labels == 1) / len(all_train_labels) * 100:.2f}% positive")

# ============================================
# 4. Stratified Split
# ============================================

print("\n" + "="*70)
print("SPLITTING INTO TRAIN/VAL")
print("="*70)

# Separate indices by class for stratification
normal_indices = np.where(all_train_labels == 0)[0]
dyson_indices = np.where(all_train_labels == 1)[0]

print(f"\nStratifying by class:")
print(f"  Normal indices: {len(normal_indices)}")
print(f"  Dyson indices: {len(dyson_indices)}")

# Shuffle each class independently
np.random.shuffle(normal_indices)
np.random.shuffle(dyson_indices)

# Split each class
normal_val_size = int(len(normal_indices) * VAL_RATIO)
dyson_val_size = int(len(dyson_indices) * VAL_RATIO)

normal_val_idx = normal_indices[:normal_val_size]
normal_train_idx = normal_indices[normal_val_size:]

dyson_val_idx = dyson_indices[:dyson_val_size]
dyson_train_idx = dyson_indices[dyson_val_size:]

# Combine train/val indices
final_train_idx = np.concatenate([normal_train_idx, dyson_train_idx])
final_val_idx = np.concatenate([normal_val_idx, dyson_val_idx])

# Shuffle final sets
np.random.shuffle(final_train_idx)
np.random.shuffle(final_val_idx)

print(f"\nFinal split:")
print(f"  Train: {len(final_train_idx)} samples")
print(f"    Class 0: {np.sum(all_train_labels[final_train_idx] == 0)}")
print(f"    Class 1: {np.sum(all_train_labels[final_train_idx] == 1)}")
print(f"  Val: {len(final_val_idx)} samples")
print(f"    Class 0: {np.sum(all_train_labels[final_val_idx] == 0)}")
print(f"    Class 1: {np.sum(all_train_labels[final_val_idx] == 1)}")

# ============================================
# 5. Create Final Datasets
# ============================================

print("\n" + "="*70)
print("CREATING FINAL DATASETS")
print("="*70)

# Extract train data
train_lcs = [all_train_lcs[i] for i in final_train_idx]
train_labels = all_train_labels[final_train_idx]

# Extract val data
val_lcs = [all_train_lcs[i] for i in final_val_idx]
val_labels = all_train_labels[final_val_idx]

# Add any pre-existing validation normals to val set
if len(normal_val_lcs) > 0:
    print(f"\nAdding {len(normal_val_lcs)} pre-existing validation normals to val set...")
    val_lcs.extend(normal_val_lcs)
    val_labels = np.concatenate([val_labels, np.zeros(len(normal_val_lcs), dtype=np.int32)])
    
    print(f"Updated val set:")
    print(f"  Total: {len(val_lcs)} samples")
    print(f"  Class 0: {np.sum(val_labels == 0)}")
    print(f"  Class 1: {np.sum(val_labels == 1)}")

# ============================================
# 6. Save
# ============================================

print("\n" + "="*70)
print("SAVING FINAL DATASETS")
print("="*70)

# Save training
train_file = OUTPUT_TRAIN_DIR / "tcn_train_data.npz"
print(f"\nSaving training data to {train_file}...")

# Convert list to object array to handle variable-length light curves
train_lcs_array = np.array(train_lcs, dtype=object)

np.savez_compressed(
    train_file,
    light_curves=train_lcs_array,
    labels=train_labels
)
print(f"  ✓ Saved {len(train_lcs)} training samples")
print(f"    Class 0 (normal): {np.sum(train_labels == 0)}")
print(f"    Class 1 (dyson): {np.sum(train_labels == 1)}")

# Save validation
val_file = OUTPUT_VAL_DIR / "tcn_val_data.npz"
print(f"\nSaving validation data to {val_file}...")

# Convert list to object array to handle variable-length light curves
val_lcs_array = np.array(val_lcs, dtype=object)

np.savez_compressed(
    val_file,
    light_curves=val_lcs_array,
    labels=val_labels
)
print(f"  ✓ Saved {len(val_lcs)} validation samples")
print(f"    Class 0 (normal): {np.sum(val_labels == 0)}")
print(f"    Class 1 (dyson): {np.sum(val_labels == 1)}")

# ============================================
# Summary
# ============================================

print("\n" + "="*70)
print("SPLIT COMPLETE")
print("="*70)
print(f"\nDatasets created:")
print(f"  Train: {len(train_lcs)} samples ({np.sum(train_labels == 1) / len(train_labels) * 100:.2f}% positive)")
print(f"  Val: {len(val_lcs)} samples ({np.sum(val_labels == 1) / len(val_labels) * 100:.2f}% positive)")

print(f"\nClass balance (Train):")
print(f"  Normal/Dyson ratio: {np.sum(train_labels == 0) / np.sum(train_labels == 1):.1f}:1")

print(f"\nFiles saved:")
print(f"  {train_file}")
print(f"  {val_file}")

print(f"\nNext step:")
print(f"  Set TCN_TRAIN = True in train.py and run training")
print("="*70)