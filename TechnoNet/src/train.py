import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from TechnoNet.data import pre
from TechnoNet.src.metrics import AnomolyMetrics
import TechnoNet.src.models as models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set matmul precision for better performance on modern GPUs
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")

# Get TechnoNet directory (where all ML data will be stored)
TECHNONET_DIR = Path(__file__).resolve().parent.parent
print(f"TechnoNet directory: {TECHNONET_DIR}\n")

# Training flags
TAE_TRAIN = False
TCN_TRAIN = True

# Fix: Corrected assertion logic
assert not (TAE_TRAIN and TCN_TRAIN), "Train one model at a time"

if TAE_TRAIN:
    print("\n" + "=" * 60)
    print("TRAINING TEMPORAL AUTOENCODER (TAE)")
    print("=" * 60 + "\n")

    # ================================
    # 1. Dataset Definition
    # ================================

    class TAE_TESS_Dataset(Dataset):
        """Dataset wrapper for TESS light curves."""

        def __init__(self, light_curves):
            self.light_curves = light_curves

            # Print statistics
            lengths = [len(lc) for lc in light_curves]
            print(f"\nDataset Statistics:")
            print(f"  Total light curves: {len(light_curves)}")
            print(
                f"  Length - min: {min(lengths)}, max: {max(lengths)}, mean: {np.mean(lengths):.1f}"
            )
            print(f"  Memory usage: ~{sum(lengths) * 4 / 1e6:.1f} MB\n")

        def __len__(self):
            return len(self.light_curves)

        def __getitem__(self, idx):
            lc = torch.tensor(self.light_curves[idx], dtype=torch.float32)
            return lc.unsqueeze(0)  # Add channel dimension: [1, time_steps]
        
    def collate_fn_pad(batch):
        """Pad sequences to the same length in a batch."""
        # Find max length in this batch
        max_len = max(x.shape[-1] for x in batch)
        
        # Pad each sequence
        padded_batch = []
        for x in batch:
            if x.shape[-1] < max_len:
                # Pad with zeros on the right
                padding = torch.zeros(x.shape[0], max_len - x.shape[-1])
                x = torch.cat([x, padding], dim=-1)
            padded_batch.append(x)
        
        return torch.stack(padded_batch, dim=0)

    # ================================
    # 2. Hyperparameters
    # ================================

    # Data loading - PRODUCTION MODE: WILL GET 10K LIGHT CURVES
    SECTORS = None  # None = auto-select 5 diverse sectors
    PER_SECTOR_TARGET = 2300  # Target per sector (expect ~2000 actual)
    TOTAL_TARGET = 10000  # HARD TARGET - script won't stop until reached
    BATCH_SIZE_DOWNLOAD = 15  # Download 15 at a time (optimal speed/reliability)

    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 20

    # Paths (all inside TechnoNet directory)
    CACHE_DIR = TECHNONET_DIR / "cached_lc_train"
    TEMP_FITS_DIR = TECHNONET_DIR / "temp_fits"
    SAVE_DIR = TECHNONET_DIR / "checkpoints"

    # Model architecture
    IN_CHANNELS = 1
    CHANNELS = [64, 64, 64]  # Encoder/decoder channel progression
    KERNEL_SIZE = 3
    REDUCED_CHANNELS = 16
    BOTTLENECK_CHANNELS = 32
    LATENT_TIME = 4
    DROPOUT = 0.2

    # ================================
    # 3. Load or Generate Data
    # ================================

    # Create directories inside TechnoNet
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_FITS_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Check if we already have cached NPZ data
    cache_files = [f for f in CACHE_DIR.iterdir() if f.suffix == ".npz"]

    if cache_files:
        print(f"{'='*60}")
        print(f"LOADING FROM CACHE")
        print(f"{'='*60}")
        print(
            f"Found {len(cache_files)} cached sector files in '{CACHE_DIR.relative_to(Path.cwd())}'"
        )
        print("Loading from cache (skipping MAST download)...\n")

        normalized_light_curves = []

        for cache_file in sorted(cache_files):
            data = np.load(cache_file)
            sector_lcs = [data[key] for key in data.files]
            normalized_light_curves.extend(sector_lcs)
            print(f"  ✓ Loaded {len(sector_lcs)} LCs from {cache_file.name}")

        print(f"\nTotal loaded: {len(normalized_light_curves)} light curves")
        print(f"Cache directory: {CACHE_DIR}")
        print(f"{'='*60}\n")

    else:
        print(f"{'='*60}")
        print(f"NO CACHE FOUND - DOWNLOADING FROM MAST")
        print(f"{'='*60}\n")

        # Use production-grade downloader with hard target
        normalized_light_curves = pre.load_data_from_mast(
            sectors=SECTORS,
            per_sector_target=PER_SECTOR_TARGET,
            total_target=TOTAL_TARGET,
            out_dir=str(CACHE_DIR),
            temp_download_dir=str(TEMP_FITS_DIR),
            batch_size=BATCH_SIZE_DOWNLOAD,
        )

        if len(normalized_light_curves) == 0:
            raise RuntimeError(
                "CRITICAL: Failed to download ANY light curves!\n"
                "Possible issues:\n"
                "  1. No internet connection\n"
                "  2. MAST service is down (check status.mast.stsci.edu)\n"
                "  3. astroquery not installed: pip install astroquery\n"
                "  4. Firewall blocking MAST (check GCE firewall rules)"
            )

        print(
            f"\n✓ Successfully downloaded and cached {len(normalized_light_curves)} light curves"
        )

        # Check if we need more data
        if len(normalized_light_curves) < TOTAL_TARGET * 0.8:
            print(
                f"\n⚠ WARNING: Only got {len(normalized_light_curves)}/{TOTAL_TARGET} light curves"
            )
            print(f"  Recommendation: Run script again with more sectors")
            print(f"  Add this to train.py to get more:")
            print(
                f"    SECTORS = [5, 6, 14, 18, 20, 1, 2, 15, 21, 24]  # 10 sectors for more data"
            )

            if len(normalized_light_curves) < 5000:
                raise RuntimeError(
                    f"Insufficient data: need at least 5000 light curves for reliable training, "
                    f"got {len(normalized_light_curves)}. Run again with more sectors."
                )

    # Verify we have enough data for reliable training
    min_required = 5000

    if len(normalized_light_curves) < min_required:
        raise RuntimeError(
            f"CRITICAL: Insufficient data for training!\n"
            f"  Got: {len(normalized_light_curves)} light curves\n"
            f"  Need: {min_required} minimum (10k-20k recommended)\n"
            f"\n"
            f"SOLUTION: Increase sectors or run again\n"
            f"  Already-cached sectors will load instantly on re-run\n"
            f"  Add more sectors by setting: SECTORS = [5, 6, 14, 18, 20, 1, 2, 15, 21]"
        )

    print(
        f"\n✓ Dataset size validation passed: {len(normalized_light_curves)} light curves"
    )

    if len(normalized_light_curves) < 10000:
        print(f"  Note: {len(normalized_light_curves)} is workable but <10k")
        print(f"  For best results, consider collecting more data later")

    VAL_CACHE_DIR = TECHNONET_DIR / "cached_lc_val"
    val_cache_files = [f for f in VAL_CACHE_DIR.iterdir() if f.suffix == ".npz"]

    if val_cache_files:
        print(f"\nLoading validation data from '{VAL_CACHE_DIR.relative_to(Path.cwd())}'\n")
        val_light_curves = []
        
        for cache_file in sorted(val_cache_files):
            data = np.load(cache_file)
            sector_lcs = [data[key] for key in data.files]
            val_light_curves.extend(sector_lcs)
            print(f"  ✓ Loaded {len(sector_lcs)} val LCs from {cache_file.name}")
        
        print(f"\nTotal validation: {len(val_light_curves)} light curves\n")
        
        # Create validation dataset and loader
        val_dataset = TAE_TESS_Dataset(val_light_curves)
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,  # Don't shuffle validation
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=collate_fn_pad
        )

    else:
        print("Warning: No validation data found. Training without validation.\n")
        val_loader = None


    # ================================
    # 4. Create DataLoader
    # ================================

    dataset = TAE_TESS_Dataset(normalized_light_curves)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True,  # Keep workers alive between epochs
        collate_fn=collate_fn_pad,
    )

    # ================================
    # 5. Initialize Model
    # ================================

    TAE = models.TemporalAutoEncoder(
        in_channels=IN_CHANNELS,
        channels=CHANNELS,
        kernel_size=KERNEL_SIZE,
        reduced_channels=REDUCED_CHANNELS,
        bottleneck_channels=BOTTLENECK_CHANNELS,
        latent_time=LATENT_TIME,
        dropout=DROPOUT,
    ).to(device)

    print(f"{'='*60}")
    print(f"MODEL ARCHITECTURE")
    print(f"{'='*60}")
    print(f"  Encoder channels: {CHANNELS}")
    print(f"  Latent time steps: {LATENT_TIME}")
    print(f"  Bottleneck channels: {BOTTLENECK_CHANNELS}")
    print(f"  Total parameters: {sum(p.numel() for p in TAE.parameters()):,}")
    print(f"{'='*60}\n")

    # ================================
    # 6. Loss and Optimizer
    # ================================

    # Loss function (L1/MAE loss for reconstruction)
    loss_fn = torch.nn.L1Loss()
    optimizer = Adam(TAE.parameters(), lr=LEARNING_RATE)

    # Use GradScaler for mixed precision training (PyTorch 2.0+ syntax)
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ================================
    # 7. Training Loop
    # ================================
    print(f"{'='*60}")
    print(f"TRAINING")
    print(f"{'='*60}")
    print(f"  Training size: {len(dataset)} light curves")
    print(f"  Validation size: {len(val_dataset) if val_loader else 0} light curves")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Device: {device}")
    print(f"  Mixed precision: {use_amp}")
    print(f"{'='*60}\n")

    best_val_loss = float("inf")  # Track best validation loss

    for epoch in range(1, EPOCHS + 1):
        # ==================
        # TRAINING PHASE
        # ==================
        TAE.train()
        train_loss = 0.0
        
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast("cuda"):
                    recon, z, _, _ = TAE(batch)
                    loss = loss_fn(recon, batch)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                recon, z, _, _ = TAE(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item() * batch.size(0)
        
        avg_train_loss = train_loss / len(dataset)
        
        # ==================
        # VALIDATION PHASE
        # ==================
        if val_loader is not None:
            TAE.eval()  # Set to evaluation mode
            val_loss = 0.0
            
            with torch.no_grad():  # No gradients needed
                for batch in val_loader:
                    batch = batch.to(device)
                    
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            recon, z, _, _ = TAE(batch)
                            loss = loss_fn(recon, batch)
                    else:
                        recon, z, _, _ = TAE(batch)
                        loss = loss_fn(recon, batch)
                    
                    val_loss += loss.item() * batch.size(0)
            
            avg_val_loss = val_loss / len(val_dataset)
            
            # Print with both train and val loss
            print(f"Epoch {epoch:2d}/{EPOCHS} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}", end="")
            
            # Save best model based on VALIDATION loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = SAVE_DIR / "best_TAE_model.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": TAE.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "model_config": {
                            "in_channels": IN_CHANNELS,
                            "channels": CHANNELS,
                            "kernel_size": KERNEL_SIZE,
                            "reduced_channels": REDUCED_CHANNELS,
                            "bottleneck_channels": BOTTLENECK_CHANNELS,
                            "latent_time": LATENT_TIME,
                            "dropout": DROPOUT,
                        },
                    },
                    checkpoint_path,
                )
                print(f"  ← NEW BEST VAL (saved)")
            else:
                print()
        
        else:
            # No validation - use training loss (original behavior)
            print(f"Epoch {epoch:2d}/{EPOCHS} | Train: {avg_train_loss:.6f}", end="")
            
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                checkpoint_path = SAVE_DIR / "best_TAE_model.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": TAE.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_val_loss,
                        "model_config": {
                            "in_channels": IN_CHANNELS,
                            "channels": CHANNELS,
                            "kernel_size": KERNEL_SIZE,
                            "reduced_channels": REDUCED_CHANNELS,
                            "bottleneck_channels": BOTTLENECK_CHANNELS,
                            "latent_time": LATENT_TIME,
                            "dropout": DROPOUT,
                        },
                    },
                    checkpoint_path,
                )
                print(f"  ← NEW BEST (saved)")
            else:
                print()
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = SAVE_DIR / f"TAE_epoch_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": TAE.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss if val_loader else None,
                },
                checkpoint_path,
            )
            print(f"  → Checkpoint saved: epoch_{epoch}.pth")

    print("\n" + "=" * 60)
    print("TAE TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Validation Loss: {best_val_loss:.6f}")
    print(f"  Models saved in: {SAVE_DIR.relative_to(Path.cwd())}/")
    print(f"  Best model: best_TAE_model.pth")
    print("=" * 60 + "\n")

# Replace your entire TCN_TRAIN section with this:

if TCN_TRAIN:
    print("\n" + "=" * 60)
    print("TRAINING TEMPORAL CONVOLUTIONAL NETWORK (TCN)")
    print("=" * 60 + "\n")

    DATA_DIR = TECHNONET_DIR / "final_training_data_TCN_train"
    VAL_DATA_DIR = TECHNONET_DIR / "final_training_data_TCN_val"

    class TCN_Dataset(Dataset):
        def __init__(self, light_curves, labels):
            self.light_curves = light_curves
            self.labels = labels

        def __len__(self):
            return len(self.light_curves)

        def __getitem__(self, idx):
            lc = torch.tensor(self.light_curves[idx], dtype=torch.float32)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return lc.unsqueeze(0), label

    def collate_fn_pad(batch):
        """Pad sequences to the same length in a batch."""
        data_list = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch])
        
        max_len = max(x.shape[-1] for x in data_list)
        
        padded_batch = []
        for x in data_list:
            if x.shape[-1] < max_len:
                padding = torch.zeros(x.shape[0], max_len - x.shape[-1])
                x = torch.cat([x, padding], dim=-1)
            padded_batch.append(x)
        
        return torch.stack(padded_batch, dim=0), labels

    def compute_metrics(model, loader, device, use_amp=False):
        """Compute accuracy, confusion matrix, and collect predictions for ROC."""
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(batch_data)
                else:
                    logits = model(batch_data)
                
                probs = torch.softmax(logits, dim=1)[:, 1]  # Prob of class 1
                preds = logits.max(1)[1]
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score
        
        accuracy = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
        
        # Only compute ROC if both classes present
        if len(np.unique(all_labels)) > 1:
            roc_auc = roc_auc_score(all_labels, all_probs)
        else:
            roc_auc = None
        
        cm = confusion_matrix(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'predictions': all_preds,
            'probabilities': all_probs,
            'labels': all_labels
        }

    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 30

    # Load training data
    print(f"Loading training data from {DATA_DIR}...")
    
    if not DATA_DIR.exists():
        raise RuntimeError(f"Training data directory not found: {DATA_DIR}")
    
    light_curves_train = []
    labels_train = []
    
    for file_path in sorted(DATA_DIR.glob("*.npz")):
        data = np.load(file_path, allow_pickle=True)
        
        if 'light_curves' in data and 'labels' in data:
            lcs = data['light_curves']
            light_curves_train.extend([lc for lc in lcs] if lcs.dtype == object else [lcs[i] for i in range(len(lcs))])
            labels_train.extend(data['labels'])

    labels_train = np.array(labels_train)
    print(f"✓ Loaded {len(light_curves_train)} training samples (Class 0: {sum(labels_train == 0)}, Class 1: {sum(labels_train == 1)})")
    
    # Load validation data
    print(f"\nLoading validation data from {VAL_DATA_DIR}...")
    
    if VAL_DATA_DIR.exists():
        light_curves_val = []
        labels_val = []
        
        for file_path in sorted(VAL_DATA_DIR.glob("*.npz")):
            data = np.load(file_path, allow_pickle=True)
            
            if 'light_curves' in data and 'labels' in data:
                lcs = data['light_curves']
                light_curves_val.extend([lc for lc in lcs] if lcs.dtype == object else [lcs[i] for i in range(len(lcs))])
                labels_val.extend(data['labels'])
        
        labels_val = np.array(labels_val)
        print(f"✓ Loaded {len(light_curves_val)} validation samples (Class 0: {sum(labels_val == 0)}, Class 1: {sum(labels_val == 1)})")
        
        val_dataset = TCN_Dataset(light_curves_val, labels_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=collate_fn_pad,
        )
    else:
        print("WARNING: No validation data found. Training without validation.")
        val_loader = None
        val_dataset = None
    
    # Create training dataset and loader
    dataset = TCN_Dataset(light_curves_train, labels_train)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True,
        collate_fn=collate_fn_pad,
    )

    # After loading data, before creating the model
    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels_train),
        y=labels_train
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print(f"\nClass weights: {class_weights}")
    print(f"  Class 0 (normal): {class_weights[0]:.4f}")
    print(f"  Class 1 (dyson): {class_weights[1]:.4f}\n")

    # ... model initialization ...


    # Model architecture
    NUM_INPUTS = 1
    NUM_CHANNELS = [64, 64, 64]
    KERNEL_SIZE = 3
    DROPOUT = 0.2
    CLS_INPUT_DIM = NUM_CHANNELS[-1]
    CLS_HIDDEN_DIM = 32
    CLS_OUTPUT_DIM = 2

    class TCNClassifier(nn.Module):
        def __init__(self, tcn, classifier):
            super().__init__()
            self.tcn = tcn
            self.classifier = classifier

        def forward(self, x):
            feats = self.tcn(x).mean(dim=-1)  # [B, C, T] -> [B, C]
            return self.classifier(feats)

    # Initialize model
    model = TCNClassifier(
        models.TemporalConvNet(NUM_INPUTS, NUM_CHANNELS, KERNEL_SIZE, DROPOUT),
        models.Classifier(CLS_INPUT_DIM, CLS_HIDDEN_DIM, CLS_OUTPUT_DIM, DROPOUT)
    ).to(device)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Update criterion to use class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    print(f"\nTraining: {len(dataset)} samples, {EPOCHS} epochs, batch size {BATCH_SIZE}\n")

    SAVE_DIR = TECHNONET_DIR / "checkpoints"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_metrics = None  # Initialize this!
    
    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for batch_data, batch_labels in loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(batch_data)
                    loss = criterion(logits, batch_labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(batch_data)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item() * batch_data.size(0)
            train_correct += logits.max(1)[1].eq(batch_labels).sum().item()

        avg_train_loss = train_loss / len(dataset)
        train_acc = 100.0 * train_correct / len(dataset)
        
        # Validation
        if val_loader is not None:
            # Compute detailed metrics
            val_metrics = compute_metrics(model, val_loader, device, use_amp)
            
            # Compute validation loss properly
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            logits = model(batch_data)
                            loss = criterion(logits, batch_labels)
                    else:
                        logits = model(batch_data)
                        loss = criterion(logits, batch_labels)
                    
                    val_loss += loss.item() * batch_data.size(0)
            
            avg_val_loss = val_loss / len(val_dataset)
            val_acc = val_metrics['accuracy']
            
            print(f"Epoch {epoch:2d}/{EPOCHS} | Train: {avg_train_loss:.4f} ({train_acc:.1f}%) | "
                  f"Val: {avg_val_loss:.4f} ({val_acc:.1f}%)", end="")
            
            if val_metrics['roc_auc'] is not None:
                print(f" | ROC-AUC: {val_metrics['roc_auc']:.4f}", end="")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_metrics = val_metrics  # Save best metrics
                
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_acc,
                    "roc_auc": val_metrics['roc_auc'],
                    "confusion_matrix": val_metrics['confusion_matrix'].tolist(),
                    "precision": val_metrics['precision'],
                    "recall": val_metrics['recall'],
                }, SAVE_DIR / "best_TCN_Classifier_model.pth")
                print(f"  ← BEST")
            else:
                print()
        else:
            print(f"Epoch {epoch:2d}/{EPOCHS} | Train: {avg_train_loss:.4f} ({train_acc:.1f}%)")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), SAVE_DIR / f"TCN_Classifier_epoch_{epoch}.pth")
    
    print(f"\n{'='*60}")
    print(f"TCN TRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved in: {SAVE_DIR}")
    print(f"{'='*60}\n")

    # After training loop completes
    if val_loader is not None and best_metrics is not None and best_metrics['roc_auc'] is not None:
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        
        fpr, tpr, thresholds = roc_curve(best_metrics['labels'], best_metrics['probabilities'])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('TCN Classifier ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(SAVE_DIR / 'tcn_roc_curve.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved ROC curve to {SAVE_DIR / 'tcn_roc_curve.png'}")
        
        # Print confusion matrix
        print("\nBest Model Performance:")
        print(f"  Accuracy: {best_metrics['accuracy']:.2f}%")
        print(f"  ROC-AUC: {best_metrics['roc_auc']:.4f}")
        print(f"  Precision: {best_metrics['precision']:.4f}")
        print(f"  Recall: {best_metrics['recall']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  {best_metrics['confusion_matrix']}")
        print(f"  TN={best_metrics['confusion_matrix'][0,0]}, FP={best_metrics['confusion_matrix'][0,1]}")
        print(f"  FN={best_metrics['confusion_matrix'][1,0]}, TP={best_metrics['confusion_matrix'][1,1]}")