#!/bin/bash
# ================================
# Train TechnoNet on TESS light curves
# ================================

# Fail fast on any error
set -e

echo "🔧 Setting up environment..."

# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip, setuptools, and wheel (critical for package installation & wheels)
pip install --upgrade pip setuptools wheel
echo "✅ pip, setuptools, and wheel upgraded."

# 3. Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install scientific Python dependencies
pip install lightkurve astroquery numpy tqdm matplotlib scikit-learn

# 5. Optional: for reproducibility
export PYTHONHASHSEED=0

# 6. Configure PyTorch for best performance on L4 (medium precision for tensor cores)
python3 - <<'PY'
import torch
torch.set_float32_matmul_precision('medium')
print("✅ Torch available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("💻 CUDA device:", torch.cuda.get_device_name(0))
    print("⚡ Matmul precision set to 'medium' (Tensor Cores enabled)")
PY

echo "✅ Environment setup complete."