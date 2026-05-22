#!/bin/bash
set -e

echo "========================================"
echo "=== Lambda GH200 Simulation Setup ==="
echo "========================================"

echo "=== Updating base packages ==="
sudo apt update -y
sudo apt install -y python3-full python3-venv python3-pip git

echo "=== Ensuring pip is available system-wide ==="
python3 -m ensurepip --upgrade

echo "=== Setting up virtual environment ==="
# Remove broken or old venvs
if [ -d "venv" ]; then
  echo "Existing venv found. Removing it for a clean setup..."
  rm -rf venv
fi

python3 -m venv venv --copies
source venv/bin/activate

echo "=== Upgrading pip, setuptools, wheel ==="
pip install --upgrade pip setuptools wheel --upgrade-strategy eager

echo "=== Installing core dependencies ==="
pip install numpy scipy matplotlib pandas tqdm rebound --upgrade-strategy eager

echo "=== Installing JAX (CUDA 12 build) ==="
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --upgrade-strategy eager

echo "=== Verifying GPU visibility via JAX ==="
python3 - <<'EOF'
import jax
print("\nDetected JAX devices:")
for d in jax.devices():
    print(" -", d)
EOF

echo "========================================"
echo "✅ Setup complete!"
echo "To activate your venv, run:"
echo "  source ~/code/venv/bin/activate"
echo "Then start your simulation with:"
echo "  python3 -m Simulation.sim"
echo "========================================"
