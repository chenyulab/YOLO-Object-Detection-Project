#!/bin/bash
set -e

VENV_DIR=".yolo"

echo "=========================="
echo " YOLO environment setup"
echo "=========================="

# -----------------------------
# Create virtual environment
# -----------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment: $VENV_DIR"

    if py -3.12 --version >/dev/null 2>&1; then
        py -3.12 -m venv "$VENV_DIR"
    elif py -3.11 --version >/dev/null 2>&1; then
        py -3.11 -m venv "$VENV_DIR"
    else
        echo "Python 3.12 or 3.11 was not found."
        echo "Install Python 3.12, then rerun setup.sh."
        exit 1
    fi
else
    echo "Virtual environment already exists: $VENV_DIR"
fi

# -----------------------------
# Activate virtual environment
# -----------------------------
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "Could not find activation script inside $VENV_DIR."
    exit 1
fi

echo "Activated environment:"
python -c "import sys; print(sys.executable); print(sys.version)"

# -----------------------------
# Upgrade pip tools
# -----------------------------
python -m pip install --upgrade pip setuptools wheel

# -----------------------------
# Check for NVIDIA GPU
# -----------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi was not found."
    echo "This script is intended for NVIDIA GPU computers."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)
echo "Detected GPU:"
echo "$GPU_NAME"

# Guard against Blackwell GPUs
if echo "$GPU_NAME" | grep -Eiq "5090|Blackwell|RTX PRO 6000"; then
    echo ""
    echo "This appears to be a Blackwell-class GPU:"
    echo "$GPU_NAME"
    echo ""
    echo "Do not use this normal-GPU setup on Miller/Blackwell."
    echo "Use the working .yolo environment or a CUDA 12.8+ wheelhouse setup instead."
    exit 1
fi

# -----------------------------
# Install PyTorch first
# -----------------------------
echo "Removing existing PyTorch packages if present..."
python -m pip uninstall -y torch torchvision torchaudio || true

echo "Installing CUDA PyTorch for normal NVIDIA GPUs..."
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# -----------------------------
# Verify PyTorch before requirements
# -----------------------------
echo "Verifying GPU before installing project requirements..."
python verify_gpu.py

# -----------------------------
# Install project requirements
# -----------------------------
if [ -f "requirements.txt" ]; then
    echo "Installing project requirements..."
    python -m pip install -r requirements.txt
else
    echo "No requirements.txt found."
    exit 1
fi

# -----------------------------
# Final verification
# -----------------------------
echo "Final verification..."
python - <<'PY'
import torch
import ultralytics
import yaml

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is no longer available after installing requirements.")

print(torch.randn(1, device="cuda"))
print("Ultralytics import OK")
print("PyYAML import OK")
PY

echo "=========================="
echo " Setup complete"
echo "=========================="
echo ""
echo "To activate later:"
echo "source .yolo/Scripts/activate"
echo ""
echo "To train:"
echo "python -m scripts.train_model"