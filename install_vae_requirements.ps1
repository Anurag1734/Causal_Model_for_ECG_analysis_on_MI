# Installation script for VAE training requirements
# Run this in PowerShell

Write-Host "Installing required packages for VAE training..." -ForegroundColor Green

# Install PyTorch with CUDA 11.8 support (for RTX 4050)
Write-Host "`n[1/4] Installing PyTorch with CUDA support..." -ForegroundColor Cyan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install WFDB for ECG file reading
Write-Host "`n[2/4] Installing WFDB..." -ForegroundColor Cyan
pip install wfdb

# Install NeuroKit2 for ECG processing
Write-Host "`n[3/4] Installing NeuroKit2..." -ForegroundColor Cyan
pip install neurokit2

# Install PyArrow for Parquet file support
Write-Host "`n[4/4] Installing PyArrow..." -ForegroundColor Cyan
pip install pyarrow

Write-Host "`n✅ Installation complete!" -ForegroundColor Green
Write-Host "`nVerifying installation..." -ForegroundColor Cyan

# Verify installations
python -c "import torch; print(f'PyTorch {torch.__version__} installed. CUDA available: {torch.cuda.is_available()}')"
python -c "import wfdb; print(f'WFDB {wfdb.__version__} installed')"
python -c "import neurokit2; print(f'NeuroKit2 {neurokit2.__version__} installed')"
python -c "import pyarrow; print(f'PyArrow {pyarrow.__version__} installed')"

Write-Host "`nRun pre-flight check:" -ForegroundColor Yellow
Write-Host "python scripts/preflight_vae_training.py" -ForegroundColor White
