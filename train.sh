#!/bin/bash

#SBATCH --job-name=midi-t5-train
#SBATCH --output=slurm_logs/train_%j.out
#SBATCH --error=slurm_logs/train_%j.err

# === GPU Resources ===
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# === CPU/Memory ===
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# === Time (HH:MM:SS) ===
#SBATCH --time=72:00:00

# === Email notifications (optional - uncomment and edit) ===
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your_email@brown.edu

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "========================================"

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Load required modules on Oscar
module load python/3.11.0
module load cuda/12.1.1
module load cudnn/8.9.6

# Activate virtual environment (create one if needed)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate

# Install requirements
echo "Installing/updating requirements..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Print environment info
echo ""
echo "========================================"
echo "Environment Info:"
echo "========================================"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo "========================================"
echo ""

# Only tokenize if data doesn't exist
if [ ! -f "dataset/train_data.json" ]; then
    echo "Running tokenization..."
    python src/03_tokenize.py
else
    echo "Tokenized data already exists, skipping..."
fi

# Run training (GPU)
echo "Starting training..."
python src/04_train.py

echo ""
echo "========================================"
echo "Job finished: $(date)"
echo "========================================"

