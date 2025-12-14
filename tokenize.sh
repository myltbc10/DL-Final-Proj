#!/bin/bash

#SBATCH --job-name=midi-prep
#SBATCH --output=slurm_logs/prep_%j.out
#SBATCH --error=slurm_logs/prep_%j.err

# === CPU only (no GPU needed) ===
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# === Time (corruption + tokenization) ===
#SBATCH --time=03:00:00

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

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate

# Install requirements (if needed)
echo "Checking requirements..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Remove old data to ensure fresh generation
echo ""
echo "========================================"
echo "Removing old data..."
echo "========================================"
rm -f dataset/train_data.json
rm -f dataset/val_data.json
rm -f dataset/train_data.pt
rm -f dataset/val_data.pt
rm -f dataset/tokenizer_info.json
rm -rf dataset/tokenizer
rm -rf dataset/paired_train
echo "Old data removed."

# Step 1: Run corruption ("Too Many Notes" strategy)
echo ""
echo "========================================"
echo "Step 1: Running corruption..."
echo "========================================"
python src/02_corrupt_data.py

# Step 2: Run tokenization with augmentation
echo ""
echo "========================================"
echo "Step 2: Running tokenization with augmentation..."
echo "========================================"
python src/03_tokenize.py --augment

echo ""
echo "========================================"
echo "Job finished: $(date)"
echo "========================================"
echo ""
echo "Next: Run 'sbatch train.sh' to start training"

