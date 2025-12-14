#!/bin/bash
#SBATCH --job-name=midi_test
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1

echo "============================================"
echo "MIDI Model Testing"
echo "============================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load python/3.10.8
module load cuda/12.2.0

# Activate virtual environment
cd /users/mlu85/DL-Practice
source venv/bin/activate

# Force unbuffered output
export PYTHONUNBUFFERED=1

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run inference on test set (using batch mode with higher limit)
echo ""
echo "Running inference on test set..."
python src/05_run_test.py --split test --limit 100 --model best_model.pt --output_dir best_model_results

# Run evaluation if demo_outputs exist
echo ""
echo "Running evaluation..."
python src/06_evaluate.py --output_dir epoch_50_results

echo ""
echo "============================================"
echo "Testing Complete!"
echo "============================================"

