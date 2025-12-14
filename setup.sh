#!/bin/bash
# Quick Setup Script for Piano Project

echo "=================================================="
echo "Piano MIDI Simplification - Environment Setup"
echo "=================================================="

# Check Python version
echo ""
echo "📋 Checking Python version..."
python3 --version

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install --break-system-packages -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Next steps:"
echo "   1. Place POP909 dataset in: raw_data/pop909/"
echo "   2. Run: python3 src/00_verify_dataset.py"
echo "   3. Run: python3 src/01_make_ground_truth.py"
