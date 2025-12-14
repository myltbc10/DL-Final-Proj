"""
Plot Training Loss Over Epochs
Run on any models folder containing training_history.json
"""
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def plot_training_history(models_dir):
    """
    Plot training and validation loss from a training_history.json file.
    
    Args:
        models_dir: Path to folder containing training_history.json
    """
    models_dir = Path(models_dir)
    history_path = models_dir / "training_history.json"
    
    if not history_path.exists():
        print(f"Error: training_history.json not found in {models_dir}")
        return False
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    if not history:
        print("Error: training_history.json is empty")
        return False
    
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h['val_loss'] for h in history]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=3)
    ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=3)
    
    best_epoch = val_losses.index(min(val_losses)) + 1
    best_val = min(val_losses)
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Val (epoch {best_epoch})')
    ax.scatter([best_epoch], [best_val], color='g', s=100, zorder=5, marker='*')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training Progress - {models_dir.name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax.set_xlim(0, max(epochs) + 1)
    ax.set_ylim(0, max(max(train_losses), max(val_losses)) * 1.1)
    
    plot_path = models_dir / "training_loss_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    try:
        plt.show()
    except:
        pass
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Plot training loss from any models folder')
    parser.add_argument('--folder', '-f', type=str, default='models',
                        help='Path to models folder (default: models/)')
    args = parser.parse_args()
    
    folder_path = Path(args.folder)
    
    if not folder_path.is_absolute():
        base_path = Path(__file__).parent.parent
        folder_path = base_path / args.folder
    
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return
    
    plot_training_history(folder_path)

if __name__ == "__main__":
    main()
