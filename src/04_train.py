"""
Step 5: Model Training
Train a Transformer (T5) to translate corrupted MIDI to clean MIDI.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Config, T5ForConditionalGeneration
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

class MIDIDataset(Dataset):
    """PyTorch Dataset for tokenized MIDI pairs."""
    
    def __init__(self, data_path, max_length=512):
        """
        Args:
            data_path: Path to JSON file with tokenized data
            max_length: Maximum sequence length
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_ids = item['input_ids'][:self.max_length]
        label_ids = item['label_ids'][:self.max_length]
        
        # pad to max length
        input_ids = input_ids + [0] * (self.max_length - len(input_ids))
        # -100 = ignore index for cross entropy
        label_ids = label_ids + [-100] * (self.max_length - len(label_ids))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        # gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    """Validation loop."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
    
    return total_loss / len(dataloader)

def main():
    base_path = Path(__file__).parent.parent
    dataset_dir = base_path / "dataset"
    models_dir = base_path / "models_v2"
    models_dir.mkdir(exist_ok=True)
    
    tokenizer_info_path = dataset_dir / "tokenizer_info.json"
    
    if not tokenizer_info_path.exists():
        print(f"ERROR: Tokenizer info not found")
        return
    
    with open(tokenizer_info_path, 'r') as f:
        tokenizer_info = json.load(f)
    
    vocab_size = tokenizer_info['vocab_size']
    max_seq_len = tokenizer_info['max_seq_len']
    pad_token_id = tokenizer_info['pad_token_id']
    
    print(f"Train: {tokenizer_info['train_samples']}, Val: {tokenizer_info['val_samples']}")
    
    # hyperparameters
    config = {
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 5e-5,
        'warmup_steps': 500,
        'save_every': 5,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # t5 transformer model config
    model_config = T5Config(
        vocab_size=vocab_size,
        d_model=512,
        d_ff=2048,
        num_layers=6,
        num_heads=8,
        dropout_rate=0.1,
        decoder_start_token_id=pad_token_id
    )
    
    model = T5ForConditionalGeneration(model_config)
    model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Num of params: {num_params:,}")
    
    train_dataset = MIDIDataset(dataset_dir / "train_data.json", max_seq_len)
    val_dataset = MIDIDataset(dataset_dir / "val_data.json", max_seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # adamw with linear warmup then decay
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(1, config['num_epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss = validate(model, val_loader, device)
        
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
        
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        
        # save checkpoint
        if epoch % config['save_every'] == 0:
            checkpoint_path = models_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
        
        # save if new best val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = models_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'model_config': model_config.to_dict(),
            }, best_model_path)
    
    # save final model
    final_model_path = models_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config.to_dict(),
        'training_history': training_history,
    }, final_model_path)
    
    history_path = models_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"Done. Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
