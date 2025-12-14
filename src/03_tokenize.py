"""
Step 4: Tokenization
Convert MIDI files to token sequences for Transformer training.
Uses MidiTok with REMI representation.

Supports pitch augmentation (transposition) for better generalization.

Usage:
    python 03_tokenize.py                    # Normal tokenization
    python 03_tokenize.py --augment          # With pitch augmentation (12 keys)
    python 03_tokenize.py --verify           # Preview augmentation without tokenizing
    python 03_tokenize.py --augment --verify # Preview augmented data
"""
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from pathlib import Path
import json
import torch
from tqdm import tqdm
import pretty_midi
import argparse
import tempfile
import shutil
import random


def transpose_midi(input_path, semitones):
    """
    Transpose a MIDI file by a number of semitones.
    
    Args:
        input_path: Path to the MIDI file
        semitones: Number of semitones to transpose (+/- 12)
        
    Returns:
        Transposed PrettyMIDI object, or None if failed
    """
    try:
        midi = pretty_midi.PrettyMIDI(str(input_path))
        
        for instrument in midi.instruments:
            for note in instrument.notes:
                new_pitch = note.pitch + semitones
                if 0 <= new_pitch <= 127:
                    note.pitch = new_pitch
                else:
                    while new_pitch > 127:
                        new_pitch -= 12
                    while new_pitch < 0:
                        new_pitch += 12
                    note.pitch = new_pitch
        
        return midi
    except Exception as e:
        return None


def save_transposed_midi(midi_obj, output_path):
    """Save a PrettyMIDI object to file."""
    midi_obj.write(str(output_path))


def verify_augmentation(input_files, paired_dir, output_dir):
    """Preview mode: Show what augmentation would create and save samples."""
    transpositions = list(range(-5, 7))
    
    print(f"Original files: {len(input_files)}")
    print(f"After augmentation: {len(input_files) * len(transpositions)}")
    
    sample_dir = output_dir / "augmentation_samples"
    sample_dir.mkdir(exist_ok=True)
    
    if len(input_files) > 0:
        sample_input = input_files[0]
        
        for semitones in transpositions:
            transposed = transpose_midi(sample_input, semitones)
            if transposed:
                sign = "+" if semitones >= 0 else ""
                output_name = f"sample_{sign}{semitones}_semitones.mid"
                save_transposed_midi(transposed, sample_dir / output_name)
        
        print(f"Samples saved to: {sample_dir}/")


def create_tokenizer(midi_files, config_params=None):
    """
    Create and train a REMI tokenizer on sample MIDI files.
    
    Args:
        midi_files: List of MIDI file paths for vocabulary building
        config_params: Optional tokenizer configuration parameters
        
    Returns:
        REMI tokenizer object
    """
    if config_params is None:
        config_params = {
            'use_chords': False,
            'use_rests': False,
            'use_tempos': True,
            'use_time_signatures': True,
            'use_programs': False,
            'beat_res': {(0, 4): 4},
            'num_velocities': 4,
            'special_tokens': ['PAD', 'BOS', 'EOS'],
        }
    
    config = TokenizerConfig(**config_params)
    tokenizer = REMI(config)
    
    return tokenizer

def tokenize_file_pair(input_path, label_path, tokenizer, max_seq_len=512):
    """
    Tokenize a single (input, label) pair.
    
    Args:
        input_path: Path to corrupted input MIDI
        label_path: Path to clean label MIDI
        tokenizer: Trained REMI tokenizer
        max_seq_len: Maximum sequence length for chunking
        
    Returns:
        List of (input_tokens, label_tokens) tuples
    """
    try:
        input_tokens = tokenizer(str(input_path))
        label_tokens = tokenizer(str(label_path))
        
        if isinstance(input_tokens, list):
            input_ids = []
            for track in input_tokens:
                input_ids.extend(track.ids)
        else:
            input_ids = input_tokens.ids
            
        if isinstance(label_tokens, list):
            label_ids = []
            for track in label_tokens:
                label_ids.extend(track.ids)
        else:
            label_ids = label_tokens.ids
        
        chunks = []
        num_chunks = max(len(input_ids), len(label_ids)) // max_seq_len + 1
        
        for i in range(num_chunks):
            start_idx = i * max_seq_len
            end_idx = start_idx + max_seq_len
            
            input_chunk = input_ids[start_idx:end_idx]
            label_chunk = label_ids[start_idx:end_idx]
            
            # drop chunks with len < 50
            if len(input_chunk) < 50 or len(label_chunk) < 50:
                continue
            
            chunks.append({
                'input_ids': input_chunk,
                'label_ids': label_chunk,
                'source_file': input_path.stem
            })
        
        return chunks
        
    except Exception as e:
        return []

def main():
    parser = argparse.ArgumentParser(description='Tokenize MIDI files for training')
    parser.add_argument('--augment', action='store_true', 
                        help='Enable pitch augmentation (transpose to all 12 keys)')
    parser.add_argument('--verify', action='store_true',
                        help='Preview augmentation without full tokenization')
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent
    paired_dir = base_path / "dataset" / "paired_train"
    output_dir = base_path / "dataset"
    
    if not paired_dir.exists():
        print(f"ERROR: Directory not found: {paired_dir}")
        return
    
    all_input_files = sorted(paired_dir.glob("*_input.mid"))
    all_label_files = sorted(paired_dir.glob("*_label.mid"))
    
    input_files = [f for f in all_input_files if "versions" not in f.name]
    label_files = [f for f in all_label_files if "versions" not in f.name]
    
    if len(input_files) == 0:
        print(f"ERROR: No training pairs found")
        return
    
    print(f"Found {len(input_files)} training pairs")
    
    if args.verify:
        verify_augmentation(input_files, paired_dir, output_dir)
        return
    
    sample_files = input_files[:min(100, len(input_files))] + label_files[:min(100, len(label_files))]
    tokenizer = create_tokenizer(sample_files)
    print(f"Vocab size: {len(tokenizer.vocab)}")
    
    tokenizer_path = output_dir / "tokenizer"
    tokenizer.save(tokenizer_path)
    
    all_chunks = []
    max_seq_len = 512
    
    if args.augment:
        all_possible_transpositions = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
        num_random_keys = 2
        print(f"Augmentation: original + {num_random_keys} keys per file")
    else:
        all_possible_transpositions = []
        num_random_keys = 0
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        for input_file in tqdm(input_files, desc="Tokenizing"):
            label_file = paired_dir / f"{input_file.stem.replace('_input', '_label')}.mid"
            
            if not label_file.exists():
                continue
            
            if args.augment:
                random_keys = random.sample(all_possible_transpositions, num_random_keys)
                transpositions = [0] + random_keys
            else:
                transpositions = [0]
            
            for semitones in transpositions:
                if semitones == 0:
                    chunks = tokenize_file_pair(input_file, label_file, tokenizer, max_seq_len)
                else:
                    transposed_input = transpose_midi(input_file, semitones)
                    transposed_label = transpose_midi(label_file, semitones)
                    
                    if transposed_input is None or transposed_label is None:
                        continue
                    
                    sign = "p" if semitones >= 0 else "m"
                    temp_input = temp_dir / f"{input_file.stem}_{sign}{abs(semitones)}.mid"
                    temp_label = temp_dir / f"{label_file.stem}_{sign}{abs(semitones)}.mid"
                    
                    save_transposed_midi(transposed_input, temp_input)
                    save_transposed_midi(transposed_label, temp_label)
                    
                    chunks = tokenize_file_pair(temp_input, temp_label, tokenizer, max_seq_len)
                    
                    for chunk in chunks:
                        chunk['source_file'] = f"{input_file.stem}_transpose_{semitones}"
                
                all_chunks.extend(chunks)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # 80/10/10 training/val/testing split
    random.seed(42)
    random.shuffle(all_chunks)
    
    train_idx = int(len(all_chunks) * 0.8)
    val_idx = int(len(all_chunks) * 0.9)
    train_chunks = all_chunks[:train_idx]
    val_chunks = all_chunks[train_idx:val_idx]
    test_chunks = all_chunks[val_idx:]
    
    train_path = output_dir / "train_data.json"
    val_path = output_dir / "val_data.json"
    test_path = output_dir / "test_data.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_chunks, f, indent=2)
    
    with open(val_path, 'w') as f:
        json.dump(val_chunks, f, indent=2)
    
    with open(test_path, 'w') as f:
        json.dump(test_chunks, f, indent=2)
    
    torch_train_path = output_dir / "train_data.pt"
    torch_val_path = output_dir / "val_data.pt"
    torch_test_path = output_dir / "test_data.pt"
    
    torch.save(train_chunks, torch_train_path)
    torch.save(val_chunks, torch_val_path)
    torch.save(test_chunks, torch_test_path)
    
    vocab_info = {
        'vocab_size': len(tokenizer.vocab),
        'max_seq_len': max_seq_len,
        'pad_token_id': tokenizer['PAD_None'],
        'bos_token_id': tokenizer['BOS_None'],
        'eos_token_id': tokenizer['EOS_None'],
        'train_samples': len(train_chunks),
        'val_samples': len(val_chunks),
        'test_samples': len(test_chunks)
    }
    
    vocab_info_path = output_dir / "tokenizer_info.json"
    with open(vocab_info_path, 'w') as f:
        json.dump(vocab_info, f, indent=2)
    
    print(f"Train: {len(train_chunks)}, Val: {len(val_chunks)}, Test: {len(test_chunks)}")

if __name__ == "__main__":
    main()
