"""
Step 2: Generate Ground Truth using K-Means Clustering
This script converts POP909 MIDI files into clean "Left Hand / Right Hand" format.
"""
import numpy as np
from sklearn.cluster import KMeans
import pretty_midi
from pathlib import Path
from tqdm import tqdm
import os

def separate_hands_kmeans(midi_obj):
    """
    Takes a PrettyMIDI object (with merged tracks) and returns 
    two new Instrument objects: Right Hand and Left Hand.
    
    Args:
        midi_obj: PrettyMIDI object with potentially multiple tracks
        
    Returns:
        tuple: (right_hand_instrument, left_hand_instrument)
    """
    all_notes = []
    for inst in midi_obj.instruments:
        if not inst.is_drum:
            all_notes.extend(inst.notes)
        
    if not all_notes:
        return None, None, None

    # use only pitch - time causes chronological clustering instead of register separation
    data = np.array([[note.pitch] for note in all_notes])

    # k-means with 2 clusters
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = kmeans.fit_predict(data)

    # figure out which cluster is high (right hand)
    avg_pitch_0 = data[labels == 0, 0].mean()
    avg_pitch_1 = data[labels == 1, 0].mean()
    
    rh_label = 0 if avg_pitch_0 > avg_pitch_1 else 1
    lh_label = 1 - rh_label

    rh_inst = pretty_midi.Instrument(program=0, name="Right Hand")
    lh_inst = pretty_midi.Instrument(program=0, name="Left Hand")

    for i, note in enumerate(all_notes):
        if labels[i] == rh_label:
            rh_inst.notes.append(note)
        else:
            lh_inst.notes.append(note)
    
    rh_pitches = [note.pitch for note in rh_inst.notes]
    lh_pitches = [note.pitch for note in lh_inst.notes]
    
    # rh_avg should be higher than lh_avg
    stats = {
        'rh_avg': np.mean(rh_pitches) if rh_pitches else 0,
        'lh_avg': np.mean(lh_pitches) if lh_pitches else 0,
    }
            
    return rh_inst, lh_inst, stats

def process_midi_file(input_path, output_path):
    """
    Process a single MIDI file: load, separate hands, save.
    
    Args:
        input_path: Path to input MIDI file
        output_path: Path to save the cleaned output
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(str(input_path))
        rh_inst, lh_inst, stats = separate_hands_kmeans(midi_data)
        
        if rh_inst is None or lh_inst is None:
            return False
        
        if len(rh_inst.notes) == 0 or len(lh_inst.notes) == 0:
            return False
        
        output_midi = pretty_midi.PrettyMIDI()
        output_midi.instruments.append(lh_inst)
        output_midi.instruments.append(rh_inst)
        
        output_midi.write(str(output_path))
        return True, stats
        
    except Exception as e:
        return False

def main():
    base_path = Path(__file__).parent.parent
    raw_data_path = base_path / "raw_data" / "pop909"
    output_path = base_path / "dataset" / "clean_split"
    
    if not raw_data_path.exists():
        print(f"ERROR: Input directory not found: {raw_data_path}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    midi_files = list(raw_data_path.rglob("*.mid"))
    print(f"Found {len(midi_files)} MIDI files")
    
    if len(midi_files) == 0:
        return
    
    successful = 0
    
    for midi_file in tqdm(midi_files, desc="Processing"):
        rel_path = midi_file.relative_to(raw_data_path)
        output_name = str(rel_path).replace(os.sep, "_")
        output_file = output_path / output_name
        
        result = process_midi_file(midi_file, output_file)
        if result and result[0]:
            successful += 1
    
    print(f"Processed {successful}/{len(midi_files)} files")

if __name__ == "__main__":
    main()
