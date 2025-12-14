"""
Step 3: Corruption Pipeline
Create "messy" input files from clean ground truth for training.

NEW STRATEGY: "Too Many Notes" corruption
- Instead of adding random noise, we ADD EXTRA NOTES to make the music unplayable
- The model must learn to REDUCE/SIMPLIFY (pick which notes matter)
- This teaches the model to be a "summarizer" not a "spell checker"
"""
import numpy as np
import pretty_midi
from pathlib import Path
from tqdm import tqdm
import random
import shutil


def add_extra_notes(notes, config):
    """
    Add extra notes around existing notes to create an "unplayable" mess.
    The model must learn which notes to keep (reduction task).
    
    Strategies:
    1. Duplicate notes with pitch shifts (±1-2 semitones) - creates dissonance
    2. Add octave duplicates - creates thickness
    3. Add chord cluster notes - fills in gaps
    4. Add passing tones - notes between existing notes
    """
    extra_notes = []
    
    duplicate_prob = config.get('duplicate_prob', 0.4)
    octave_prob = config.get('octave_prob', 0.3)
    cluster_prob = config.get('cluster_prob', 0.25)
    passing_prob = config.get('passing_prob', 0.2)
    
    for note in notes:
        # pitch-shifted duplicates
        if random.random() < duplicate_prob:
            shift = random.choice([-2, -1, 1, 2])
            new_pitch = note.pitch + shift
            if 21 <= new_pitch <= 108:
                extra_notes.append(pretty_midi.Note(
                    velocity=max(30, note.velocity - random.randint(10, 30)),
                    pitch=new_pitch,
                    start=note.start + random.uniform(-0.02, 0.02),
                    end=note.end
                ))
        
        # octave duplicates
        if random.random() < octave_prob:
            octave_shift = random.choice([-12, 12])
            new_pitch = note.pitch + octave_shift
            if 21 <= new_pitch <= 108:
                extra_notes.append(pretty_midi.Note(
                    velocity=max(30, note.velocity - random.randint(5, 20)),
                    pitch=new_pitch,
                    start=note.start,
                    end=note.end
                ))
        
        # cluster notes
        if random.random() < cluster_prob:
            cluster_shift = random.choice([-3, -4, 3, 4])
            new_pitch = note.pitch + cluster_shift
            if 21 <= new_pitch <= 108:
                extra_notes.append(pretty_midi.Note(
                    velocity=max(30, note.velocity - random.randint(15, 35)),
                    pitch=new_pitch,
                    start=note.start + random.uniform(-0.03, 0.03),
                    end=note.end
                ))
    
    # passing tones between consecutive notes if close enough
    sorted_notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    for i in range(len(sorted_notes) - 1):
        if random.random() < passing_prob:
            note1 = sorted_notes[i]
            note2 = sorted_notes[i + 1]
            
            time_gap = note2.start - note1.end
            pitch_gap = abs(note2.pitch - note1.pitch)
            
            if 0 < time_gap < 0.5 and 2 < pitch_gap < 7:
                mid_pitch = (note1.pitch + note2.pitch) // 2
                mid_time = (note1.end + note2.start) / 2
                extra_notes.append(pretty_midi.Note(
                    velocity=max(30, min(note1.velocity, note2.velocity) - 20),
                    pitch=mid_pitch,
                    start=mid_time,
                    end=mid_time + 0.15
                ))
    
    return extra_notes


def corrupt_midi(midi_obj, config):
    """
    Apply "Too Many Notes" corruption to create training input.
    
    Strategy: Add extra notes to make the music unplayable.
    The model learns to REDUCE this mess back to the clean original.
    
    Args:
        midi_obj: PrettyMIDI object with 2 tracks (LH, RH)
        config: Dictionary with corruption parameters
        
    Returns:
        PrettyMIDI object with too many notes
    """
    all_notes = []
    for inst in midi_obj.instruments:
        for note in inst.notes:
            all_notes.append(pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start,
                end=note.end
            ))
    
    if not all_notes:
        return None
    
    original_count = len(all_notes)
    
    # add timing noise
    timing_noise_range = config.get('timing_noise', 0.03)
    for note in all_notes:
        shift = random.uniform(-timing_noise_range, timing_noise_range)
        note.start = max(0, note.start + shift)
        note.end = max(note.start + 0.05, note.end + shift)
    
    extra_notes = add_extra_notes(all_notes, config)
    all_notes.extend(extra_notes)
    
    # vary velocities
    for note in all_notes:
        velocity_shift = random.randint(-15, 15)
        note.velocity = max(20, min(127, note.velocity + velocity_shift))
    
    corrupted_midi = pretty_midi.PrettyMIDI()
    merged_inst = pretty_midi.Instrument(program=0, name="Corrupted Piano")
    merged_inst.notes = sorted(all_notes, key=lambda n: n.start)
    corrupted_midi.instruments.append(merged_inst)
    
    new_count = len(all_notes)
    
    return corrupted_midi, original_count, new_count

def create_training_pair(input_path, output_dir, config):
    """
    Process a single clean file and create (input, label) pair.
    
    Args:
        input_path: Path to clean MIDI file
        output_dir: Directory to save pairs
        config: Corruption configuration
        
    Returns:
        tuple: (success, original_notes, corrupted_notes) or (False, 0, 0)
    """
    try:
        clean_midi = pretty_midi.PrettyMIDI(str(input_path))
        
        if len(clean_midi.instruments) != 2:
            return False, 0, 0
        
        result = corrupt_midi(clean_midi, config)
        
        if result is None:
            return False, 0, 0
        
        corrupted_midi, orig_count, new_count = result
        
        base_name = input_path.stem
        input_file = output_dir / f"{base_name}_input.mid"
        label_file = output_dir / f"{base_name}_label.mid"
        
        corrupted_midi.write(str(input_file))
        shutil.copy(input_path, label_file)
        
        return True, orig_count, new_count
        
    except Exception as e:
        return False, 0, 0

def main():
    base_path = Path(__file__).parent.parent
    clean_dir = base_path / "dataset" / "clean_split"
    output_dir = base_path / "dataset" / "paired_train"
    
    if not clean_dir.exists() or not list(clean_dir.glob("*.mid")):
        print(f"ERROR: No clean files found in {clean_dir}")
        return
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'timing_noise': 0.03,
        'duplicate_prob': 0.4,
        'octave_prob': 0.3,
        'cluster_prob': 0.25,
        'passing_prob': 0.2,
    }
    
    all_clean_files = list(clean_dir.glob("*.mid"))
    clean_files = [f for f in all_clean_files if "versions" not in f.name]
    
    print(f"Found {len(clean_files)} clean files")
    
    successful = 0
    total_orig = 0
    total_new = 0
    
    for clean_file in tqdm(clean_files, desc="Creating pairs"):
        success, orig, new = create_training_pair(clean_file, output_dir, config)
        if success:
            successful += 1
            total_orig += orig
            total_new += new
    
    multiplier = total_new / total_orig if total_orig > 0 else 0
    print(f"Created {successful} pairs, avg {multiplier:.1f}x note increase")

if __name__ == "__main__":
    main()
