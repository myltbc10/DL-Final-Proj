"""
Step 6: Evaluation Script
Compare model outputs against ground truth labels.
"""
import pretty_midi
from pathlib import Path
import numpy as np
from collections import defaultdict
import argparse

def extract_notes(midi_path):
    """Extract (pitch, start_time, duration) tuples from MIDI."""
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        notes = []
        for inst in midi.instruments:
            for note in inst.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': round(note.start, 2),
                    'end': round(note.end, 2),
                    'duration': round(note.end - note.start, 2)
                })
        return notes
    except Exception as e:
        return []

def compute_note_f1(pred_notes, gt_notes, time_tolerance=0.5, pitch_tolerance=2, octave_invariant=True):
    """
    Compute precision, recall, F1 for note matching.
    A note matches if pitch is within tolerance and onset is within time tolerance.
    
    Args:
        time_tolerance: Max time difference in seconds (default 0.5s - relaxed)
        pitch_tolerance: Max pitch difference in semitones (default ±2)
        octave_invariant: If True, ignore octave differences (default True)
    """
    if not pred_notes or not gt_notes:
        return {'precision': 0, 'recall': 0, 'f1': 0}
    
    gt_matched = [False] * len(gt_notes)
    pred_matched = [False] * len(pred_notes)
    
    for i, pred in enumerate(pred_notes):
        for j, gt in enumerate(gt_notes):
            if gt_matched[j]:
                continue
            
            pred_pitch = pred['pitch']
            gt_pitch = gt['pitch']
            
            if octave_invariant:
                pred_pitch = pred_pitch % 12
                gt_pitch = gt_pitch % 12
            
            pitch_diff = abs(pred_pitch - gt_pitch)
            time_diff = abs(pred['start'] - gt['start'])
            
            if pitch_diff <= pitch_tolerance and time_diff <= time_tolerance:
                pred_matched[i] = True
                gt_matched[j] = True
                break
    
    true_positives = sum(pred_matched)
    precision = true_positives / len(pred_notes) if pred_notes else 0
    recall = true_positives / len(gt_notes) if gt_notes else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def compute_note_density_ratio(pred_notes, gt_notes):
    """Ratio of predicted notes to ground truth notes."""
    if not gt_notes:
        return 0
    return len(pred_notes) / len(gt_notes)

def compute_pitch_histogram_similarity(pred_notes, gt_notes):
    """Cosine similarity between pitch histograms."""
    if not pred_notes or not gt_notes:
        return 0
    
    pred_hist = np.zeros(128)
    gt_hist = np.zeros(128)
    
    for n in pred_notes:
        pred_hist[n['pitch']] += 1
    for n in gt_notes:
        gt_hist[n['pitch']] += 1
    
    pred_hist = pred_hist / (np.linalg.norm(pred_hist) + 1e-8)
    gt_hist = gt_hist / (np.linalg.norm(gt_hist) + 1e-8)
    
    return np.dot(pred_hist, gt_hist)

def compute_playability(midi_path):
    """Check playability per hand/track: max simultaneous notes and max chord span."""
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        
        if not midi.instruments:
            return {'max_simultaneous': 0, 'max_span': 0, 'unplayable_chords': 0, 'tracks': 0}
        
        results = {
            'max_simultaneous': 0,
            'max_span': 0,
            'unplayable_chords': 0,
            'tracks': len(midi.instruments)
        }
        
        is_single_track = len(midi.instruments) == 1
        
        for inst in midi.instruments:
            track_notes = []
            for note in inst.notes:
                track_notes.append({
                    'pitch': note.pitch,
                    'start': round(note.start, 2)
                })
            
            if not track_notes:
                continue
            
            time_groups = defaultdict(list)
            for n in track_notes:
                t = round(n['start'] * 20) / 20
                time_groups[t].append(n['pitch'])
            
            for t, pitches in time_groups.items():
                results['max_simultaneous'] = max(results['max_simultaneous'], len(pitches))
                if len(pitches) > 1:
                    span = max(pitches) - min(pitches)
                    results['max_span'] = max(results['max_span'], span)
                    
                    if is_single_track:
                        if len(pitches) > 10:
                            results['unplayable_chords'] += 1
                    else:
                        if span > 12 or len(pitches) > 5:
                            results['unplayable_chords'] += 1
        
        return results
        
    except Exception as e:
        return {'max_simultaneous': 0, 'max_span': 0, 'unplayable_chords': 0, 'tracks': 0}

def evaluate_pair(output_path, label_path):
    """Evaluate a single output/label pair."""
    pred_notes = extract_notes(output_path)
    gt_notes = extract_notes(label_path)
    
    metrics = {}
    
    note_metrics = compute_note_f1(pred_notes, gt_notes)
    metrics.update(note_metrics)
    
    metrics['density_ratio'] = compute_note_density_ratio(pred_notes, gt_notes)
    metrics['pitch_similarity'] = compute_pitch_histogram_similarity(pred_notes, gt_notes)
    
    play_metrics = compute_playability(output_path)
    metrics.update(play_metrics)
    
    metrics['pred_notes'] = len(pred_notes)
    metrics['gt_notes'] = len(gt_notes)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate model outputs against ground truth')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory with model outputs')
    parser.add_argument('--label_dir', type=str, default=None, help='Directory with ground truth labels')
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent
    
    # handle both relative and absolute paths
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = base_path / output_dir
    else:
        output_dir = base_path / "demo_outputs"
    
    if args.label_dir:
        label_dir = Path(args.label_dir)
        if not label_dir.is_absolute():
            label_dir = base_path / label_dir
    else:
        label_dir = base_path / "dataset" / "paired_train"
    
    output_files = list(output_dir.glob("cleaned_*.mid"))
    print(f"Found {len(output_files)} output files")
    
    all_metrics = []
    
    for output_path in output_files:
        name = output_path.name.replace("cleaned_", "").replace("_input.mid", "_label.mid")
        label_path = label_dir / name
        
        if not label_path.exists():
            continue
        
        metrics = evaluate_pair(output_path, label_path)
        metrics['file'] = output_path.name
        all_metrics.append(metrics)
    
    if all_metrics:
        avg_f1 = np.mean([m['f1'] for m in all_metrics])
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])
        avg_density = np.mean([m['density_ratio'] for m in all_metrics])
        avg_pitch_sim = np.mean([m['pitch_similarity'] for m in all_metrics])
        total_unplayable = sum(m['unplayable_chords'] for m in all_metrics)
        
        print(f"F1: {avg_f1:.3f}, P: {avg_precision:.3f}, R: {avg_recall:.3f}")
        print(f"Pitch similarity: {avg_pitch_sim:.3f}, Density ratio: {avg_density:.2f}")
        print(f"Unplayable chords: {total_unplayable}")

if __name__ == "__main__":
    main()
