"""
File Crawler for POP909 Dataset
This script verifies that the POP909 dataset is accessible and counts all .mid files.
"""
import os
from pathlib import Path
from collections import defaultdict

def crawl_pop909(base_path):
    """
    Crawl through POP909 directory and find all MIDI files.
    
    Args:
        base_path: Path to the pop909 directory
    
    Returns:
        Dictionary with file counts and list of all MIDI paths
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"ERROR: Directory not found: {base_path}")
        return None
    
    midi_files = list(base_path.rglob("*.mid"))
    
    stats = {
        'total_files': len(midi_files),
        'by_folder': defaultdict(int),
        'sample_paths': []
    }
    
    for midi_path in midi_files:
        rel_path = midi_path.relative_to(base_path)
        folder = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
        stats['by_folder'][folder] += 1
        
        if len(stats['sample_paths']) < 5:
            stats['sample_paths'].append(str(rel_path))
    
    return midi_files, stats

def main():
    pop909_path = Path(__file__).parent.parent / "raw_data" / "pop909"
    
    result = crawl_pop909(pop909_path)
    
    if result is None:
        return
    
    midi_files, stats = result
    print(f"Found {stats['total_files']} MIDI files")

if __name__ == "__main__":
    main()
