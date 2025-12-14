"""
Run inference on the held-out split (train/val/test) produced by 03_tokenize.py.

This uses dataset/{split}_data.pt (or .json) to pick which source MIDIs belong
to the split, then runs inference on the corresponding raw *_input.mid files
inside dataset/paired_train/.

Outputs cleaned MIDIs to --output_dir.
"""
import json
import torch
from transformers import T5Config, T5ForConditionalGeneration
from miditok import REMI
from pathlib import Path
import pretty_midi
import argparse
from datetime import datetime
import traceback


def limit_chord_span(midi_path, max_span=12):
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        notes_removed = 0

        for instrument in midi.instruments:
            time_tolerance = 0.05
            notes_by_time = {}

            for note in instrument.notes:
                time_bucket = round(note.start / time_tolerance) * time_tolerance
                notes_by_time.setdefault(time_bucket, []).append(note)

            notes_to_keep = []
            for _, chord_notes in notes_by_time.items():
                if len(chord_notes) <= 2:
                    notes_to_keep.extend(chord_notes)
                else:
                    pitches = [n.pitch for n in chord_notes]
                    min_pitch = min(pitches)
                    max_pitch = max(pitches)
                    span = max_pitch - min_pitch

                    if span <= max_span:
                        notes_to_keep.extend(chord_notes)
                    else:
                        sorted_notes = sorted(chord_notes, key=lambda n: n.pitch)
                        keep = [sorted_notes[0], sorted_notes[-1]]

                        for note in sorted_notes[1:-1]:
                            if note.pitch - min_pitch <= max_span:
                                keep.append(note)
                                if len(keep) >= 4:
                                    break

                        notes_to_keep.extend(keep)
                        notes_removed += len(chord_notes) - len(keep)

            instrument.notes = notes_to_keep

        midi.write(str(midi_path))
        return notes_removed

    except Exception:
        return 0


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = T5Config(**checkpoint["model_config"])
    model = T5ForConditionalGeneration(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_model_from_checkpoint(model_path, device, tokenizer_info):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    vocab_size = tokenizer_info["vocab_size"]
    pad_token_id = tokenizer_info.get("pad_token_id", 0)

    model_config = T5Config(
        vocab_size=vocab_size,
        d_model=512,
        d_ff=2048,
        num_layers=6,
        num_heads=8,
        dropout_rate=0.1,
        decoder_start_token_id=pad_token_id,
    )

    model = T5ForConditionalGeneration(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def inference_on_midi(model, tokenizer, input_midi_path, device, max_length=512):
    input_tokens = tokenizer(str(input_midi_path))

    if isinstance(input_tokens, list):
        input_ids = []
        for track in input_tokens:
            input_ids.extend(track.ids)
    else:
        input_ids = input_tokens.ids

    if len(input_ids) < 50:
        chunks = [input_ids] if len(input_ids) > 0 else []
    else:
        chunks = []
        for i in range(0, len(input_ids), max_length):
            chunk = input_ids[i : i + max_length]
            if len(chunk) >= 10:
                chunks.append(chunk)

    if not chunks:
        return []

    special_tokens = set()
    for name in ["PAD", "BOS", "EOS", "PAD_None", "BOS_None", "EOS_None"]:
        try:
            special_tokens.add(tokenizer[name])
        except Exception:
            pass

    all_output_ids = []
    with torch.no_grad():
        for chunk in chunks:
            input_tensor = torch.tensor([chunk], dtype=torch.long).to(device)
            output = model.generate(
                input_tensor,
                max_length=max_length,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                no_repeat_ngram_size=2,
            )
            output_ids = output[0].cpu().tolist()
            output_ids = [t for t in output_ids if t not in special_tokens]
            all_output_ids.extend(output_ids)

    return all_output_ids


def demo_single_file(input_path, output_path, model, tokenizer, device):
    output_ids = inference_on_midi(model, tokenizer, input_path, device)
    try:
        if not output_ids:
            return False

        output_midi = tokenizer.decode([output_ids])
        output_midi.dump_midi(str(output_path))
        limit_chord_span(output_path, max_span=12)
        return True

    except Exception:
        traceback.print_exc()
        return False


def load_split_chunks(split_path: Path):
    if split_path.suffix == ".pt":
        return torch.load(split_path)
    if split_path.suffix == ".json":
        with open(split_path, "r") as f:
            return json.load(f)
    raise ValueError(f"Unsupported split file type: {split_path}")


def run_on_split(
    split_path: Path,
    paired_dir: Path,
    output_dir: Path,
    model,
    tokenizer,
    device,
    limit: int,
    include_transposed: bool,
):
    chunks = load_split_chunks(split_path)

    # split files are chunk-level; we want unique source MIDIs
    sources = []
    seen = set()
    for c in chunks:
        src = c.get("source_file", "")
        if not src:
            continue
        if (not include_transposed) and ("_transpose_" in src):
            continue
        if src not in seen:
            seen.add(src)
            sources.append(src)

    sources = sorted(sources)[:limit]

    manifest = {
        "split_file": str(split_path),
        "paired_dir": str(paired_dir),
        "output_dir": str(output_dir),
        "num_unique_sources_selected": len(sources),
        "include_transposed": include_transposed,
        "runs": [],
    }

    successful = 0
    for src in sources:
        # src is typically like: "<something>_input"
        input_path = paired_dir / f"{src}.mid"
        if not input_path.exists():
            # fallback in case src omitted "_input" somehow
            alt = paired_dir / f"{src}_input.mid"
            if alt.exists():
                input_path = alt

        ok = False
        out_path = output_dir / f"cleaned_{src}.mid"
        if input_path.exists():
            ok = demo_single_file(input_path, out_path, model, tokenizer, device)

        manifest["runs"].append(
            {
                "source_file": src,
                "input_path": str(input_path),
                "output_path": str(out_path),
                "success": bool(ok),
            }
        )
        successful += int(ok)

    # save a manifest so you can prove what you tested on
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Processed {successful}/{len(sources)} unique test MIDIs")
    print(f"Manifest saved to: {output_dir / 'run_manifest.json'}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on a dataset split")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--split_file", type=str, default=None, help="Override: path to *_data.pt or *_data.json")
    parser.add_argument("--output_dir", type=str, default="test_outputs")
    parser.add_argument("--model", type=str, default="best_model.pt")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--include_transposed", action="store_true", help="Include *_transpose_* items from split")
    parser.add_argument("--input", type=str, default=None, help="Optional: run a single MIDI file instead")

    args = parser.parse_args()

    base_path = Path(__file__).parent.parent
    models_dir = base_path / "models_v2"
    dataset_dir = base_path / "dataset"
    tokenizer_dir = dataset_dir / "tokenizer"
    paired_dir = dataset_dir / "paired_train"

    model_path = models_dir / args.model
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return

    if not tokenizer_dir.exists():
        print(f"ERROR: Tokenizer not found: {tokenizer_dir}")
        return

    if not paired_dir.exists():
        print(f"ERROR: paired_train directory not found: {paired_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer_config = tokenizer_dir / "config.txt"
    if tokenizer_json.exists():
        tokenizer = REMI(params=str(tokenizer_json))
    elif tokenizer_config.exists():
        tokenizer = REMI(params=str(tokenizer_config))
    else:
        print("ERROR: No tokenizer config found")
        return

    # model
    if "checkpoint" in model_path.name:
        tokenizer_info_path = dataset_dir / "tokenizer_info.json"
        if tokenizer_info_path.exists():
            with open(tokenizer_info_path, "r") as f:
                tokenizer_info = json.load(f)
        else:
            tokenizer_info = {"vocab_size": len(tokenizer.vocab), "pad_token_id": 0}
        model = load_model_from_checkpoint(model_path, device, tokenizer_info)
    else:
        model = load_model(model_path, device)

    output_dir = base_path / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # single-file mode (optional)
    if args.input:
        inp = Path(args.input)
        if not inp.exists():
            print(f"ERROR: Input file not found: {inp}")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outp = output_dir / f"cleaned_{timestamp}.mid"
        ok = demo_single_file(inp, outp, model, tokenizer, device)
        print(f"Single-file success={ok}. Output: {outp}")
        return

    # split mode
    if args.split_file:
        split_path = Path(args.split_file)
        if not split_path.is_absolute():
            split_path = base_path / split_path
    else:
        split_path = dataset_dir / f"{args.split}_data.pt"

    if not split_path.exists():
        print(f"ERROR: Split file not found: {split_path}")
        return

    run_on_split(
        split_path=split_path,
        paired_dir=paired_dir,
        output_dir=output_dir,
        model=model,
        tokenizer=tokenizer,
        device=device,
        limit=args.limit,
        include_transposed=args.include_transposed,
    )

    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
