from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten scanpath JSON files into CSV tables."
    )
    parser.add_argument(
        "--scanpaths-dir",
        default="data/processed/scanpaths",
        help="Directory containing per-item/per-subject scanpath JSON files.",
    )
    parser.add_argument(
        "--out-fixations",
        default="data/processed/scanpaths_fixations.csv",
        help="CSV output with one row per fixation in the scanpath.",
    )
    parser.add_argument(
        "--out-sequences",
        default="data/processed/scanpaths_sequences.csv",
        help="CSV output with one row per subject/item and JSON-encoded scanpath arrays.",
    )
    return parser.parse_args()


def iter_scanpath_files(scanpaths_dir: Path):
    for item_dir in sorted(path for path in scanpaths_dir.iterdir() if path.is_dir()):
        for json_file in sorted(item_dir.glob("*.json")):
            yield item_dir.name, json_file


def read_scanpath(json_file: Path) -> dict:
    with json_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_lengths(scanpath: dict, json_file: Path) -> int:
    lengths = {
        "word_ids": len(scanpath.get("word_ids", [])),
        "words": len(scanpath.get("words", [])),
        "trial_fix": len(scanpath.get("trial_fix", [])),
        "durations": len(scanpath.get("durations", [])),
        "screens": len(scanpath.get("screens", [])),
        "screen_fix": len(scanpath.get("screen_fix", [])),
    }
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"Inconsistent scanpath lengths in {json_file}: {lengths}")
    return unique_lengths.pop() if unique_lengths else 0


def write_tables(scanpaths_dir: Path, out_fixations: Path, out_sequences: Path) -> tuple[int, int]:
    out_fixations.parent.mkdir(parents=True, exist_ok=True)
    out_sequences.parent.mkdir(parents=True, exist_ok=True)

    n_sequences = 0
    n_fixations = 0

    with (
        out_fixations.open("w", encoding="utf-8-sig", newline="") as fix_handle,
        out_sequences.open("w", encoding="utf-8-sig", newline="") as seq_handle,
    ):
        fix_writer = csv.DictWriter(
            fix_handle,
            fieldnames=[
                "item",
                "subj",
                "n_fixations",
                "fixation_index",
                "word_id",
                "word",
                "trial_fix",
                "duration",
                "screen",
                "screen_fix",
            ],
        )
        seq_writer = csv.DictWriter(
            seq_handle,
            fieldnames=[
                "item",
                "subj",
                "n_fixations",
                "word_ids_json",
                "words_json",
                "trial_fix_json",
                "durations_json",
                "screens_json",
                "screen_fix_json",
            ],
        )
        fix_writer.writeheader()
        seq_writer.writeheader()

        for item_name, json_file in iter_scanpath_files(scanpaths_dir):
            scanpath = read_scanpath(json_file)
            seq_len = validate_lengths(scanpath, json_file)
            n_sequences += 1
            n_fixations += seq_len

            seq_writer.writerow(
                {
                    "item": scanpath.get("item", item_name),
                    "subj": scanpath.get("subj", json_file.stem),
                    "n_fixations": scanpath.get("n_fixations", seq_len),
                    "word_ids_json": json.dumps(scanpath.get("word_ids", []), ensure_ascii=False),
                    "words_json": json.dumps(scanpath.get("words", []), ensure_ascii=False),
                    "trial_fix_json": json.dumps(scanpath.get("trial_fix", []), ensure_ascii=False),
                    "durations_json": json.dumps(scanpath.get("durations", []), ensure_ascii=False),
                    "screens_json": json.dumps(scanpath.get("screens", []), ensure_ascii=False),
                    "screen_fix_json": json.dumps(scanpath.get("screen_fix", []), ensure_ascii=False),
                }
            )

            for fixation_index, (
                word_id,
                word,
                trial_fix,
                duration,
                screen,
                screen_fix,
            ) in enumerate(
                zip(
                    scanpath.get("word_ids", []),
                    scanpath.get("words", []),
                    scanpath.get("trial_fix", []),
                    scanpath.get("durations", []),
                    scanpath.get("screens", []),
                    scanpath.get("screen_fix", []),
                )
            ):
                fix_writer.writerow(
                    {
                        "item": scanpath.get("item", item_name),
                        "subj": scanpath.get("subj", json_file.stem),
                        "n_fixations": scanpath.get("n_fixations", seq_len),
                        "fixation_index": fixation_index,
                        "word_id": word_id,
                        "word": word,
                        "trial_fix": trial_fix,
                        "duration": duration,
                        "screen": screen,
                        "screen_fix": screen_fix,
                    }
                )

    return n_sequences, n_fixations


def main() -> None:
    args = parse_args()
    scanpaths_dir = Path(args.scanpaths_dir)
    out_fixations = Path(args.out_fixations)
    out_sequences = Path(args.out_sequences)

    if not scanpaths_dir.is_dir():
        raise SystemExit(f"Scanpaths directory not found: {scanpaths_dir}")

    n_sequences, n_fixations = write_tables(scanpaths_dir, out_fixations, out_sequences)
    print(f"Wrote {n_sequences} scanpath sequences to {out_sequences}")
    print(f"Wrote {n_fixations} scanpath fixations to {out_fixations}")


if __name__ == "__main__":
    main()
