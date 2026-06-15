from argparse import ArgumentParser
from csv import DictWriter, QUOTE_MINIMAL
from pathlib import Path

from scipy.io import loadmat


def as_list(value):
    if isinstance(value, list):
        return value
    return [value]


def screen_texts(stimulus_file, line_separator):
    stimulus = loadmat(stimulus_file, simplify_cells=True)
    lines = as_list(stimulus["lines"])
    screens = as_list(stimulus["screens"])
    grouped_lines = {screen_id: [] for screen_id in range(1, len(screens) + 1)}

    for line in lines:
        grouped_lines[int(line["screen"])].append(str(line["text"]))

    return [
        {"index": screen_id - 1, "all": line_separator.join(grouped_lines[screen_id])}
        for screen_id in sorted(grouped_lines)
    ]


def export_stimulus(stimulus_file, output_dir, line_separator):
    output_file = output_dir / f"{stimulus_file.stem}.csv"
    rows = screen_texts(stimulus_file, line_separator)

    with output_file.open("w", encoding="utf-8", newline="") as file:
        writer = DictWriter(
            file,
            fieldnames=["index", "all"],
            quoting=QUOTE_MINIMAL,
            doublequote=True,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_file


def get_stimulus_files(stimuli_path, item):
    if item == "all":
        return sorted(stimuli_path.glob("*.mat"))

    stimulus_file = stimuli_path / f"{item}.mat"
    if not stimulus_file.exists():
        raise FileNotFoundError(f"Stimulus not found: {stimulus_file}")
    return [stimulus_file]


def main():
    parser = ArgumentParser(
        description="Export generated MATLAB stimuli to CSV files grouped by screen."
    )
    parser.add_argument(
        "--stimuli-path",
        type=Path,
        default=Path("stimuli"),
        help="Directory containing generated .mat stimuli.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("stimuli_csv"),
        help="Directory where CSV files will be written.",
    )
    parser.add_argument(
        "--item",
        default="all",
        help="Stimulus name without .mat, or 'all' to export every stimulus.",
    )
    parser.add_argument(
        "--line-separator",
        choices=["newline", "space"],
        default="space",
        help="How to join text lines that belong to the same screen.",
    )
    args = parser.parse_args()

    line_separator = "\n" if args.line_separator == "newline" else " "
    args.output_path.mkdir(parents=True, exist_ok=True)

    output_files = [
        export_stimulus(stimulus_file, args.output_path, line_separator)
        for stimulus_file in get_stimulus_files(args.stimuli_path, args.item)
    ]

    print(f"Exported {len(output_files)} CSV file(s) to {args.output_path}")


if __name__ == "__main__":
    main()
