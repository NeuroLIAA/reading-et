from collections import defaultdict
from pathlib import Path
import argparse


SKIP_FILES = {"metadata.mat", "Test.mat"}


def collect_story_subjects(raw_path):
    story_subjects = defaultdict(list)

    for subject_dir in sorted(path for path in raw_path.iterdir() if path.is_dir()):
        for story_file in sorted(subject_dir.glob("*.mat")):
            if story_file.name in SKIP_FILES:
                continue
            story_subjects[story_file.stem].append(subject_dir.name)

    return dict(sorted(story_subjects.items()))


def print_story_subjects(story_subjects):
    for story, subjects in story_subjects.items():
        print(f"{story} ({len(subjects)} subjects)")
        print(", ".join(subjects))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List which subjects have each story in data/raw"
    )
    parser.add_argument(
        "--raw",
        type=str,
        default="data/raw",
        help="Path where subject raw data is stored",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data path not found: {raw_path}")

    print_story_subjects(collect_story_subjects(raw_path))
