"""Generate one subject-neutral canonical stimulus JSON file per story."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .pipeline import CorrelationConfig, load_canonical_stimulus


CORRELATION_DIR = Path(__file__).resolve().parent.parent


def story_filename(story: str) -> str:
    return re.sub(r"[^\w.-]+", "_", story, flags=re.UNICODE).strip("_") + ".json"


def validate_canonical_story(story: str, screens: list[dict]) -> dict:
    expected_story_index = 0
    word_count = 0
    for expected_screen_index, screen in enumerate(screens):
        if screen["screen_index"] != expected_screen_index:
            raise ValueError(f"{story}: non-consecutive screen index")
        boundaries = screen["line_boundaries"]
        if len(boundaries) < 2 or any(
            left >= right for left, right in zip(boundaries, boundaries[1:])
        ):
            raise ValueError(f"{story}, screen {expected_screen_index}: invalid line boundaries")
        for expected_screen_word_index, word in enumerate(screen["words"]):
            if word["story_word_index"] != expected_story_index:
                raise ValueError(f"{story}: non-consecutive story word index")
            if word["screen_word_index"] != expected_screen_word_index:
                raise ValueError(f"{story}: non-consecutive screen word index")
            if not word["x_min"] <= word["x_center"] < word["x_max"]:
                raise ValueError(f"{story}: word center outside horizontal box")
            if not word["y_min"] <= word["y_center"] < word["y_max"]:
                raise ValueError(f"{story}: word center outside vertical box")
            expected_story_index += 1
            word_count += 1
    return {"screens": len(screens), "words": word_count}


def build_all(output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stimuli_dir = CorrelationConfig(story="placeholder").stimuli_dir
    stories = sorted(path.stem for path in stimuli_dir.glob("*.mat") if path.stem != "Test")
    index = []
    for requested_story in stories:
        config = CorrelationConfig(story=requested_story)
        story, screens = load_canonical_stimulus(config, use_cache=False)
        counts = validate_canonical_story(story, screens)
        filename = story_filename(story)
        document = {
            "story": story,
            "source_stimulus": str(config.stimuli_dir / f"{story}.mat"),
            "source_config": str(config.stimuli_config),
            "coordinate_system": "pixels; origin at top-left; zero-based indexes",
            "screens": screens,
        }
        with (output_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
        index.append({"story": story, "file": filename, **counts})
    with (output_dir / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=CORRELATION_DIR / "canonical_stimuli",
        help="Directory for per-story JSON files",
    )
    args = parser.parse_args()
    index = build_all(args.output.resolve())
    print(f"Generated and validated {len(index)} stories in {args.output.resolve()}")


if __name__ == "__main__":
    main()
