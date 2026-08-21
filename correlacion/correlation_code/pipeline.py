"""Prepare neutral simulated trials and run the original measure pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import unicodedata

import pandas as pd
from scipy.io import loadmat


CORRELATION_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CORRELATION_DIR.parent
REQUIRED_SIMULATION_COLUMNS = {
    "text_id", "fixation_counter", "fixation_duration", "foveal_word_index"
}


def _default_excluded_humans() -> list[str]:
    # Preserved from correlation_original.ipynb. Set excluded_humans=[] to keep all.
    return [
        "sub-020", "sub-113", "sub-029", "sub-069", "sub-031", "sub-087",
        "sub-054", "sub-085", "sub-077", "sub-060", "sub-097", "sub-070",
        "sub-053", "sub-036", "sub-065", "sub-039", "sub-111", "sub-003",
        "sub-013", "sub-066", "sub-091", "sub-018", "sub-052",
    ]


@dataclass
class CorrelationConfig:
    """All user-editable parameters for one story-level experiment."""

    story: str
    simulation_subjects: list[str] | None = None
    measures: list[str] = field(
        default_factory=lambda: ["FFD", "FPRT", "TFD", "SFD", "SPRT", "RRT"]
    )
    n_bins: int = 10
    excluded_humans: list[str] = field(default_factory=_default_excluded_humans)
    simulations_root: Path = REPO_ROOT / "model_output"
    human_measures_dir: Path | None = None
    experiments_root: Path = CORRELATION_DIR / "outputs" / "experiments"
    canonical_stimuli_dir: Path = CORRELATION_DIR / "canonical_stimuli"
    human_results_root: Path = CORRELATION_DIR / "human_results"
    stimuli_dir: Path = REPO_ROOT / "stimuli"
    stimuli_config: Path = REPO_ROOT / "metadata" / "stimuli_config.mat"
    all_stimuli_json: Path = CORRELATION_DIR / "all_stimuli.json"
    processed_examples: Path = REPO_ROOT / "data" / "processed_examples"
    original_human_trials: Path = REPO_ROOT / "data" / "processed_og_all" / "trials"
    geometry_mode: str = "canonical"
    python_executable: Path = REPO_ROOT / "tesis" / "bin" / "python"
    plot_min: float = 0.0
    plot_max: float = 0.4

    def __post_init__(self) -> None:
        for name in (
            "simulations_root", "experiments_root", "canonical_stimuli_dir",
            "human_results_root", "stimuli_dir", "stimuli_config", "all_stimuli_json",
            "processed_examples", "original_human_trials",
        ):
            setattr(self, name, Path(getattr(self, name)).expanduser().resolve())
        python = Path(self.python_executable).expanduser()
        self.python_executable = python if python.is_absolute() else (Path.cwd() / python).absolute()
        if self.human_measures_dir is not None:
            self.human_measures_dir = Path(self.human_measures_dir).expanduser().resolve()
        if not self.story.strip():
            raise ValueError("story cannot be empty")
        if self.n_bins < 1:
            raise ValueError("n_bins must be at least 1")
        if not self.measures:
            raise ValueError("measures cannot be empty")
        if self.geometry_mode not in {
            "canonical", "json_x_canonical_y", "original_compatibility"
        }:
            raise ValueError(
                "geometry_mode must be 'canonical', 'json_x_canonical_y', "
                "or 'original_compatibility'"
            )


@dataclass
class ExperimentRun:
    config: CorrelationConfig
    experiment_dir: Path
    story_key: str
    simulation_paths: list[Path]
    complete_simulations: list[Path]
    skipped_simulations: list[dict]
    simulated_measures_dir: Path
    human_measures_dir: Path

    @property
    def correlation_dir(self) -> Path:
        return self.experiment_dir / "measures_correlation"

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "story": self.story_key,
            "selected_simulations": len(self.simulation_paths),
            "processed_simulations": len(self.complete_simulations),
            "skipped_simulations": len(self.skipped_simulations),
            "experiment_dir": str(self.experiment_dir),
            "human_measures_dir": str(self.human_measures_dir),
        }])


def normalize_name(value: str) -> str:
    value = value.replace("_", " ")
    return "".join(
        char for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    ).casefold().strip()


def _resolve_unique_name(requested: str, available: list[str], kind: str) -> str:
    if requested in available:
        return requested
    matches = [name for name in available if normalize_name(name) == normalize_name(requested)]
    if len(matches) == 1:
        return matches[0]
    choices = ", ".join(sorted(available))
    raise KeyError(f"{kind} '{requested}' did not resolve uniquely. Available: {choices}")


def available_stories(simulations_root: Path = REPO_ROOT / "model_output") -> list[str]:
    root = Path(simulations_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Simulation root does not exist: {root}")
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def discover_simulations(
    story: str,
    subjects: list[str] | None = None,
    simulations_root: Path = REPO_ROOT / "model_output",
) -> tuple[str, list[Path]]:
    root = Path(simulations_root)
    story_dir_name = _resolve_unique_name(story, available_stories(root), "Simulation story")
    paths = sorted((root / story_dir_name).rglob("*.tsv"))
    by_subject: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in paths:
        subject = path.stem
        if subject in by_subject:
            duplicates.add(subject)
        by_subject[subject] = path.resolve()
    if duplicates:
        raise ValueError(f"Duplicate TSV subjects in {story_dir_name}: {sorted(duplicates)}")
    if subjects is not None:
        missing = [subject for subject in subjects if subject not in by_subject]
        if missing:
            raise FileNotFoundError(f"Simulations not found for subjects: {missing}")
        paths = [by_subject[subject] for subject in subjects]
    else:
        paths = list(by_subject.values())
    if not paths:
        raise FileNotFoundError(f"No simulation TSVs found for {story_dir_name}")
    return story_dir_name, paths


def load_canonical_stimulus(
    config: CorrelationConfig, *, use_cache: bool = True
) -> tuple[str, list[dict]]:
    """Load subject-independent word geometry from the original stimulus files."""

    mat_names = [path.stem for path in config.stimuli_dir.glob("*.mat")]
    story_key = _resolve_unique_name(config.story, mat_names, "Stimulus story")
    index_path = config.canonical_stimuli_dir / "index.json"
    if use_cache and index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            entries = json.load(handle)
        matches = [
            entry for entry in entries
            if normalize_name(entry["story"]) == normalize_name(story_key)
        ]
        if len(matches) == 1:
            cache_path = config.canonical_stimuli_dir / matches[0]["file"]
            with cache_path.open(encoding="utf-8") as handle:
                document = json.load(handle)
            return document["story"], document["screens"]

    stimulus = loadmat(config.stimuli_dir / f"{story_key}.mat", simplify_cells=True)
    settings = loadmat(config.stimuli_config, simplify_cells=True)
    short_names = [str(name).strip() for name in settings["short_stimuli"]]
    long_names = [str(name).strip() for name in settings["long_stimuli"]]
    if normalize_name(story_key) in {normalize_name(name) for name in short_names}:
        line_spacing = float(settings["short_config"]["linespacing"])
    elif normalize_name(story_key) in {normalize_name(name) for name in long_names}:
        line_spacing = float(settings["long_config"]["linespacing"])
    else:
        raise KeyError(f"'{story_key}' is absent from short_stimuli and long_stimuli")

    screens = []
    story_word_index = 0
    for screen_id in range(1, len(stimulus["screens"]) + 1):
        lines = [line for line in stimulus["lines"] if int(line["screen"]) == screen_id]
        words = []
        boundaries = []
        for line_index, line in enumerate(lines):
            spaces = list(line["spaces_pos"])
            if str(line["text"])[:3] == "   ":
                spaces = spaces[3:]
            tokens = str(line["text"]).split()
            if len(spaces) != len(tokens) + 1:
                raise ValueError(
                    f"{story_key}, screen {screen_id}, line {line_index}: "
                    "word and horizontal-boundary counts differ"
                )
            bbox = line["bbox"]
            y = (float(bbox[1]) + float(bbox[3])) / 2
            boundaries.append(float(bbox[1]) - line_spacing / 2)
            for line_word_index, (token, left, right) in enumerate(
                zip(tokens, spaces, spaces[1:])
            ):
                words.append({
                    "word": token,
                    "story_word_index": story_word_index,
                    "screen_word_index": len(words),
                    "line_index": line_index,
                    "line_word_index": line_word_index,
                    "x_min": float(left),
                    "x_max": float(right),
                    "x_center": (float(left) + float(right)) / 2,
                    "y_center": y,
                })
                story_word_index += 1
        if not lines:
            raise ValueError(f"{story_key}, screen {screen_id} contains no lines")
        boundaries.append(boundaries[-1] + line_spacing)
        for word in words:
            line_index = word["line_index"]
            word["y_min"] = boundaries[line_index]
            word["y_max"] = boundaries[line_index + 1]
        screens.append({
            "screen_index": screen_id - 1,
            "line_boundaries": boundaries,
            "words": words,
        })
    return story_key, screens


def _load_original_geometry(config: CorrelationConfig, story_key: str) -> list[dict]:
    with config.all_stimuli_json.open(encoding="utf-8") as handle:
        all_stimuli = json.load(handle)
    json_story = _resolve_unique_name(story_key, list(all_stimuli), "all_stimuli story")
    screens = []
    for screen_index, screen in enumerate(all_stimuli[json_story]):
        words = [{
            **word,
            "x_center": float(word["x"]),
            "y_center": float(word["y"]),
        } for word in screen["words"]]
        screens.append({"screen_index": screen_index, "words": words})
    return screens


def _json_x_canonical_y_geometry(
    config: CorrelationConfig, story_key: str, canonical_screens: list[dict]
) -> list[dict]:
    """Combine JSON horizontal positions with canonical vertical geometry."""

    json_screens = _load_original_geometry(config, story_key)
    if len(json_screens) != len(canonical_screens):
        raise ValueError("JSON and canonical geometry have different screen counts")
    combined = []
    for screen_index, (json_screen, canonical_screen) in enumerate(
        zip(json_screens, canonical_screens)
    ):
        json_words = json_screen["words"]
        canonical_words = canonical_screen["words"]
        if len(json_words) != len(canonical_words):
            raise ValueError(
                f"Screen {screen_index}: JSON and canonical word counts differ"
            )
        words = []
        for word_index, (json_word, canonical_word) in enumerate(
            zip(json_words, canonical_words)
        ):
            if str(json_word["word"]).casefold() != str(canonical_word["word"]).casefold():
                raise ValueError(
                    f"Screen {screen_index}, word {word_index}: JSON word "
                    f"'{json_word['word']}' != canonical word '{canonical_word['word']}'"
                )
            words.append({**canonical_word, "x_center": json_word["x_center"]})
        combined.append({**canonical_screen, "words": words})
    return combined


def _resolve_processed_example(config: CorrelationConfig, story_key: str) -> Path:
    names = [path.name for path in config.processed_examples.iterdir() if path.is_dir()]
    name = _resolve_unique_name(story_key, names, "Processed-example story")
    return config.processed_examples / name


def _validate_simulation(df: pd.DataFrame, path: Path, screen_count: int) -> str | None:
    missing_columns = REQUIRED_SIMULATION_COLUMNS - set(df.columns)
    if missing_columns:
        return f"missing columns: {sorted(missing_columns)}"
    seen = set(df["text_id"].dropna().astype(int).unique())
    expected = set(range(screen_count))
    if not expected.issubset(seen):
        return f"missing screens: {sorted(expected - seen)}"
    return None


def _canonical_line_boundaries(screen: dict) -> list[float]:
    """Return line bands taken from the canonical stimulus definition."""

    return list(screen["line_boundaries"])


def _create_neutral_trial(destination: Path, screens: list[dict]) -> None:
    """Create only the subject-independent files required by em_analysis.py."""

    destination.mkdir(parents=True)
    pd.DataFrame({"currentscreenid": range(1, len(screens) + 1)}).to_pickle(
        destination / "screen_sequence.pkl"
    )
    pd.DataFrame({"edited": [True], "iswrong": [False]}).to_pickle(
        destination / "flags.pkl"
    )
    # em_analysis always exports the unrelated word-association task. Simulations
    # have no answers, so an empty table is the neutral representation.
    pd.DataFrame({0: pd.Series(dtype=object)}).to_pickle(destination / "words.pkl")
    for screen_id, screen in enumerate(screens, start=1):
        screen_dir = destination / f"screen_{screen_id}"
        screen_dir.mkdir()
        pd.DataFrame({"y": _canonical_line_boundaries(screen)}).to_pickle(
            screen_dir / "lines.pkl"
        )


def _replace_fixations(df: pd.DataFrame, destination: Path, screens: list[dict]) -> None:
    for screen_id in range(len(screens)):
        screen_df = df[df["text_id"] == screen_id]
        rows = []
        elapsed = 0
        words = screens[screen_id]["words"]
        for _, fixation in screen_df.iterrows():
            word_idx = int(fixation["foveal_word_index"])
            if word_idx < 0 or word_idx >= len(words):
                raise IndexError(
                    f"Screen {screen_id}: word index {word_idx} is outside 0..{len(words) - 1}"
                )
            duration = fixation["fixation_duration"]
            start = elapsed
            elapsed += duration
            rows.append({
                "index": fixation["fixation_counter"], "eye": "R",
                "tStart": start, "tEnd": elapsed, "duration": duration,
                "xAvg": words[word_idx]["x_center"],
                "yAvg": words[word_idx]["y_center"],
                "pupilAvg": -1,
            })
        pd.DataFrame(rows).to_pickle(
            destination / f"screen_{screen_id + 1}" / "fixations.pkl"
        )


def _run_em_analysis(
    config: CorrelationConfig,
    story_key: str,
    participants: Path,
    output: Path,
    work: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(config.python_executable), "em_analysis.py", "--item", story_key,
        "--reprocess", "--participants", str(participants),
        "--wordsfix", str(work / "words_fixations"),
        "--stats", str(work / "words_fixations" / "stats.csv"),
        "--measures", str(output / "measures"), "--output", str(output),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _resolve_or_build_human_measures(
    config: CorrelationConfig, story_key: str, experiment_dir: Path
) -> Path:
    if config.human_measures_dir is not None:
        child_dirs = [path for path in config.human_measures_dir.iterdir() if path.is_dir()] \
            if config.human_measures_dir.is_dir() else []
        matching_children = [
            path for path in child_dirs if normalize_name(path.name) == normalize_name(story_key)
        ]
        candidates = [config.human_measures_dir, *matching_children]
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.glob("*.pkl")):
                return candidate
        raise FileNotFoundError(
            f"human_measures_dir has no .pkl files (directly or under '{story_key}'): "
            f"{config.human_measures_dir}"
        )

    cache_story_dir = config.human_results_root / story_key.replace(" ", "_")
    cache_measures_root = cache_story_dir / "measures"
    if cache_measures_root.is_dir():
        matches = [
            path for path in cache_measures_root.iterdir() if path.is_dir()
            and normalize_name(path.name) == normalize_name(story_key)
        ]
        if len(matches) == 1 and any(matches[0].glob("*.pkl")):
            return matches[0]

    slug = story_key.replace(" ", "_")
    cached_root = CORRELATION_DIR / f"results_og_all_{slug}" / "measures"
    if cached_root.is_dir():
        matches = [
            path for path in cached_root.iterdir() if path.is_dir()
            and normalize_name(path.name) == normalize_name(story_key)
        ]
        if len(matches) == 1 and any(matches[0].glob("*.pkl")):
            return matches[0]

    output = cache_story_dir
    participants_view = experiment_dir / "human_work" / "trials"
    participants_view.mkdir(parents=True)
    matched = 0
    for subject in config.original_human_trials.iterdir():
        if not subject.is_dir():
            continue
        trials = [
            trial for trial in subject.iterdir() if trial.is_dir()
            and normalize_name(trial.name) == normalize_name(story_key)
        ]
        if len(trials) == 1:
            subject_view = participants_view / subject.name
            subject_view.mkdir()
            (subject_view / story_key).symlink_to(trials[0], target_is_directory=True)
            matched += 1
    if not matched:
        raise FileNotFoundError(
            f"No original human trials matching '{story_key}' in {config.original_human_trials}"
        )
    _run_em_analysis(
        config, story_key, participants_view, output,
        experiment_dir / "human_work",
    )
    measures = output / "measures" / story_key
    if not measures.is_dir() or not any(measures.glob("*.pkl")):
        raise RuntimeError(f"Human processing produced no measures for '{story_key}'")
    return measures


def _serializable_config(config: CorrelationConfig) -> dict:
    values = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}


def prepare_and_process(config: CorrelationConfig) -> ExperimentRun:
    """Discover simulations, reconstruct trials, and run the original pipeline."""

    try:
        _, simulations = discover_simulations(
            config.story, config.simulation_subjects, config.simulations_root
        )
    except KeyError:
        legacy_root = CORRELATION_DIR / "simulations"
        if config.simulations_root != (REPO_ROOT / "model_output").resolve():
            raise
        _, simulations = discover_simulations(
            config.story, config.simulation_subjects, legacy_root
        )
    story_key, canonical_screens = load_canonical_stimulus(config)
    if config.geometry_mode == "original_compatibility":
        screens = _load_original_geometry(config, story_key)
        processed_example = _resolve_processed_example(config, story_key)
    elif config.geometry_mode == "json_x_canonical_y":
        screens = _json_x_canonical_y_geometry(config, story_key, canonical_screens)
        processed_example = None
    else:
        screens = canonical_screens
        processed_example = None
    stamp = datetime.now().strftime("%Y_%m_%d_%H%M%S_%f")
    experiment_dir = config.experiments_root / f"{stamp}_{story_key.replace(' ', '_')}"
    trials = experiment_dir / "simulation_work" / "trials"
    trials.mkdir(parents=True)

    complete: list[Path] = []
    skipped: list[dict] = []
    for simulation in simulations:
        df = pd.read_csv(simulation, sep="\t")
        reason = _validate_simulation(df, simulation, len(screens))
        if reason:
            skipped.append({"path": str(simulation), "reason": reason})
            continue
        destination = trials / simulation.stem / story_key
        try:
            if processed_example is None:
                _create_neutral_trial(destination, screens)
            else:
                shutil.copytree(processed_example, destination)
            _replace_fixations(df, destination, screens)
        except (IndexError, KeyError, TypeError, ValueError) as error:
            shutil.rmtree(destination)
            skipped.append({"path": str(simulation), "reason": str(error)})
            continue
        complete.append(simulation)

    if not complete:
        raise RuntimeError("None of the selected simulations was complete and valid")

    simulated_output = experiment_dir / "simulation_results"
    _run_em_analysis(
        config, story_key, trials, simulated_output,
        experiment_dir / "simulation_work",
    )
    human_measures = _resolve_or_build_human_measures(config, story_key, experiment_dir)
    run = ExperimentRun(
        config=config, experiment_dir=experiment_dir, story_key=story_key,
        simulation_paths=simulations, complete_simulations=complete,
        skipped_simulations=skipped,
        simulated_measures_dir=simulated_output / "measures" / story_key,
        human_measures_dir=human_measures,
    )
    manifest = {
        "config": _serializable_config(config), "story_key": story_key,
        "selected_simulations": [str(path) for path in simulations],
        "complete_simulations": [str(path) for path in complete],
        "skipped_simulations": skipped,
        "human_measures_dir": str(human_measures),
        "simulated_measures_dir": str(run.simulated_measures_dir),
    }
    with (experiment_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return run
