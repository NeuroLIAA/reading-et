from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io import loadmat


# -----------------------------
# Repo-aware paths
# -----------------------------

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent
DATA_ROOT = REPO_ROOT / "data"
DEFAULT_TRIALS_PATH = DATA_ROOT / "processed" / "trials"
FALLBACK_TRIALS_PATH = DATA_ROOT / "trials"
DEFAULT_STIMULI_PATH = REPO_ROOT / "stimuli"
DEFAULT_OUT_WORDS_FIXATIONS = DATA_ROOT / "processed" / "words_fixations"
DEFAULT_OUT_SCANPATHS = DATA_ROOT / "processed" / "scanpaths"
PROCESSED_ZIP_PATH = DATA_ROOT / "processed.zip"


def resolve_cli_path(raw_path: str | None, *, base_dir: Path, default_path: Path) -> Path:
    if raw_path is None:
        return default_path

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def detect_default_trials_path() -> Path:
    if DEFAULT_TRIALS_PATH.is_dir():
        return DEFAULT_TRIALS_PATH
    if FALLBACK_TRIALS_PATH.is_dir():
        return FALLBACK_TRIALS_PATH
    return DEFAULT_TRIALS_PATH


def find_trials_dir(extract_root: Path) -> Path | None:
    candidates = [
        extract_root / "trials",
        extract_root / "processed" / "trials",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def extract_trials_archive(zip_path: Path, extract_root: Path) -> Path:
    print("Trials directory not found. Extracting archive...")
    print(f"  source: {zip_path}")
    print(f"  destination: {extract_root}")

    extract_root.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        archive.extractall(extract_root)

    extracted_trials_path = find_trials_dir(extract_root)
    if extracted_trials_path is None:
        raise SystemExit(
            "Archive extraction completed, but no trials directory was found.\n"
            f"Archive: {zip_path}\n"
            f"Extraction root: {extract_root}\n"
            "Expected archive contents to include either 'trials/' or 'processed/trials/'."
        )

    print(f"  extracted trials_path: {extracted_trials_path}")
    print()
    return extracted_trials_path


def prepare_trials_path(trials_path: Path) -> Path:
    if trials_path.is_dir():
        return trials_path

    if trials_path.suffix.lower() == ".zip":
        return extract_trials_archive(trials_path, trials_path.with_suffix(""))

    sibling_zip = trials_path.parent.with_suffix(".zip")
    if sibling_zip.is_file():
        return extract_trials_archive(sibling_zip, trials_path.parent)

    if trials_path == FALLBACK_TRIALS_PATH and PROCESSED_ZIP_PATH.is_file():
        return extract_trials_archive(PROCESSED_ZIP_PATH, DATA_ROOT / "processed")

    return trials_path


def validate_inputs(
    trials_path: Path,
    stimuli_path: Path,
    out_words_fixations: Path,
    out_scanpaths: Path,
    subj: str,
    item: str,
) -> None:
    if not stimuli_path.is_dir():
        raise SystemExit(
            "Stimuli directory not found.\n"
            f"Resolved stimuli path: {stimuli_path}\n"
            f"Expected default path: {DEFAULT_STIMULI_PATH}"
        )

    if not trials_path.is_dir():
        if PROCESSED_ZIP_PATH.is_file():
            raise SystemExit(
                "Trials directory not found.\n"
                f"Resolved trials path: {trials_path}\n"
                f"Found ZIP archive: {PROCESSED_ZIP_PATH}\n"
                f"Automatic extraction should have created: {DEFAULT_TRIALS_PATH}\n"
                "If the problem persists, verify that the ZIP contains a top-level 'trials/' folder."
            )
        raise SystemExit(
            "Trials directory not found.\n"
            f"Resolved trials path: {trials_path}\n"
            f"Looked for default paths:\n"
            f"  - {DEFAULT_TRIALS_PATH}\n"
            f"  - {FALLBACK_TRIALS_PATH}"
        )

    if subj != "all":
        subj_path = trials_path / subj
        if not subj_path.is_dir():
            raise SystemExit(f"Subject directory not found: {subj_path}")

    if item != "all":
        item_path = stimuli_path / f"{item}.mat"
        if not item_path.is_file():
            raise SystemExit(f"Stimulus file not found for item '{item}': {item_path}")

    out_words_fixations.mkdir(parents=True, exist_ok=True)
    out_scanpaths.mkdir(parents=True, exist_ok=True)


def print_resolved_paths(
    trials_path: Path,
    stimuli_path: Path,
    out_words_fixations: Path,
    out_scanpaths: Path,
) -> None:
    print("Resolved paths:")
    print(f"  repo_root: {REPO_ROOT}")
    print(f"  data_root: {DATA_ROOT}")
    print(f"  trials_path: {trials_path}")
    print(f"  stimuli_path: {stimuli_path}")
    print(f"  out_words_fixations: {out_words_fixations}")
    print(f"  out_scanpaths: {out_scanpaths}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build word-fixation alignments and scanpaths from Reading-ET processed trials.",
        epilog=(
            "Examples:\n"
            "  python build_scanpaths_from_trials.py\n"
            "  python build_scanpaths_from_trials.py --subj sub-073 --item \"Wakefield\"\n"
            "  python build_scanpaths_from_trials.py --trials_path data/processed/trials\n"
            "  python build_scanpaths_from_trials.py --trials_path data/processed.zip\n"
            "  python build_scanpaths_from_trials.py --out_scanpaths data/processed/scanpaths"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trials_path",
        type=str,
        default=None,
        help=(
            "Path to processed trials. Absolute paths are used as-is. "
            f"Relative paths resolve from {REPO_ROOT}. "
            f"Default auto-detects {DEFAULT_TRIALS_PATH} and then {FALLBACK_TRIALS_PATH}. "
            "You can also point this argument to a .zip file; it will be extracted automatically."
        ),
    )
    parser.add_argument(
        "--stimuli_path",
        type=str,
        default=None,
        help=(
            "Path to the stimuli .mat files. Absolute paths are used as-is. "
            f"Relative paths resolve from {REPO_ROOT}. Default: {DEFAULT_STIMULI_PATH}"
        ),
    )
    parser.add_argument(
        "--out_words_fixations",
        type=str,
        default=None,
        help=(
            "Where to save one .pkl per subject/item with fixation->word alignment. "
            f"Relative paths resolve from {REPO_ROOT}. Default: {DEFAULT_OUT_WORDS_FIXATIONS}"
        ),
    )
    parser.add_argument(
        "--out_scanpaths",
        type=str,
        default=None,
        help=(
            "Where to save one .json per subject/item with the reconstructed scanpath. "
            f"Relative paths resolve from {REPO_ROOT}. Default: {DEFAULT_OUT_SCANPATHS}"
        ),
    )
    parser.add_argument("--subj", type=str, default="all", help="Subject name or 'all'")
    parser.add_argument("--item", type=str, default="all", help="Item name or 'all'")
    parser.add_argument("--reprocess", action="store_true", help="Overwrite outputs if they already exist")
    return parser.parse_args()


# -----------------------------
# Stimuli helpers
# -----------------------------

def load_matfile(path: Path):
    return loadmat(str(path), simplify_cells=True)


def load_lines_by_screen(stimulus_mat: Path) -> Dict[int, List[dict]]:
    item_cfg = load_matfile(stimulus_mat)
    lines = item_cfg["lines"]
    num_screens = len(item_cfg["screens"])
    screens_lines = {screen_id: [] for screen_id in range(1, num_screens + 1)}

    for line in lines:
        screens_lines[int(line["screen"])].append(
            {
                "text": line["text"],
                "spaces_pos": np.asarray(line["spaces_pos"]),
            }
        )
    return screens_lines


def build_item_words(screens_lines: Dict[int, List[dict]]) -> List[str]:
    words: List[str] = []
    for screen_id in sorted(screens_lines):
        for line in screens_lines[screen_id]:
            words.extend(line["text"].split())
    return words


def build_screen_word_offsets(screens_lines: Dict[int, List[dict]]) -> Dict[int, int]:
    offsets: Dict[int, int] = {}
    running = 0
    for screen_id in sorted(screens_lines):
        offsets[screen_id] = running
        running += sum(len(line["text"].split()) for line in screens_lines[screen_id])
    return offsets


# -----------------------------
# Trial helpers
# -----------------------------

def trial_is_correct(trial_path: Path) -> bool:
    flags_file = trial_path / "flags.pkl"
    if not flags_file.exists():
        return False
    flags = pd.read_pickle(flags_file)
    return bool(flags["edited"].iloc[0]) and not bool(flags["iswrong"].iloc[0])


def get_screen_filenames(screen_times_read: int):
    fix_filename = "fixations.pkl"
    lines_filename = "lines.pkl"
    if screen_times_read > 0:
        fix_filename = f"fixations_{screen_times_read}.pkl"
        lines_filename = f"lines_{screen_times_read}.pkl"
    return fix_filename, lines_filename


def get_last_fixation_index(screen_dir: Path, prev_screen_times_read: int) -> int:
    last_fixation_index = 0
    for it in range(prev_screen_times_read):
        fix_filename, _ = get_screen_filenames(it)
        fixations = pd.read_pickle(screen_dir / fix_filename)
        last_fixation_index += int(fixations.iloc[-1].name)
    return last_fixation_index


def load_screen_data(trial_path: Path, screen_id: int, screen_counter: Dict[int, int]):
    screen_dir = trial_path / f"screen_{screen_id}"
    fix_filename, lines_filename = get_screen_filenames(screen_counter[screen_id])
    fixations = pd.read_pickle(screen_dir / fix_filename)
    lines_pos = pd.read_pickle(screen_dir / lines_filename).sort_values("y")["y"].to_numpy()

    if screen_counter[screen_id] > 0:
        last_fixation_index = get_last_fixation_index(screen_dir, screen_counter[screen_id])
        fixations = fixations.copy()
        fixations.index += last_fixation_index + 1

    return fixations, lines_pos


def get_line_fixations(fixations: pd.DataFrame, line_number: int, lines_pos: np.ndarray) -> pd.DataFrame:
    line_fixations = fixations[
        fixations["yAvg"].between(lines_pos[line_number], lines_pos[line_number + 1], inclusive="left")
    ]
    if not line_fixations.empty:
        if int(line_fixations.iloc[0].name) == 0:
            line_fixations = line_fixations.drop([0])
        if not line_fixations.empty and int(line_fixations.iloc[-1].name) == len(fixations) - 1:
            line_fixations = line_fixations.drop([len(fixations) - 1])
    return line_fixations


def assign_line_fixations_to_words(
    word_pos: int,
    line_fix: pd.DataFrame,
    line_num: int,
    spaces_pos: np.ndarray,
    screen_id: int,
    subj_name: str,
    out_rows: List[list],
) -> None:
    for idx in range(len(spaces_pos) - 1):
        word_fix = line_fix[
            line_fix["xAvg"].between(spaces_pos[idx], spaces_pos[idx + 1], inclusive="left")
        ]
        if word_fix.empty:
            out_rows.append([subj_name, screen_id, line_num, word_pos, None, None, None, None])
        else:
            word_fix = word_fix[["index", "duration", "xAvg"]].copy()
            word_fix = word_fix.rename(columns={"index": "trial_fix", "xAvg": "x"})
            word_fix.reset_index(names="screen_fix", inplace=True)
            word_fix["x"] -= spaces_pos[idx]
            word_fix["subj"] = subj_name
            word_fix["screen"] = screen_id
            word_fix["line"] = line_num
            word_fix["word_pos"] = word_pos
            word_fix = word_fix[["subj", "screen", "line", "word_pos", "trial_fix", "screen_fix", "duration", "x"]]
            out_rows.extend(word_fix.values.tolist())
        word_pos += 1


def n_fix(df_fix: pd.DataFrame) -> int:
    return len(df_fix[~df_fix["screen_fix"].isna()])


def is_regression(df_fix: pd.DataFrame, fix_num: int, following_fix_num: int) -> bool:
    first_fix = df_fix[df_fix["screen_fix"] == fix_num]
    second_fix = df_fix[df_fix["screen_fix"] == following_fix_num]
    if first_fix.empty or second_fix.empty:
        return False
    return (
        first_fix["word_pos"].iloc[0] > second_fix["word_pos"].iloc[0]
        or (
            first_fix["word_pos"].iloc[0] == second_fix["word_pos"].iloc[0]
            and first_fix["x"].iloc[0] > second_fix["x"].iloc[0]
        )
    )


def remove_return_sweeps_from_line(line_fix: pd.DataFrame) -> pd.DataFrame:
    fst_fix_num = line_fix["screen_fix"].min()
    if pd.isna(fst_fix_num):
        return line_fix

    first_saccade_is_regressive = is_regression(line_fix, int(fst_fix_num), int(fst_fix_num) + 1)
    if first_saccade_is_regressive:
        first_word_with_fix = line_fix[~line_fix["screen_fix"].isna()]["word_pos"].min()
        if not pd.isna(first_word_with_fix):
            first_word_fix = line_fix[line_fix["word_pos"] == first_word_with_fix]
            left_most_fix = first_word_fix[first_word_fix["x"] == first_word_fix["x"].min()]
            line_fix = line_fix.copy()
            line_fix.loc[
                line_fix["screen_fix"].between(
                    int(fst_fix_num),
                    int(left_most_fix["screen_fix"].iloc[0]),
                    inclusive="left",
                ),
                ["trial_fix", "screen_fix", "duration", "x"],
            ] = np.nan
    return line_fix


def remove_na_from_fixated_words(words_fix: pd.DataFrame) -> pd.DataFrame:
    if n_fix(words_fix) > 0:
        return words_fix.dropna()
    return words_fix.head(1)


def make_screen_fix_consecutive(trial_fix_by_word: pd.DataFrame) -> pd.DataFrame:
    trial_fix_by_word = trial_fix_by_word.sort_values(["screen", "screen_fix"])
    consecutive = trial_fix_by_word[~trial_fix_by_word["screen_fix"].isna()].copy()
    consecutive["screen_fix"] = consecutive.groupby("screen").cumcount()
    trial_fix_by_word.update(consecutive)
    return trial_fix_by_word


def cast_to_int(trial_fix_by_word: pd.DataFrame) -> pd.DataFrame:
    for col in ["screen", "line", "word_pos", "trial_fix", "screen_fix", "duration"]:
        trial_fix_by_word[col] = trial_fix_by_word[col].astype(pd.Int64Dtype())
    return trial_fix_by_word


def postprocess_word_fixations(trial_fix_by_word: pd.DataFrame) -> pd.DataFrame:
    trial_fix_by_word = (
        trial_fix_by_word.groupby(["screen", "line"], group_keys=False)[trial_fix_by_word.columns]
        .apply(remove_return_sweeps_from_line)
    )
    trial_fix_by_word = (
        trial_fix_by_word.groupby(["screen", "word_pos"], group_keys=False)[trial_fix_by_word.columns]
        .apply(remove_na_from_fixated_words)
    )
    trial_fix_by_word = make_screen_fix_consecutive(trial_fix_by_word)
    trial_fix_by_word = cast_to_int(trial_fix_by_word)
    trial_fix_by_word = trial_fix_by_word.sort_values(["screen", "line", "word_pos", "screen_fix"])
    return trial_fix_by_word


def process_trial_to_word_fixations(trial_path: Path, screens_lines: Dict[int, List[dict]], subj_name: str) -> pd.DataFrame:
    screen_sequence = pd.read_pickle(trial_path / "screen_sequence.pkl")["currentscreenid"].to_numpy()
    screen_counter = {int(screen_id): 0 for screen_id in np.unique(screen_sequence)}
    rows: List[list] = []

    for screen_id in screen_sequence:
        screen_id = int(screen_id)
        fixations, lines_pos = load_screen_data(trial_path, screen_id, screen_counter)
        word_pos = 0
        for line_num, line in enumerate(screens_lines[screen_id]):
            words = line["text"].split()
            spaces_pos = np.asarray(line["spaces_pos"])
            if line["text"][:3] == "   ":
                spaces_pos = spaces_pos[3:]
            line_fix = get_line_fixations(fixations, line_num, lines_pos)
            assign_line_fixations_to_words(word_pos, line_fix, line_num, spaces_pos, screen_id, subj_name, rows)
            word_pos += len(words)
        screen_counter[screen_id] += 1

    trial_fix_by_word = pd.DataFrame(
        rows,
        columns=["subj", "screen", "line", "word_pos", "trial_fix", "screen_fix", "duration", "x"],
    )
    return postprocess_word_fixations(trial_fix_by_word)


# -----------------------------
# Scanpath building
# -----------------------------

def build_scanpath_from_word_fixations(
    trial_fix_by_word: pd.DataFrame,
    screens_lines: Dict[int, List[dict]],
    item_name: str,
) -> dict:
    item_words = build_item_words(screens_lines)
    screen_offsets = build_screen_word_offsets(screens_lines)

    fixated = trial_fix_by_word.dropna(subset=["trial_fix", "duration"]).copy()
    if fixated.empty:
        subj_name = str(trial_fix_by_word["subj"].iloc[0]) if not trial_fix_by_word.empty else ""
        return {
            "item": item_name,
            "subj": subj_name,
            "n_fixations": 0,
            "word_ids": [],
            "words": [],
            "trial_fix": [],
            "durations": [],
            "screens": [],
            "screen_fix": [],
        }

    fixated["screen"] = fixated["screen"].astype(int)
    fixated["word_pos"] = fixated["word_pos"].astype(int)
    fixated["global_word_idx"] = fixated.apply(
        lambda row: screen_offsets[int(row["screen"])] + int(row["word_pos"]),
        axis=1,
    )
    fixated = fixated.sort_values("trial_fix")

    word_ids = fixated["global_word_idx"].astype(int).tolist()
    words = [item_words[idx] for idx in word_ids]

    return {
        "item": item_name,
        "subj": str(fixated["subj"].iloc[0]),
        "n_fixations": len(word_ids),
        "word_ids": word_ids,
        "words": words,
        "trial_fix": fixated["trial_fix"].astype(int).tolist(),
        "durations": fixated["duration"].astype(int).tolist(),
        "screens": fixated["screen"].astype(int).tolist(),
        "screen_fix": fixated["screen_fix"].astype(int).tolist(),
    }


# -----------------------------
# Main driver
# -----------------------------

def iter_subjects(trials_path: Path, subj: str) -> Iterable[Path]:
    if subj != "all":
        candidate = trials_path / subj
        if candidate.is_dir():
            yield candidate
        return

    for candidate in sorted(trials_path.iterdir()):
        if candidate.is_dir():
            yield candidate


def process_dataset(
    trials_path: Path,
    stimuli_path: Path,
    out_words_fixations: Path,
    out_scanpaths: Path,
    subj: str = "all",
    item: str = "all",
    reprocess: bool = False,
) -> None:
    n_done = 0
    n_skipped = 0
    n_errors = 0

    for subj_dir in iter_subjects(trials_path, subj):
        for trial_path in sorted(path for path in subj_dir.iterdir() if path.is_dir()):
            item_name = trial_path.name
            if item != "all" and item_name != item:
                continue

            flags_file = trial_path / "flags.pkl"
            if not flags_file.exists():
                print(f"[skip] {subj_dir.name}/{item_name}: missing flags.pkl")
                n_skipped += 1
                continue

            if not trial_is_correct(trial_path):
                print(f"[skip] {subj_dir.name}/{item_name}: flags say trial is not correct")
                n_skipped += 1
                continue

            stimulus_mat = stimuli_path / f"{item_name}.mat"
            if not stimulus_mat.exists():
                print(f"[skip] {subj_dir.name}/{item_name}: missing stimulus {stimulus_mat.name}")
                n_skipped += 1
                continue

            item_out_fix = out_words_fixations / item_name
            item_out_sp = out_scanpaths / item_name
            item_out_fix.mkdir(parents=True, exist_ok=True)
            item_out_sp.mkdir(parents=True, exist_ok=True)

            fix_out = item_out_fix / f"{subj_dir.name}.pkl"
            sp_out = item_out_sp / f"{subj_dir.name}.json"
            if not reprocess and fix_out.exists() and sp_out.exists():
                print(f"[skip] {subj_dir.name}/{item_name}: already processed")
                continue

            try:
                screens_lines = load_lines_by_screen(stimulus_mat)
                trial_fix_by_word = process_trial_to_word_fixations(trial_path, screens_lines, subj_dir.name)
                scanpath = build_scanpath_from_word_fixations(trial_fix_by_word, screens_lines, item_name)
            except FileNotFoundError as exc:
                print(f"[error] {subj_dir.name}/{item_name}: missing file -> {exc}")
                n_errors += 1
                continue
            except KeyError as exc:
                print(f"[error] {subj_dir.name}/{item_name}: unexpected trial structure -> missing key {exc}")
                n_errors += 1
                continue
            except Exception as exc:
                print(f"[error] {subj_dir.name}/{item_name}: {exc}")
                n_errors += 1
                continue

            trial_fix_by_word.to_pickle(fix_out)
            with sp_out.open("w", encoding="utf-8") as handle:
                json.dump(scanpath, handle, ensure_ascii=False)

            print(f"[ok] {subj_dir.name}/{item_name}: {scanpath['n_fixations']} fixations in scanpath")
            n_done += 1

    print(f"\nFinished. Processed: {n_done} | Skipped: {n_skipped} | Errors: {n_errors}")


def main() -> None:
    args = parse_args()

    trials_path = resolve_cli_path(args.trials_path, base_dir=REPO_ROOT, default_path=detect_default_trials_path())
    trials_path = prepare_trials_path(trials_path)
    stimuli_path = resolve_cli_path(args.stimuli_path, base_dir=REPO_ROOT, default_path=DEFAULT_STIMULI_PATH)
    out_words_fixations = resolve_cli_path(
        args.out_words_fixations,
        base_dir=REPO_ROOT,
        default_path=DEFAULT_OUT_WORDS_FIXATIONS,
    )
    out_scanpaths = resolve_cli_path(
        args.out_scanpaths,
        base_dir=REPO_ROOT,
        default_path=DEFAULT_OUT_SCANPATHS,
    )

    validate_inputs(
        trials_path=trials_path,
        stimuli_path=stimuli_path,
        out_words_fixations=out_words_fixations,
        out_scanpaths=out_scanpaths,
        subj=args.subj,
        item=args.item,
    )
    print_resolved_paths(trials_path, stimuli_path, out_words_fixations, out_scanpaths)

    process_dataset(
        trials_path=trials_path,
        stimuli_path=stimuli_path,
        out_words_fixations=out_words_fixations,
        out_scanpaths=out_scanpaths,
        subj=args.subj,
        item=args.item,
        reprocess=args.reprocess,
    )


if __name__ == "__main__":
    main()
