"""Correlation calculations extracted from correlation_original.ipynb."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .pipeline import ExperimentRun


IDENTITY_COLUMNS = ["screen", "word_idx", "word", "sentence_idx", "sentence_pos", "screen_pos"]


@dataclass
class CorrelationResults:
    run: ExperimentRun
    matrices: dict[str, np.ndarray]
    human_names: list[str]
    comparison_names: list[str]
    human_correlation_means: pd.DataFrame
    correlations: pd.DataFrame
    subject_mapping: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "story": self.run.story_key,
            "human_subjects": len(self.human_names),
            "simulation_subjects": len(self.comparison_names) - 1,
            "retained_aligned_rows": self._aligned_rows,
            "measures": ", ".join(self.run.config.measures),
            "correlation_rows": len(self.correlations),
        }])

    @property
    def _aligned_rows(self) -> int:
        return int(self.correlations.attrs.get("aligned_rows", 0))


def quantile_discretize(subject: pd.DataFrame, measure: str, n_bins: int) -> pd.DataFrame:
    """Preserve the notebook's qcut behavior (formerly called binarize_measure)."""

    mask = subject[measure] > 0
    subject = subject.astype({measure: np.float32})
    subject.loc[:, measure] = pd.qcut(
        subject.loc[mask, measure], n_bins, labels=False, duplicates="drop"
    )
    subject.loc[:, measure] = subject[measure].fillna(0).astype(np.int32)
    return subject


def _load_subjects(directory: Path, excluded_names: set[str]) -> list[tuple[str, pd.DataFrame]]:
    paths = sorted(path for path in directory.glob("*.pkl") if path.stem not in excluded_names)
    if not paths:
        raise FileNotFoundError(f"No retained subject measure files found in {directory}")
    return [(path.stem, pd.read_pickle(path)) for path in paths]


def _validate_and_filter(
    subjects: list[tuple[str, pd.DataFrame]], measures: list[str], label: str
) -> list[tuple[str, pd.DataFrame]]:
    required = set(IDENTITY_COLUMNS + ["excluded"] + measures)
    output = []
    for name, subject in subjects:
        missing = required - set(subject.columns)
        if missing:
            raise ValueError(f"{label} subject {name} is missing columns: {sorted(missing)}")
        output.append((name, subject.loc[~subject["excluded"]].reset_index(drop=True).copy()))
    return output


def _validate_word_alignment(
    humans: list[tuple[str, pd.DataFrame]], simulations: list[tuple[str, pd.DataFrame]]
) -> int:
    reference_name, reference = humans[0]
    reference_identity = reference[IDENTITY_COLUMNS]
    for label, subjects in (("human", humans[1:]), ("simulation", simulations)):
        for name, subject in subjects:
            if len(subject) != len(reference) or not subject[IDENTITY_COLUMNS].equals(reference_identity):
                raise ValueError(
                    f"Word alignment mismatch: {label} '{name}' does not match human "
                    f"'{reference_name}' in ordered identity columns {IDENTITY_COLUMNS}"
                )
    return len(reference)


def _discretize(
    subjects: list[tuple[str, pd.DataFrame]], measures: list[str], n_bins: int
) -> list[tuple[str, pd.DataFrame]]:
    output = []
    for name, subject in subjects:
        for measure in measures:
            subject = quantile_discretize(subject, measure, n_bins)
        output.append((name, subject))
    return output


def _human_mean(humans: list[tuple[str, pd.DataFrame]], measures: list[str]) -> tuple[str, pd.DataFrame]:
    mean = pd.DataFrame()
    for measure in measures:
        mean[measure] = pd.concat(
            [subject[measure] for _, subject in humans], axis=1
        ).mean(axis=1)
    return "human_mean", mean


def _human_correlation_means(
    humans: list[tuple[str, pd.DataFrame]], measures: list[str]
) -> pd.DataFrame:
    """Mean correlation of each human with every other retained human."""

    output = pd.DataFrame(index=[name for name, _ in humans], columns=measures, dtype=float)
    for measure in measures:
        for human_index, (human_name, human) in enumerate(humans):
            values = []
            for other_index, (_, other) in enumerate(humans):
                if human_index == other_index:
                    continue
                if human[measure].nunique() < 2 or other[measure].nunique() < 2:
                    values.append(0.0)
                else:
                    value = float(spearmanr(human[measure], other[measure])[0])
                    values.append(0.0 if np.isnan(value) else value)
            output.loc[human_name, measure] = float(np.mean(values)) if values else 0.0
    return output


def compute_correlations(run: ExperimentRun) -> CorrelationResults:
    """Validate alignment, discretize measures, correlate, and save CSV tables."""

    measures = run.config.measures
    humans = _validate_and_filter(
        _load_subjects(run.human_measures_dir, set(run.config.excluded_humans)),
        measures, "Human",
    )
    simulations = _validate_and_filter(
        _load_subjects(run.simulated_measures_dir, set()), measures, "Simulation"
    )
    aligned_rows = _validate_word_alignment(humans, simulations)
    humans = _discretize(humans, measures, run.config.n_bins)
    simulations = _discretize(simulations, measures, run.config.n_bins)
    comparisons = simulations + [_human_mean(humans, measures)]
    human_correlation_means = _human_correlation_means(humans, measures)

    matrices: dict[str, np.ndarray] = {}
    rows = []
    for measure in measures:
        matrix = np.zeros((len(humans), len(comparisons)))
        for i, (human_name, human) in enumerate(humans):
            for j, (comparison_name, comparison) in enumerate(comparisons):
                if human[measure].nunique() < 2 or comparison[measure].nunique() < 2:
                    value = 0.0
                else:
                    value = float(spearmanr(human[measure], comparison[measure])[0])
                    value = 0.0 if np.isnan(value) else value
                matrix[i, j] = value
                rows.append({
                    "item": run.story_key, "subject_1": human_name,
                    "subject_2": comparison_name, "measure": measure,
                    "correlation": value,
                })
        matrices[measure] = matrix

    human_names = [name for name, _ in humans]
    comparison_names = [name for name, _ in comparisons]
    mapping_rows = [
        {"item": run.story_key, "dataset": dataset, "index": index, "subject_name": name}
        for dataset, names in (("human", human_names), ("simulation", comparison_names))
        for index, name in enumerate(names)
    ]
    correlations = pd.DataFrame(rows)
    correlations.attrs["aligned_rows"] = aligned_rows
    mapping = pd.DataFrame(mapping_rows)
    run.correlation_dir.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(run.correlation_dir / "correlation_results.csv", index=False)
    mapping.to_csv(run.correlation_dir / "subject_mapping.csv", index=False)
    return CorrelationResults(
        run, matrices, human_names, comparison_names, human_correlation_means,
        correlations, mapping
    )
