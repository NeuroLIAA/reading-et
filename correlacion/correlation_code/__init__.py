"""Public interface for the reusable correlation workflow."""

from .analysis import CorrelationResults, compute_correlations
from .pipeline import (
    CorrelationConfig,
    ExperimentRun,
    available_stories,
    discover_simulations,
    load_canonical_stimulus,
    prepare_and_process,
)
from .plots import create_all_plots, plot_measure_heatmaps, plot_overview

__all__ = [
    "CorrelationConfig",
    "CorrelationResults",
    "ExperimentRun",
    "available_stories",
    "compute_correlations",
    "create_all_plots",
    "discover_simulations",
    "load_canonical_stimulus",
    "plot_measure_heatmaps",
    "plot_overview",
    "prepare_and_process",
]
