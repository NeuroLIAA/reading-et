"""Correlation heatmaps adapted from the final plots in the original notebook."""

from __future__ import annotations

import math
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .analysis import CorrelationResults


def _safe_name(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("_")


def plot_overview(results: CorrelationResults):
    measures = results.run.config.measures
    ncols = min(3, len(measures))
    nrows = math.ceil(len(measures) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), squeeze=False,
        sharex=True, sharey=True,
    )
    for index, measure in enumerate(measures):
        ax = axes.flat[index]
        sns.heatmap(
            results.matrices[measure], ax=ax, cmap="coolwarm", center=0,
            vmin=results.run.config.plot_min, vmax=results.run.config.plot_max,
            cbar=index == 0,
        )
        ax.set_title(measure)
    for ax in axes.flat[len(measures):]:
        ax.set_visible(False)
    fig.suptitle(f"{results.run.story_key} — {results.run.config.n_bins} bins")
    fig.tight_layout()
    path = results.run.correlation_dir / f"{_safe_name(results.run.story_key)}_overview.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def _cluster_orders(
    matrix: np.ndarray, regular_indices: list[int]
) -> tuple[list[int], list[int]]:
    """Cluster rows/regular columns, falling back cleanly for tiny matrices."""

    basis_indices = regular_indices or list(range(matrix.shape[1]))
    basis = matrix[:, basis_indices]
    row_order = list(range(matrix.shape[0]))
    regular_order = list(regular_indices)
    if not basis.size:
        return row_order, regular_order
    try:
        cluster = sns.clustermap(
            basis,
            row_cluster=matrix.shape[0] > 1,
            col_cluster=len(basis_indices) > 1,
            xticklabels=False,
            yticklabels=False,
            figsize=(6, 6),
        )
        if matrix.shape[0] > 1:
            row_order = list(cluster.dendrogram_row.reordered_ind)
        if regular_indices and len(regular_indices) > 1:
            regular_order = [
                regular_indices[index]
                for index in cluster.dendrogram_col.reordered_ind
            ]
        plt.close(cluster.fig)
    except (FloatingPointError, ValueError):
        plt.close("all")
    # Preserve the original final plot: best mean simulation is rightmost.
    regular_order = sorted(
        regular_order, key=lambda index: matrix[row_order, index].mean()
    )
    return row_order, regular_order


def plot_clustered_correlation_matrices(
    results: CorrelationResults,
    reference_simulations: tuple[str, ...] = ("ob1-001",),
) -> dict[str, object]:
    """Create the original notebook's final clustered plot for each measure."""

    figures = {}
    comparison_names = np.asarray(results.comparison_names)
    reference_names = [
        name for name in reference_simulations if name in comparison_names
    ]
    special_names = reference_names + ["human_mean"]
    special_indices = {
        name: int(np.where(comparison_names == name)[0][0])
        for name in special_names if name in comparison_names
    }
    regular_indices = [
        index for index, name in enumerate(comparison_names)
        if name not in special_indices
    ]

    for measure in results.run.config.measures:
        matrix = results.matrices[measure]
        row_order, regular_order = _cluster_orders(matrix, regular_indices)
        ordered_blocks = []
        ordered_labels = []

        if regular_order:
            ordered_blocks.append(matrix[row_order][:, regular_order])
            ordered_labels.extend(comparison_names[regular_order].tolist())
        for name in reference_names:
            ordered_blocks.append(matrix[row_order, special_indices[name]].reshape(-1, 1))
            ordered_labels.append(name)

        human_corr_mean = (
            results.human_correlation_means.loc[results.human_names, measure]
            .to_numpy()[row_order].reshape(-1, 1)
        )
        ordered_blocks.append(human_corr_mean)
        ordered_labels.append("human_corr_mean")

        if "human_mean" in special_indices:
            ordered_blocks.append(
                matrix[row_order, special_indices["human_mean"]].reshape(-1, 1)
            )
            ordered_labels.append("human_mean")

        augmented = np.concatenate(ordered_blocks, axis=1)
        column_means = augmented.mean(axis=0)
        xlabels = [
            f"{name} ({mean:.2f})"
            for name, mean in zip(ordered_labels, column_means)
        ]
        ylabels = np.asarray(results.human_names)[row_order].tolist()

        width = max(7, augmented.shape[1] * 0.38)
        height = max(5, augmented.shape[0] * 0.14)
        fig, ax = plt.subplots(figsize=(width, height))
        sns.heatmap(
            augmented,
            ax=ax,
            cmap="coolwarm",
            center=0,
            vmin=results.run.config.plot_min,
            vmax=results.run.config.plot_max,
            cbar=True,
            xticklabels=xlabels,
            yticklabels=ylabels,
        )

        special_start = len(regular_order)
        if special_start:
            ax.axvline(special_start, color="black", linewidth=2)
        for special_name in ("human_corr_mean", "human_mean"):
            if special_name in ordered_labels:
                position = ordered_labels.index(special_name)
                if position:
                    ax.axvline(position, color="black", linewidth=2)

        ax.set_title(f"{results.run.story_key} - {measure}")
        ax.set_xlabel("OB1 subjects")
        ax.set_ylabel("Human subjects")
        ax.tick_params(axis="x", labelrotation=90, labelsize=6)
        ax.tick_params(axis="y", labelrotation=0, labelsize=6)
        fig.tight_layout()
        filename = f"{_safe_name(results.run.story_key)}_clustered_{measure}.png"
        fig.savefig(results.run.correlation_dir / filename, dpi=300, bbox_inches="tight")
        figures[measure] = fig
    return figures


def create_all_plots(results: CorrelationResults) -> dict[str, object]:
    """Save the overview and the original-style final clustered heatmaps."""

    figures = {"overview": plot_overview(results)}
    figures.update(plot_clustered_correlation_matrices(results))
    return figures


# Backwards-compatible public name: these are now the clustered final plots.
plot_measure_heatmaps = plot_clustered_correlation_matrices
