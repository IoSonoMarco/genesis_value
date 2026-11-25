"""
Create a schematic temporal-valence query-key retrieval illustration.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

sys.path.insert(0, str(Path.cwd()))

from utils import filepaths


TEMPERATURE = 0.5
TEMPORAL_WEIGHT = 0.25
VALENCE_WEIGHT = 0.75
POSITIVE_COLOR = '#3D8C4A'
NEGATIVE_COLOR = '#C74343'
NEUTRAL_COLOR = '#8C98A4'
MUTED_COLOR = '#D7DDE2'
TEXT_COLOR = '#1F2933'


@dataclass(frozen=True)
class RetrievalProbabilities:
    """
    Schematic probability profiles with shape ``[100]`` per vector.
    """

    element_ids: np.ndarray
    temporal_only: np.ndarray
    positive_query: np.ndarray
    negative_query: np.ndarray
    positive_matches: np.ndarray
    negative_matches: np.ndarray


def softmax(*, scores: np.ndarray) -> np.ndarray:
    """
    Convert schematic query-key scores with shape ``[N]`` into probabilities.
    """
    stabilized_scores = scores / TEMPERATURE
    stabilized_scores -= stabilized_scores.max()
    exponentials = np.exp(stabilized_scores)
    return exponentials / exponentials.sum()


def build_retrieval_probabilities() -> RetrievalProbabilities:
    """
    Build schematic temporal and temporal-valence probability profiles.
    """
    element_ids = np.arange(1, 101)
    temporal_scores = np.linspace(-1.0, 1.0, element_ids.size)
    positive_matches = ((element_ids - 1) // 10) % 2 == 0
    negative_matches = ~positive_matches

    positive_valence_scores = np.where(positive_matches, 1.0, -1.0)
    negative_valence_scores = np.where(negative_matches, 1.0, -1.0)
    positive_query_scores = TEMPORAL_WEIGHT * temporal_scores + VALENCE_WEIGHT * positive_valence_scores
    negative_query_scores = TEMPORAL_WEIGHT * temporal_scores + VALENCE_WEIGHT * negative_valence_scores

    return RetrievalProbabilities(
        element_ids=element_ids,
        temporal_only=softmax(scores=temporal_scores),
        positive_query=softmax(scores=positive_query_scores),
        negative_query=softmax(scores=negative_query_scores),
        positive_matches=positive_matches,
        negative_matches=negative_matches,
    )


def plot_probability_panel(
    *,
    axis: Axes,
    element_ids: np.ndarray,
    probabilities: np.ndarray,
    colors: np.ndarray,
    title: str,
    y_max: float,
) -> None:
    """
    Plot a query-key probability profile with shape ``[100]``.
    """
    axis.bar(element_ids, probabilities, color=colors, width=0.85, linewidth=0)
    axis.set_xlim(0.5, 100.5)
    axis.set_ylim(0.0, y_max)
    axis.set_xticks([1, 50, 100])
    axis.set_xlabel('Element ID', color=TEXT_COLOR, fontsize=9)
    axis.set_yticks([])
    axis.tick_params(axis='x', colors=TEXT_COLOR, labelsize=8, length=3)
    axis.spines[['left', 'top', 'right']].set_visible(False)
    axis.spines['bottom'].set_color('#6B7280')
    axis.set_title(title, color=TEXT_COLOR, fontsize=10, fontweight='bold', pad=10)


def create_figure(*, savepath: Path) -> None:
    """
    Write the three-panel temporal-valence retrieval schematic to ``savepath``.
    """
    profiles = build_retrieval_probabilities()
    y_max = max(
        profiles.temporal_only.max(),
        profiles.positive_query.max(),
        profiles.negative_query.max(),
    ) * 1.08
    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(11.5, 3.2),
        layout='constrained',
        gridspec_kw={'wspace': 0.18},
    )
    panels = (
        (
            profiles.temporal_only,
            np.full(profiles.element_ids.size, NEUTRAL_COLOR),
            'Temporal-only query\n$w_T=1.00$, $T=0.5$',
        ),
        (
            profiles.positive_query,
            np.where(profiles.positive_matches, POSITIVE_COLOR, MUTED_COLOR),
            'Positive-valence query\n$w_T=0.25$, $w_V=0.75$, $T=0.5$',
        ),
        (
            profiles.negative_query,
            np.where(profiles.negative_matches, NEGATIVE_COLOR, MUTED_COLOR),
            'Negative-valence query\n$w_T=0.25$, $w_V=0.75$, $T=0.5$',
        ),
    )

    for axis, (probabilities, colors, title) in zip(axes, panels, strict=True):
        plot_probability_panel(
            axis=axis,
            element_ids=profiles.element_ids,
            probabilities=probabilities,
            colors=colors,
            title=title,
            y_max=y_max,
        )

    figure.savefig(savepath, format='pdf', bbox_inches='tight')
    plt.close(figure)


def main() -> None:
    """
    Create the temporal-valence query-key retrieval illustration PDF.
    """
    create_figure(savepath=filepaths.VISUALS_DIR / 'temporal_valence_query_key_probabilities.pdf')


if __name__ == '__main__':
    main()
