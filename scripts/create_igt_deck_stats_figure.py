"""
Create a compact four-panel summary of the Iowa Gambling Task deck schedules.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['text.parse_math'] = False
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))

from utils import filepaths


DECK_NAMES = ('A', 'B', 'C', 'D')
WIN_COLOR = '#3D8C4A'
LOSS_COLOR = '#C74343'
NEUTRAL_COLOR = '#B8B8B8'
TEXT_COLOR = '#1F2933'
MAX_GROSS_PAYOFF = 12000
BAR_WIDTH = 0.32
AXIS_GAP = 0.06


@dataclass(frozen=True)
class DeckSummary:
    """
    Summary statistics for one 100-draw IGT deck.

    Args:
        name (str): Deck identifier.
        outcomes (np.ndarray): Ordered payoff schedule with shape ``[100]``.
        net_score (int): Total payoff across the schedule.
        win_frequency (float): Proportion of positive outcomes.
        payoff_counts (dict[int, int]): Outcome counts by payoff value.
    """

    name: str
    outcomes: np.ndarray
    net_score: int
    win_frequency: float
    payoff_counts: dict[int, int]


def load_schedule(schedule_path: Path) -> pd.DataFrame:
    """
    Load the headerless A/B/C/D IGT payoff schedule.

    Args:
        schedule_path (Path): Source CSV path.

    Returns:
        pd.DataFrame: Ordered payoff schedule with shape ``[100, 4]``.
    """
    return pd.read_csv(schedule_path, header=None, names=DECK_NAMES, dtype=int)


def summarize_decks(*, schedule: pd.DataFrame) -> list[DeckSummary]:
    """
    Calculate outcome-frequency and value summaries for all IGT decks.

    Args:
        schedule (pd.DataFrame): Ordered payoff schedule with shape ``[100, 4]``.

    Returns:
        list[DeckSummary]: One summary for each A/B/C/D deck.
    """
    summaries = []

    for deck_name in DECK_NAMES:
        outcomes = schedule[deck_name].to_numpy(dtype=int)
        payoff_counts = pd.Series(outcomes).value_counts(sort=False).to_dict()
        summaries.append(
            DeckSummary(
                name=deck_name,
                outcomes=outcomes,
                net_score=int(outcomes.sum()),
                win_frequency=float((outcomes > 0).mean()),
                payoff_counts=payoff_counts,
            )
        )

    return summaries


def format_payoff(payoff: int) -> str:
    """
    Format a signed payoff as a compact dollar amount.

    Args:
        payoff (int): Scalar deck payoff.

    Returns:
        str: Signed dollar amount.
    """
    if payoff > 0:
        return f'+${payoff:,}'
    if payoff < 0:
        return f'-${abs(payoff):,}'
    return '$0'


def plot_deck_panel(*, axis: Axes, summary: DeckSummary) -> None:
    """
    Render one deck's payoff schedule and summary statistics.

    Args:
        axis (Axes): Target panel axis.
        summary (DeckSummary): Ordered deck summary.
    """
    frequencies = [
        float((summary.outcomes > 0).mean()),
        float((summary.outcomes < 0).mean()),
    ]
    labels = ['Win', 'Loss']
    colors = [WIN_COLOR, LOSS_COLOR]
    gross_win_payoff = int(summary.outcomes[summary.outcomes > 0].sum())
    gross_loss_payoff = int(-summary.outcomes[summary.outcomes < 0].sum())
    payoff_heights = [-gross_win_payoff / MAX_GROSS_PAYOFF, -gross_loss_payoff / MAX_GROSS_PAYOFF]
    x_positions = [0, 1]

    axis.bar(x_positions, frequencies, bottom=AXIS_GAP, color=colors, width=BAR_WIDTH)
    axis.bar(x_positions, payoff_heights, bottom=-AXIS_GAP, color=colors, width=BAR_WIDTH)
    axis.axhline(0, color='#6B7280', linewidth=0.9)
    axis.set_xlim(-0.65, 1.65)
    axis.set_ylim(-1.3, 1.15)
    axis.set_xticks([])
    axis.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0], labels=['$12k', '$6k', '0', '50%', '100%'])
    axis.tick_params(axis='y', colors=TEXT_COLOR, labelsize=8, length=0)
    axis.grid(axis='y', color='#D8D8D8', linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[['top', 'right', 'bottom']].set_visible(False)
    axis.spines['left'].set_color('#A9A9A9')
    axis.set_title(f'DECK {summary.name}', color=TEXT_COLOR, fontsize=12, fontweight='bold', loc='left', pad=15)

    for position, frequency in zip(x_positions, frequencies, strict=True):
        axis.text(
            position,
            frequency + 0.06,
            f'{frequency:.0%}',
            color=TEXT_COLOR,
            fontsize=8.5,
            ha='center',
            va='center',
        )

    axis.text(
        1.0,
        1.05,
        f'NET {format_payoff(summary.net_score)}',
        color=WIN_COLOR if summary.net_score > 0 else LOSS_COLOR,
        fontsize=10,
        fontweight='bold',
        transform=axis.transAxes,
        ha='right',
        va='bottom',
    )
    axis.text(
        -0.20,
        0.78,
        'Frequency (%)',
        color=TEXT_COLOR,
        fontsize=8,
        transform=axis.transAxes,
        rotation=90,
        ha='center',
        va='center',
    )
    axis.text(
        -0.20,
        0.22,
        'Amount ($)',
        color=TEXT_COLOR,
        fontsize=8,
        transform=axis.transAxes,
        rotation=90,
        ha='center',
        va='center',
    )

    for position, label, payoff_height in zip(
        x_positions,
        [f'Win\n{format_payoff(gross_win_payoff)}', f'Loss\n{format_payoff(-gross_loss_payoff)}'],
        payoff_heights,
        strict=True,
    ):
        axis.text(
            position,
            min(payoff_height - 0.12, -1.10),
            label,
            color=TEXT_COLOR,
            fontsize=8.5,
            ha='center',
            va='top',
        )


def create_figure(*, schedule_path: Path, savepath: Path) -> None:
    """
    Write the 2x2 IGT deck statistics PDF.

    Args:
        schedule_path (Path): Source headerless IGT schedule CSV.
        savepath (Path): Destination PDF path.
    """
    schedule = load_schedule(schedule_path)
    summaries = summarize_decks(schedule=schedule)
    figure, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(11.5, 3.25),
        layout='constrained',
        gridspec_kw={'wspace': 0.35},
    )

    for axis, summary in zip(axes.flat, summaries, strict=True):
        plot_deck_panel(axis=axis, summary=summary)

    figure.savefig(savepath, format='pdf', bbox_inches='tight')
    plt.close(figure)


def main() -> None:
    """
    Create the deck-summary PDF from the repository IGT schedule.
    """
    create_figure(
        schedule_path=filepaths.ROOT / 'data' / 'IGT-task-config.csv',
        savepath=filepaths.VISUALS_DIR / 'igt_deck_stats.pdf',
    )


if __name__ == '__main__':
    main()
