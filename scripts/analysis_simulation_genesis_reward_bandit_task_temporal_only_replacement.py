"""
Analyze temporal-only GENESIS reward-bandit simulations sampled with replacement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path.cwd()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr

from scripts import analysis_simulation_genesis_reward_bandit_task_temporal_only as budget_analysis
from utils import filepaths
from utils.simulation_config import get_replacement_output_dir, load_simulation_config


HIGH_EFFECT_ENVIRONMENT_ID = 46


def format_temperature(temperature: float) -> str:
    """
    Format a softmax temperature for tables and figure labels.

    Args:
        temperature (float): Softmax temperature.

    Returns:
        str: Display label.
    """
    return f'{temperature:.2g}' if temperature < 1 else f'{temperature:.1f}'


def resolve_shared_reward_ylim(
    *,
    example_curves: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[float, float]:
    """
    Resolve a common cumulative-reward range for selected example environments.

    Args:
        example_curves (Sequence[Sequence[Mapping[str, Any]]]): Reward curves for each selected environment.

    Returns:
        tuple[float, float]: Shared lower and upper cumulative-reward limits.
    """
    lower_bounds = []
    upper_bounds = []

    for curves in example_curves:
        n_nans = max(np.isnan(curve['value_mean']).sum() for curve in curves)
        for curve in curves:
            reward_mean = curve['reward_mean'][n_nans:]
            reward_sem = curve['reward_sem'][n_nans:]
            lower_bounds.append(np.nanmin(reward_mean - reward_sem / 10))
            upper_bounds.append(np.nanmax(reward_mean + reward_sem / 10))

    lower = min(lower_bounds)
    upper = max(upper_bounds)
    padding = (upper - lower) * 0.05

    return lower - padding, upper + padding


def select_replacement_example_environment(
    *,
    label: str,
    df_env_scores: pd.DataFrame,
    score_window: Sequence[float],
    seed: int,
) -> int:
    """
    Select a replacement-condition example environment.

    Args:
        label (str): Example label.
        df_env_scores (pd.DataFrame): One score row per environment.
        score_window (Sequence[float]): Inclusive score-selection window.
        seed (int): Random seed for non-fixed examples.

    Returns:
        int: Selected environment identifier.
    """
    if label == 'high':
        return HIGH_EFFECT_ENVIRONMENT_ID

    return budget_analysis.select_example_environment(
        df_env_scores=df_env_scores,
        score_window=score_window,
        seed=seed,
    )


def select_high_temperature_effect_environments(
    *,
    df_reward_analysis: pd.DataFrame,
    temperature_order: Sequence[str],
    score_window: Sequence[float],
    n_examples: int,
) -> list[int]:
    """
    Select high-score environments with the clearest maximum-temperature reward advantage.

    Args:
        df_reward_analysis (pd.DataFrame): Environment-level reward table indexed by temperature label.
        temperature_order (Sequence[str]): Ordered softmax-temperature labels.
        score_window (Sequence[float]): High-score selection window whose lower bound defines the retained tail.
        n_examples (int): Number of environments to retain.

    Returns:
        list[int]: Ordered environment identifiers.
    """
    df_pivot = df_reward_analysis.pivot(index='env_id', columns='sampling_budget', values='total_reward')
    df_env_scores = df_reward_analysis[['env_id', 'env_score']].drop_duplicates('env_id').set_index('env_id')
    low, _ = score_window
    rows = []

    for env_id in df_env_scores.query('@low <= env_score').index:
        rewards = df_pivot.loc[env_id, temperature_order].to_numpy(dtype=float)
        if rewards[-1] <= rewards[:-1].max():
            continue

        rows.append({
            'env_id': env_id,
            'n_decreases': int(np.sum(np.diff(rewards) <= 0)),
            'endpoint_advantage': rewards[-1] - rewards[0],
        })

    df_candidates = pd.DataFrame(rows)
    return (
        df_candidates
        .sort_values(['n_decreases', 'endpoint_advantage', 'env_id'], ascending=[True, False, True])
        .head(n_examples)
        .env_id
        .astype(int)
        .tolist()
    )


def resolve_temperature_groups(*, temperature_order: Sequence[str]) -> tuple[list[str], list[str]]:
    """
    Resolve low and high endpoint groups from the configured temperature order.

    Args:
        temperature_order (Sequence[str]): Ordered softmax-temperature labels.

    Returns:
        tuple[list[str], list[str]]: Low- and high-temperature endpoint groups.
    """
    n_levels = min(2, len(temperature_order) // 2)

    return list(temperature_order[:n_levels]), list(temperature_order[-n_levels:])


def compute_temperature_contrast(
    *,
    df_reward_analysis: pd.DataFrame,
    temperature_order: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute the high-minus-low temperature reward advantage across environments.

    Args:
        df_reward_analysis (pd.DataFrame): Environment-level reward table indexed by temperature label.
        temperature_order (Sequence[str]): Ordered softmax-temperature labels from the simulation configuration.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Contrast table, fitted statistics, and reward pivot table.
    """
    df_pivot = df_reward_analysis.pivot(index='env_id', columns='sampling_budget', values='total_reward')
    df_contrast = df_reward_analysis[['env_id', 'env_score']].drop_duplicates('env_id').set_index('env_id')
    low_temperatures, high_temperatures = resolve_temperature_groups(temperature_order=temperature_order)
    df_contrast['low_temperature_reward'] = df_pivot[low_temperatures].mean(axis=1)
    df_contrast['high_temperature_reward'] = df_pivot[high_temperatures].mean(axis=1)
    df_contrast['high_minus_low_temperature_reward'] = (
        df_contrast['high_temperature_reward'] - df_contrast['low_temperature_reward']
    )
    df_contrast = df_contrast.reset_index()

    x = df_contrast.env_score.to_numpy(dtype=float)
    y = df_contrast.high_minus_low_temperature_reward.to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    fit = linregress(x, y)
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    pearson_ci_low, pearson_ci_high = budget_analysis.bootstrap_corr_ci(x=x, y=y)
    df_stats = pd.DataFrame([{
        'contrast': (
            f"mean(T={','.join(high_temperatures)}) - mean(T={','.join(low_temperatures)})"
        ),
        'n_env': len(x),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'pearson_ci_low': pearson_ci_low,
        'pearson_ci_high': pearson_ci_high,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'slope': fit.slope,
        'intercept': fit.intercept,
        'slope_p': fit.pvalue,
        'r_squared': pearson_r ** 2,
    }])

    return df_contrast, df_stats, df_pivot


def save_temperature_correlation_plot(
    *,
    savepath: Path,
    df_reward_corr_stats: pd.DataFrame,
    temperature_colors: dict[str, str],
    save_figures: bool,
) -> None:
    """
    Save the reward-direction-score correlation by softmax temperature.

    Args:
        savepath (Path): Destination PDF path.
        df_reward_corr_stats (pd.DataFrame): One correlation summary per temperature.
        temperature_colors (dict[str, str]): Colors keyed by temperature label.
        save_figures (bool): Whether to save the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 6.0))
    budget_analysis.plot_reward_correlation_panel(
        ax=ax,
        df_reward_corr_stats=df_reward_corr_stats,
        budget_colors=temperature_colors,
    )
    ax.set_xlabel('Softmax temperature', fontsize=12)
    fig.tight_layout()
    if save_figures:
        fig.savefig(savepath, dpi=800, bbox_inches='tight')


def save_temperature_contrast_plot(
    *,
    savepath: Path,
    df_contrast: pd.DataFrame,
    df_stats: pd.DataFrame,
    save_figures: bool,
) -> None:
    """
    Save the high-minus-low temperature advantage by environment score.

    Args:
        savepath (Path): Destination PDF path.
        df_contrast (pd.DataFrame): Environment-level contrast table.
        df_stats (pd.DataFrame): Fitted contrast statistics.
        save_figures (bool): Whether to save the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    ax.scatter(df_contrast.env_score, df_contrast.high_minus_low_temperature_reward, color='black', alpha=0.7, s=35)
    xline = np.linspace(df_contrast.env_score.min(), df_contrast.env_score.max(), 200)
    ax.plot(xline, df_stats.intercept.iloc[0] + df_stats.slope.iloc[0] * xline, color='#EA2264', linewidth=2.5)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Early-to-late direction score', fontsize=12)
    ax.set_ylabel('Reward advantage: high T - low T', fontsize=12)
    text = (
        f'r = {df_stats.pearson_r.iloc[0]:.2f}\n'
        f'95% CI [{df_stats.pearson_ci_low.iloc[0]:.2f}, {df_stats.pearson_ci_high.iloc[0]:.2f}]\n'
        f'p = {df_stats.pearson_p.iloc[0]:.1e}'
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=10,
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8},
    )
    budget_analysis.style_axis(ax)
    fig.tight_layout()
    if save_figures:
        fig.savefig(savepath, dpi=800, bbox_inches='tight')


def save_temperature_class_plot(
    *,
    savepath: Path,
    df_class_stats: pd.DataFrame,
    temperature_order: Sequence[str],
    class_labels: Sequence[str],
    class_colors: dict[str, str],
    save_figures: bool,
) -> None:
    """
    Save reward profiles by environment-score tertile and softmax temperature.

    Args:
        savepath (Path): Destination PDF path.
        df_class_stats (pd.DataFrame): Reward statistics by class and temperature.
        temperature_order (Sequence[str]): Ordered temperature labels.
        class_labels (Sequence[str]): Ordered environment-class display labels.
        class_colors (dict[str, str]): Colors keyed by environment class.
        save_figures (bool): Whether to save the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
    budget_analysis.plot_env_class_reward_panel(
        ax=ax,
        df_class_stats=df_class_stats,
        budget_order=temperature_order,
        class_labels=class_labels,
        class_colors=class_colors,
    )
    ax.set_xlabel('Softmax temperature', fontsize=12)
    fig.tight_layout()
    if save_figures:
        fig.savefig(savepath, dpi=800, bbox_inches='tight')


def run_analysis(args: argparse.Namespace) -> None:
    """
    Run the replacement=True temporal-only temperature analysis.

    Args:
        args (argparse.Namespace): Parsed analysis arguments.
    """
    config = budget_analysis.load_analysis_config(args.config)
    save_figures = bool(config['save_figures']) and not args.no_save_figures
    simulation_config_path = Path(config['simulation_config_path'])
    if not simulation_config_path.is_absolute():
        simulation_config_path = filepaths.ROOT / simulation_config_path
    simulation_config = load_simulation_config(simulation_config_path)
    output_dir = args.output_dir or get_replacement_output_dir(
        root_dir=filepaths.OUTPUT_DIR,
        output_name=config['output_dir_name'],
        sample_replacement=True,
    )
    visual_dir = output_dir / 'visuals'
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = args.input_dir or get_replacement_output_dir(
        root_dir=filepaths.EXPERIMENTS_DIR,
        output_name=simulation_config['experiment_name'],
        sample_replacement=True,
    )

    df_experiments, load_experiment = budget_analysis.discover_experiments(experiment_dir)
    df_full = budget_analysis.build_full_table(
        df_experiments=df_experiments,
        load_experiment=load_experiment,
        savepath=output_dir / 'full_table.csv',
        rebuild_cache=args.rebuild_cache,
    )
    df_full['sampling_budget'] = df_full.sampling_temperature.map(format_temperature)
    df_env_score = budget_analysis.build_env_score_table(
        df_full=df_full,
        savepath=output_dir / 'env_score-vs-policy-table.csv',
        rebuild_cache=args.rebuild_cache,
    )
    temperature_order = [format_temperature(temperature) for temperature in simulation_config['sampling_temperature']]
    temperature_colors = budget_analysis.resolve_budget_colors(
        budget_order=temperature_order,
        configured_colors={},
    )
    df_reward_analysis = budget_analysis.build_reward_analysis_table(df_env_score)
    df_env_scores = df_reward_analysis[['env_id', 'env_score']].drop_duplicates('env_id')

    example_curves = {}
    for label, example_config in config['example_environments'].items():
        environment_id = select_replacement_example_environment(
            label=label,
            df_env_scores=df_env_scores,
            score_window=example_config['score_window'],
            seed=example_config['seed'],
        )
        example_curves[label] = budget_analysis.compute_example_environment_curves(
            df_full=df_full,
            env_id=environment_id,
            budget_order=temperature_order,
        )
    reward_ylim = resolve_shared_reward_ylim(
        example_curves=[curves[2] for curves in example_curves.values()],
    )

    for label, curves in example_curves.items():
        budget_analysis.save_example_environment_plot(
            savepath=visual_dir / f'sampling-temperature_rewards_env-score-{label}.pdf',
            p_win_left=curves[0],
            trial_ids=curves[1],
            curves=curves[2],
            budget_order=temperature_order,
            budget_colors=temperature_colors,
            save_figures=save_figures,
            reward_ylim=reward_ylim,
        )

    high_effect_environment_ids = select_high_temperature_effect_environments(
        df_reward_analysis=df_reward_analysis,
        temperature_order=temperature_order,
        score_window=config['example_environments']['high']['score_window'],
        n_examples=5,
    )
    for environment_id in high_effect_environment_ids:
        curves = budget_analysis.compute_example_environment_curves(
            df_full=df_full,
            env_id=environment_id,
            budget_order=temperature_order,
        )
        budget_analysis.save_example_environment_plot(
            savepath=visual_dir / f'sampling-temperature_rewards_env-score-high-env-id-{environment_id}.pdf',
            p_win_left=curves[0],
            trial_ids=curves[1],
            curves=curves[2],
            budget_order=temperature_order,
            budget_colors=temperature_colors,
            save_figures=save_figures,
        )

    df_corr = budget_analysis.compute_reward_correlation_stats(
        df_reward_analysis=df_reward_analysis,
        budget_order=temperature_order,
    )
    df_corr.to_csv(output_dir / 'reward_envscore_stats_by_temperature.csv', index=False)
    save_temperature_correlation_plot(
        savepath=visual_dir / 'reward_envscore_correlation_by_temperature.pdf',
        df_reward_corr_stats=df_corr,
        temperature_colors=temperature_colors,
        save_figures=save_figures,
    )

    df_contrast, df_contrast_stats, df_pivot = compute_temperature_contrast(
        df_reward_analysis=df_reward_analysis,
        temperature_order=temperature_order,
    )
    df_contrast.to_csv(output_dir / 'high_vs_low_temperature_advantage_by_envscore.csv', index=False)
    df_contrast_stats.to_csv(output_dir / 'high_vs_low_temperature_advantage_stats.csv', index=False)
    save_temperature_contrast_plot(
        savepath=visual_dir / 'high_vs_low_temperature_advantage_by_envscore.pdf',
        df_contrast=df_contrast,
        df_stats=df_contrast_stats,
        save_figures=save_figures,
    )

    class_names, class_labels, class_colors = budget_analysis.resolve_class_plot_config(config)
    df_env_bins = budget_analysis.build_env_score_classes(
        df_reward_analysis=df_reward_analysis,
        class_names=class_names,
        class_labels=config['environment_classes']['labels'],
    )
    df_class_stats, _, _ = budget_analysis.compute_env_class_stats(
        df_reward_analysis=df_reward_analysis,
        df_env_bins=df_env_bins,
        df_pivot=df_pivot,
        budget_order=temperature_order,
        low_budgets=resolve_temperature_groups(temperature_order=temperature_order)[0],
        large_budgets=resolve_temperature_groups(temperature_order=temperature_order)[1],
    )
    df_class_stats.to_csv(output_dir / 'reward_by_envscore_tertile_temperature_stats.csv', index=False)
    save_temperature_class_plot(
        savepath=visual_dir / 'reward_by_envscore_tertile_and_temperature.pdf',
        df_class_stats=df_class_stats,
        temperature_order=temperature_order,
        class_labels=class_labels,
        class_colors=class_colors,
        save_figures=save_figures,
    )
    plt.close('all')


def main(argv: list[str] | None = None) -> None:
    """
    Run the replacement analysis from command-line arguments.

    Args:
        argv (list[str] | None): Optional command-line argument list.
    """
    args = budget_analysis.parse_args(argv)
    run_analysis(args)


if __name__ == '__main__':
    main()
