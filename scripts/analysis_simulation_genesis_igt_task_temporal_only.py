"""
Analyze temporal-only GENESIS Iowa Gambling Task simulations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path.cwd()))

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.axes import Axes

from utils import filepaths


DEFAULT_ANALYSIS_CONFIG_PATH = filepaths.ROOT / 'configs' / 'analysis_simulation_genesis_igt_task_temporal_only.yaml'
DECK_NAMES = ('A', 'B', 'C', 'D')
PROBABILITY_COLUMNS = {'A': 'p_a', 'B': 'p_b', 'C': 'p_c', 'D': 'p_d'}
CHOICE_COLUMNS = {'A': 'i_a', 'B': 'i_b', 'C': 'i_c', 'D': 'i_d'}
DEFAULT_DECK_COLORS = {'A': '#D55E00', 'B': '#CC79A7', 'C': '#0072B2', 'D': '#009E73'}
DEFAULT_TEMPERATURE_COLORS = ('#440154', '#3B528B', '#21918C', '#5EC962', '#FDE725')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse temporal-only IGT analysis arguments.

    Args:
        argv (list[str] | None): Optional command-line arguments.

    Returns:
        argparse.Namespace: Parsed analysis arguments.
    """
    parser = argparse.ArgumentParser(description='Analyze temporal-only GENESIS IGT simulations.')
    parser.add_argument('--config', type=Path, default=DEFAULT_ANALYSIS_CONFIG_PATH)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-save-figures', action='store_true')
    return parser.parse_args(argv)


def load_analysis_config(config_path: Path) -> dict[str, Any]:
    """
    Load analysis controls from YAML.

    Args:
        config_path (Path): Analysis configuration file.

    Returns:
        dict[str, Any]: Parsed analysis controls.
    """
    return yaml.safe_load(config_path.read_text(encoding='utf-8'))


def config_without_retrieval_temperature(config: dict[str, Any]) -> dict[str, Any]:
    """
    Remove the per-cell grid value from a resolved simulation configuration.

    Args:
        config (dict[str, Any]): Resolved cell configuration.

    Returns:
        dict[str, Any]: Fixed configuration fields.
    """
    return {key: value for key, value in config.items() if key != 'retrieval_temperature'}


def validate_cell(*, cell: dict[str, Any], reference_cell: dict[str, Any] | None) -> None:
    """
    Validate one loaded IGT grid cell against the shared experiment contract.

    Args:
        cell (dict[str, Any]): Loaded simulation payload.
        reference_cell (dict[str, Any] | None): First validated payload.

    Raises:
        ValueError: If required fields or fixed simulation settings are incompatible.
    """
    required_cell_keys = {
        'config',
        'retrieval_temperature',
        'source_schedule_path',
        'reward_scaling_factor',
        'temporal_seed',
        'agent_runs',
    }
    missing_cell_keys = required_cell_keys.difference(cell)
    if missing_cell_keys:
        raise ValueError(f'Missing simulation payload fields: {sorted(missing_cell_keys)}.')

    config = cell['config']
    required_config_keys = {'task', 'temporal', 'episodic_sampling', 'choice_policy', 'retrieval_temperature'}
    missing_config_keys = required_config_keys.difference(config)
    if missing_config_keys:
        raise ValueError(f'Missing resolved configuration fields: {sorted(missing_config_keys)}.')

    if cell['retrieval_temperature'] != config['retrieval_temperature']:
        raise ValueError('Payload retrieval temperature does not match the resolved configuration.')

    agent_ids = []
    expected_n_trials = config['task']['n_trials']
    required_trial_keys = {
        'trial_id',
        'chosen_deck',
        'deck_draw_index',
        'reward',
        'cumulative_reward',
        'action_values',
        'p_a',
        'p_b',
        'p_c',
        'p_d',
        'choice_temperature',
        'retrieval_temperature',
        'previous_experience_counts',
        'retrieved_episode_indices',
    }
    for agent_run in cell['agent_runs']:
        if not {'agent_id', 'simulation_log'}.issubset(agent_run):
            raise ValueError('Each agent run must contain agent_id and simulation_log.')
        agent_ids.append(agent_run['agent_id'])
        simulation_log = agent_run['simulation_log']
        if len(simulation_log) != expected_n_trials:
            raise ValueError('Agent trial count does not match the resolved configuration.')
        for expected_trial_id, trial in enumerate(simulation_log):
            missing_trial_keys = required_trial_keys.difference(trial)
            if missing_trial_keys:
                raise ValueError(f'Missing trial fields: {sorted(missing_trial_keys)}.')
            if trial['trial_id'] != expected_trial_id:
                raise ValueError('Trial IDs must be contiguous and zero-based.')
            if trial['chosen_deck'] not in DECK_NAMES:
                raise ValueError(f"Unknown IGT deck: {trial['chosen_deck']}.")
            if trial['retrieval_temperature'] != cell['retrieval_temperature']:
                raise ValueError('Trial retrieval temperature does not match its cell.')
            if set(trial['action_values']) != set(DECK_NAMES):
                raise ValueError('Action values must contain all four decks.')
            if set(trial['previous_experience_counts']) != set(DECK_NAMES):
                raise ValueError('Experience counts must contain all four decks.')
            if set(trial['retrieved_episode_indices']) != set(DECK_NAMES):
                raise ValueError('Retrieved episode indices must contain all four decks.')

    if len(agent_ids) != config['n_agents']:
        raise ValueError('Agent-run count does not match the resolved configuration.')

    if reference_cell is None:
        return

    if config_without_retrieval_temperature(config) != config_without_retrieval_temperature(reference_cell['config']):
        raise ValueError('Simulation cells have incompatible fixed simulation settings.')

    for key in ('source_schedule_path', 'reward_scaling_factor', 'temporal_seed'):
        if cell[key] != reference_cell[key]:
            raise ValueError(f'Simulation cells have incompatible {key}.')

    reference_agent_ids = [agent_run['agent_id'] for agent_run in reference_cell['agent_runs']]
    if agent_ids != reference_agent_ids:
        raise ValueError('Simulation cells do not preserve the shared agent IDs and order.')


def load_simulation_cells(input_dir: Path) -> list[dict[str, Any]]:
    """
    Load, validate, and temperature-sort temporal-only IGT grid cells.

    Args:
        input_dir (Path): Directory containing saved `.npy` simulation cells.

    Returns:
        list[dict[str, Any]]: Compatible simulation payloads in temperature order.

    Raises:
        ValueError: If no cells are found or cells are incompatible.
    """
    filepaths_found = list(input_dir.glob('*_retrieval_temperature-*.npy'))
    if not filepaths_found:
        raise ValueError(f'No temporal-only IGT simulation files found in {input_dir}.')

    cells = []
    for simulation_path in filepaths_found:
        cell = np.load(simulation_path, allow_pickle=True).item()
        cell['filepath'] = simulation_path
        cells.append(cell)

    cells.sort(key=lambda cell: float(cell['retrieval_temperature']))
    reference_cell = None
    for cell in cells:
        validate_cell(cell=cell, reference_cell=reference_cell)
        if reference_cell is None:
            reference_cell = cell

    temperatures = [cell['retrieval_temperature'] for cell in cells]
    if len(temperatures) != len(set(temperatures)):
        raise ValueError('Each retrieval temperature must have exactly one simulation cell.')

    return cells


def build_trial_table(cells: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten saved agent histories into a trial-level analysis table.

    Args:
        cells (Sequence[dict[str, Any]]): Validated simulation payloads.

    Returns:
        pd.DataFrame: One row per retrieval temperature, agent, and trial.
    """
    rows = []
    for cell in cells:
        retrieval_temperature = float(cell['retrieval_temperature'])
        for agent_run in cell['agent_runs']:
            for trial in agent_run['simulation_log']:
                row = {
                    'retrieval_temperature': retrieval_temperature,
                    'agent_id': agent_run['agent_id'],
                    'trial_id': trial['trial_id'],
                    'trial_number': trial['trial_id'] + 1,
                    'chosen_deck': trial['chosen_deck'],
                    'deck_draw_index': trial['deck_draw_index'],
                    'reward': trial['reward'],
                    'cumulative_reward': trial['cumulative_reward'],
                    'choice_temperature': trial['choice_temperature'],
                    'retrieved_episode_indices': trial['retrieved_episode_indices'],
                }
                row.update({f'q_{deck.lower()}': trial['action_values'][deck] for deck in DECK_NAMES})
                row.update({column: trial[column] for column in PROBABILITY_COLUMNS.values()})
                row.update({
                    f'previous_experience_count_{deck.lower()}': trial['previous_experience_counts'][deck]
                    for deck in DECK_NAMES
                })
                row.update({
                    CHOICE_COLUMNS[deck]: int(trial['chosen_deck'] == deck)
                    for deck in DECK_NAMES
                })
                row['i_adv'] = row['i_c'] + row['i_d']
                row['i_disadv'] = row['i_a'] + row['i_b']
                row['i_rare'] = row['i_b'] + row['i_d']
                row['i_frequent'] = row['i_a'] + row['i_c']
                row['block_id'] = trial['trial_id'] // 20 + 1
                row['block_label'] = f"Block {row['block_id']}"
                rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ['retrieval_temperature', 'agent_id', 'trial_id'],
    ).reset_index(drop=True)


def bootstrap_mean_ci(
    *,
    values: Sequence[float],
    n_boot: int,
    random_state: int,
) -> tuple[float, float, float]:
    """
    Estimate an agent-level mean and percentile bootstrap confidence interval.

    Args:
        values (Sequence[float]): Independent agent-level observations.
        n_boot (int): Number of bootstrap resamples.
        random_state (int): Deterministic bootstrap seed.

    Returns:
        tuple[float, float, float]: Mean, lower 95% CI, and upper 95% CI.
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(random_state)
    sample_indices = rng.integers(0, len(array), size=(n_boot, len(array)))
    bootstrap_means = array[sample_indices].mean(axis=1)
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(array.mean()), float(ci_low), float(ci_high)


def bootstrap_mean_se(*, values: Sequence[float], n_boot: int, random_state: int) -> float:
    """Compute the bootstrap standard error of an agent-level mean."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(array), size=(n_boot, len(array)))
    return float(array[indices].mean(axis=1).std(ddof=1))


def centered_rolling_mean(*, values: pd.Series, window: int) -> pd.Series:
    """Smooth one plotted trajectory with a centered rolling mean."""
    return values.rolling(window=window, center=True, min_periods=1).mean()


def add_igt_blocks(*, table: pd.DataFrame, block_size: int) -> pd.DataFrame:
    """Assign consecutive IGT blocks to a trial table."""
    result = table.copy()
    result['block_id'] = result['trial_id'] // block_size + 1
    result['block_label'] = result['block_id'].map(lambda block_id: f'Block {block_id}')
    return result


def summarize_agent_metrics(
    *,
    table: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
    n_boot: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Summarize agent-level metrics with deterministic bootstrap confidence intervals.

    Args:
        table (pd.DataFrame): Table containing one row per independent agent unit.
        group_columns (list[str]): Grouping columns for estimates.
        metric_columns (list[str]): Agent-level metric columns to summarize.
        n_boot (int): Number of bootstrap resamples.
        random_state (int): Base random seed.

    Returns:
        pd.DataFrame: Mean and 95% CI columns for every grouped metric.
    """
    rows = []
    grouped = table.groupby(group_columns, sort=True, dropna=False)
    for group_index, (group_key, group) in enumerate(grouped):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_columns, group_key))
        for metric_index, metric in enumerate(metric_columns):
            estimate, ci_low, ci_high = bootstrap_mean_ci(
                values=group[metric],
                n_boot=n_boot,
                random_state=random_state + group_index * len(metric_columns) + metric_index,
            )
            row[f'{metric}_mean'] = estimate
            row[f'{metric}_ci_low'] = ci_low
            row[f'{metric}_ci_high'] = ci_high
            row[f'{metric}_bootstrap_se'] = bootstrap_mean_se(
                values=group[metric], n_boot=n_boot, random_state=random_state + group_index * len(metric_columns) + metric_index,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_agent_tables(
    *,
    trial_table: pd.DataFrame,
    n_anneal_trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute agent-level block and whole-task behavioral summaries.

    Args:
        trial_table (pd.DataFrame): Flattened IGT trial records.
        n_anneal_trials (int): Number of initial trials excluded from lag summaries.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Agent-block and total-agent tables.
    """
    count_columns = list(CHOICE_COLUMNS.values())
    grouped_block = (
        trial_table.groupby(['retrieval_temperature', 'agent_id', 'block_id', 'block_label'], sort=True)[count_columns]
        .sum()
        .reset_index()
    )
    grouped_block = grouped_block.rename(columns={
        'i_a': 'a_choices',
        'i_b': 'b_choices',
        'i_c': 'c_choices',
        'i_d': 'd_choices',
    })
    grouped_block['net_score'] = (
        grouped_block['c_choices']
        + grouped_block['d_choices']
        - grouped_block['a_choices']
        - grouped_block['b_choices']
    )
    grouped_block['frequency_score'] = (
        grouped_block['b_choices']
        + grouped_block['d_choices']
        - grouped_block['a_choices']
        - grouped_block['c_choices']
    )

    total_counts = (
        trial_table.groupby(['retrieval_temperature', 'agent_id'], sort=True)[count_columns]
        .sum()
        .reset_index()
        .rename(columns={
            'i_a': 'total_a_choices',
            'i_b': 'total_b_choices',
            'i_c': 'total_c_choices',
            'i_d': 'total_d_choices',
        })
    )
    final_rewards = (
        trial_table.sort_values('trial_id')
        .groupby(['retrieval_temperature', 'agent_id'], sort=True)
        .agg(total_reward=('reward', 'sum'), final_cumulative_reward=('cumulative_reward', 'last'))
        .reset_index()
    )
    lag_rows = []
    eligible_trials = trial_table[trial_table['trial_id'] >= n_anneal_trials]
    for group_key, group in eligible_trials.groupby(['retrieval_temperature', 'agent_id'], sort=True):
        lags = []
        for trial in group.itertuples(index=False):
            for deck in DECK_NAMES:
                lags.extend(trial.trial_id - episode_index for episode_index in trial.retrieved_episode_indices[deck])
        lag_rows.append({
            'retrieval_temperature': group_key[0],
            'agent_id': group_key[1],
            'mean_sampled_episodic_lag': np.mean(lags) if lags else np.nan,
        })
    lag_table = pd.DataFrame(lag_rows)

    agent_table = total_counts.merge(final_rewards, on=['retrieval_temperature', 'agent_id'])
    agent_table = agent_table.merge(lag_table, on=['retrieval_temperature', 'agent_id'])
    agent_table['total_net_score'] = (
        agent_table['total_c_choices']
        + agent_table['total_d_choices']
        - agent_table['total_a_choices']
        - agent_table['total_b_choices']
    )
    agent_table['total_frequency_score'] = (
        agent_table['total_b_choices']
        + agent_table['total_d_choices']
        - agent_table['total_a_choices']
        - agent_table['total_c_choices']
    )
    agent_table['b_minus_a'] = agent_table['total_b_choices'] - agent_table['total_a_choices']
    agent_table['d_minus_c'] = agent_table['total_d_choices'] - agent_table['total_c_choices']
    return grouped_block, agent_table


def build_trial_probability_summary(
    *,
    trial_table: pd.DataFrame,
    n_boot: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Summarize stored deck-choice probabilities across agents.

    Args:
        trial_table (pd.DataFrame): Flattened IGT trial records.
        n_boot (int): Number of bootstrap resamples.
        random_state (int): Base random seed.

    Returns:
        pd.DataFrame: Temperature, trial, and deck probability summaries.
    """
    rows = []
    group_number = 0
    for retrieval_temperature, temperature_group in trial_table.groupby('retrieval_temperature', sort=True):
        for trial_number, trial_group in temperature_group.groupby('trial_number', sort=True):
            for deck in DECK_NAMES:
                mean, ci_low, ci_high = bootstrap_mean_ci(
                    values=trial_group[PROBABILITY_COLUMNS[deck]],
                    n_boot=n_boot,
                    random_state=random_state + group_number,
                )
                standard_error = bootstrap_mean_se(
                    values=trial_group[PROBABILITY_COLUMNS[deck]],
                    n_boot=n_boot,
                    random_state=random_state + group_number,
                )
                rows.append({
                    'retrieval_temperature': retrieval_temperature,
                    'trial_number': trial_number,
                    'deck': deck,
                    'mean_choice_probability': mean,
                    'choice_probability_ci_low': ci_low,
                    'choice_probability_ci_high': ci_high,
                    'choice_probability_bootstrap_se': standard_error,
                    'observed_choice_proportion': trial_group[CHOICE_COLUMNS[deck]].mean(),
                })
                group_number += 1
    return pd.DataFrame(rows)


def build_block_behavior_summary(
    *,
    agent_block_table: pd.DataFrame,
    n_boot: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Summarize block-level IGT behavioral measures across agents.

    Args:
        agent_block_table (pd.DataFrame): One row per temperature, agent, and block.
        n_boot (int): Number of bootstrap resamples.
        random_state (int): Base random seed.

    Returns:
        pd.DataFrame: Temperature-block psychology summaries.
    """
    return summarize_agent_metrics(
        table=agent_block_table,
        group_columns=['retrieval_temperature', 'block_id', 'block_label'],
        metric_columns=['net_score', 'frequency_score', 'a_choices', 'b_choices', 'c_choices', 'd_choices'],
        n_boot=n_boot,
        random_state=random_state,
    )


def style_axis(axis: Axes) -> None:
    """
    Apply the shared manuscript plot style.

    Args:
        axis (Axes): Matplotlib axis to style.
    """
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.tick_params(axis='both', length=0)


def resolve_temperature_colors(
    *,
    temperatures: Sequence[float],
    configured_colors: dict[str, str] | None,
) -> dict[float, str]:
    """
    Resolve an ordered color mapping for retrieval temperatures.

    Args:
        temperatures (Sequence[float]): Sorted retrieval temperatures.
        configured_colors (dict[str, str] | None): Optional YAML palette.

    Returns:
        dict[float, str]: Temperature color mapping.
    """
    return {
        temperature: (configured_colors or {}).get(str(temperature), DEFAULT_TEMPERATURE_COLORS[index % 5])
        for index, temperature in enumerate(temperatures)
    }


def temperature_positions(temperatures: Sequence[float]) -> dict[float, int]:
    """
    Map ordered retrieval temperatures to evenly spaced plot coordinates.

    Args:
        temperatures (Sequence[float]): Retrieval temperatures in display order.

    Returns:
        dict[float, int]: Evenly spaced position for every temperature.
    """
    return {temperature: index for index, temperature in enumerate(temperatures)}


def save_figure(*, figure: plt.Figure, savepath: Path) -> None:
    """
    Export one manuscript figure as PDF and SVG.

    Args:
        figure (plt.Figure): Figure to export.
        savepath (Path): Base path without suffix.
    """
    figure.savefig(savepath.with_suffix('.pdf'), bbox_inches='tight')
    figure.savefig(savepath.with_suffix('.svg'), bbox_inches='tight')
    plt.close(figure)


def plot_figure_1(
    *,
    trial_summary: pd.DataFrame,
    deck_colors: dict[str, str],
    temperature_colors: dict[float, str],
    visual_dir: Path,
) -> None:
    """
    Plot unsmoothed stored deck-choice probability trajectories.

    Args:
        trial_summary (pd.DataFrame): Agent-aggregated probability table.
        deck_colors (dict[str, str]): Stable deck color mapping.
        visual_dir (Path): Figure destination directory.
    """
    temperatures = sorted(trial_summary['retrieval_temperature'].unique())
    figure, axes = plt.subplots(2, 3, figsize=(12.3, 6.2), squeeze=False)
    for index, deck in enumerate(DECK_NAMES):
        axis = axes.flat[index]
        for temperature in temperatures:
            data = trial_summary[(trial_summary['retrieval_temperature'] == temperature) & (trial_summary['deck'] == deck)]
            mean = centered_rolling_mean(values=data['mean_choice_probability'], window=5)
            standard_error = centered_rolling_mean(values=data['choice_probability_bootstrap_se'], window=5)
            axis.plot(data['trial_number'], mean, color=temperature_colors[temperature], label=f'$T_R={temperature:g}$')
            axis.fill_between(data['trial_number'], mean - standard_error, mean + standard_error, color=temperature_colors[temperature], alpha=0.16)
        axis.axhline(0.25, color='black', linewidth=1, linestyle=':', alpha=0.5)
        axis.axvline(10.5, color='black', linewidth=1, linestyle='--', alpha=0.35)
        axis.set_title(f'{chr(65 + index)}. Deck {deck}')
        axis.set_xlim(1, 100)
        axis.set_ylim(0, 1)
        axis.set_xlabel('Trial')
        axis.set_ylabel('Mean choice probability')
        style_axis(axis)
    axis = axes.flat[4]
    for deck in DECK_NAMES:
        data = trial_summary[(trial_summary['retrieval_temperature'] == 0.25) & (trial_summary['deck'] == deck)]
        mean = centered_rolling_mean(values=data['mean_choice_probability'], window=5)
        standard_error = centered_rolling_mean(values=data['choice_probability_bootstrap_se'], window=5)
        axis.plot(data['trial_number'], mean, color=deck_colors[deck], label=deck)
        axis.fill_between(data['trial_number'], mean - standard_error, mean + standard_error, color=deck_colors[deck], alpha=0.16)
    axis.axhline(0.25, color='black', linewidth=1, linestyle=':', alpha=0.5)
    axis.axvline(10.5, color='black', linewidth=1, linestyle='--', alpha=0.35)
    axis.set_title('E. Deck comparison ($T_R=0.25$)')
    axis.set_xlim(1, 100)
    axis.set_ylim(0, 1)
    axis.set_xlabel('Trial')
    axis.set_ylabel('Mean choice probability')
    style_axis(axis)
    axes.flat[5].set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    save_figure(figure=figure, savepath=visual_dir / 'figure_1_igt_deck_choice_probability_trajectories')


def plot_figure_1b(
    *,
    trial_summary: pd.DataFrame,
    deck_colors: dict[str, str],
    visual_dir: Path,
) -> None:
    """
    Plot deck-choice probability trajectories separately for every temperature.

    Args:
        trial_summary (pd.DataFrame): Agent-aggregated probability table.
        deck_colors (dict[str, str]): Stable deck color mapping.
        visual_dir (Path): Figure destination directory.
    """
    temperatures = sorted(trial_summary['retrieval_temperature'].unique())
    figure, axes = plt.subplots(2, 3, figsize=(12.3, 6.2), squeeze=False)
    for index, temperature in enumerate(temperatures):
        axis = axes.flat[index]
        for deck in DECK_NAMES:
            data = trial_summary[
                (trial_summary['retrieval_temperature'] == temperature) & (trial_summary['deck'] == deck)
            ]
            mean = centered_rolling_mean(values=data['mean_choice_probability'], window=5)
            standard_error = centered_rolling_mean(values=data['choice_probability_bootstrap_se'], window=5)
            axis.plot(data['trial_number'], mean, color=deck_colors[deck], label=deck)
            axis.fill_between(
                data['trial_number'],
                mean - standard_error,
                mean + standard_error,
                color=deck_colors[deck],
                alpha=0.16,
            )
        axis.axhline(0.25, color='black', linewidth=1, linestyle=':', alpha=0.5)
        axis.axvline(10.5, color='black', linewidth=1, linestyle='--', alpha=0.35)
        axis.set_title(f'{chr(65 + index)}. $T_R={temperature:g}$')
        axis.set_xlim(1, 100)
        axis.set_ylim(0, 1)
        axis.set_xlabel('Trial')
        axis.set_ylabel('Mean choice probability')
        style_axis(axis)
    for axis in axes.flat[len(temperatures):]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    save_figure(
        figure=figure,
        savepath=visual_dir / 'figure_1b_igt_deck_choice_probability_trajectories_by_temperature',
    )


def plot_block_metric(
    *,
    axis: Axes,
    block_summary: pd.DataFrame,
    metric: str,
    temperatures: Sequence[float],
    temperature_colors: dict[float, str],
    ylabel: str,
    ribbon: bool = False,
) -> None:
    """
    Plot an agent-bootstrap IGT block metric for every temperature.

    Args:
        axis (Axes): Destination axis.
        block_summary (pd.DataFrame): Block-level summary table.
        metric (str): Summary metric base name.
        temperatures (Sequence[float]): Retrieval temperature order.
        temperature_colors (dict[float, str]): Temperature color mapping.
        ylabel (str): Axis label.
    """
    for temperature in temperatures:
        group = block_summary[block_summary['retrieval_temperature'] == temperature].sort_values('block_id')
        estimate = group[f'{metric}_mean'].to_numpy()
        if ribbon:
            axis.plot(group['block_id'], estimate, color=temperature_colors[temperature], marker='o', label=f'$T_R={temperature:g}$')
            se = group[f'{metric}_bootstrap_se'].to_numpy()
            axis.fill_between(group['block_id'], estimate - se, estimate + se, color=temperature_colors[temperature], alpha=0.16)
        else:
            axis.errorbar(group['block_id'], estimate, yerr=np.vstack((estimate - group[f'{metric}_ci_low'], group[f'{metric}_ci_high'] - estimate)), color=temperature_colors[temperature], marker='o', capsize=0, label=f'$T_R={temperature:g}$')
    axis.axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.5)
    axis.set_xticks(range(1, 6))
    axis.set_xlabel('IGT block')
    axis.set_ylabel(ylabel)
    style_axis(axis)


def plot_figure_2(
    *,
    block_summary: pd.DataFrame,
    block_summary_10: pd.DataFrame,
    agent_table: pd.DataFrame,
    temperatures: Sequence[float],
    temperature_colors: dict[float, str],
    n_boot: int,
    random_state: int,
    visual_dir: Path,
) -> None:
    """
    Plot classical IGT Net Score learning and whole-task scores.

    Args:
        block_summary (pd.DataFrame): Block-level behavior summary.
        agent_table (pd.DataFrame): Whole-task agent table.
        temperatures (Sequence[float]): Retrieval temperature order.
        temperature_colors (dict[float, str]): Temperature color mapping.
        n_boot (int): Number of bootstrap resamples.
        random_state (int): Bootstrap and jitter seed.
        visual_dir (Path): Figure destination directory.
    """
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    plot_block_metric(
        axis=axes[0, 0],
        block_summary=block_summary,
        metric='net_score',
        temperatures=temperatures,
        temperature_colors=temperature_colors,
        ylabel='Net Score',
        ribbon=True,
    )
    axes[0, 0].set_title('A. Advantageous-choice learning')
    axes[0, 0].legend(frameon=False, fontsize=8)

    rng = np.random.default_rng(random_state)
    positions = temperature_positions(temperatures)
    for index, temperature in enumerate(temperatures):
        values = agent_table.loc[agent_table['retrieval_temperature'] == temperature, 'total_net_score']
        jitter = rng.uniform(-0.025, 0.025, len(values))
        axes[0, 1].scatter(
            np.full(len(values), positions[temperature]) + jitter,
            values,
            color=temperature_colors[temperature],
            alpha=0.25,
            s=14,
        )
        mean = float(values.mean())
        standard_error = bootstrap_mean_se(values=values, n_boot=n_boot, random_state=random_state + index)
    means = [agent_table.loc[agent_table['retrieval_temperature'] == temperature, 'total_net_score'].mean() for temperature in temperatures]
    standard_errors = [bootstrap_mean_se(values=agent_table.loc[agent_table['retrieval_temperature'] == temperature, 'total_net_score'], n_boot=n_boot, random_state=random_state + index) for index, temperature in enumerate(temperatures)]
    x_positions = list(positions.values())
    axes[0, 1].plot(x_positions, means, color='black', marker='o', linewidth=1)
    axes[0, 1].fill_between(x_positions, np.asarray(means) - np.asarray(standard_errors), np.asarray(means) + np.asarray(standard_errors), color='black', alpha=0.16)
    axes[0, 1].axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.5)
    axes[0, 1].set_xticks(list(positions.values()), [f'{temperature:g}' for temperature in temperatures])
    axes[0, 1].set_xlabel('Retrieval temperature')
    axes[0, 1].set_ylabel('Total Net Score')
    axes[0, 1].set_title('B. Overall advantageous choice')
    style_axis(axes[0, 1])
    plot_block_metric(
        axis=axes[1, 0],
        block_summary=block_summary_10,
        metric='net_score',
        temperatures=temperatures,
        temperature_colors=temperature_colors,
        ylabel='Net Score',
        ribbon=True,
    )
    axes[1, 0].set_title('C. Advantageous-choice learning (10-trial blocks)')
    axes[1, 1].set_visible(False)
    figure.tight_layout()
    save_figure(figure=figure, savepath=visual_dir / 'figure_2_igt_net_score_learning')


def plot_figure_3(
    *,
    block_summary: pd.DataFrame,
    block_summary_10: pd.DataFrame,
    agent_table: pd.DataFrame,
    temperatures: Sequence[float],
    temperature_colors: dict[float, str],
    n_boot: int,
    random_state: int,
    visual_dir: Path,
) -> None:
    """
    Plot punishment-frequency behavior and realized episodic retrieval lag.

    Args:
        block_summary (pd.DataFrame): Block-level behavior summary.
        agent_table (pd.DataFrame): Whole-task agent table.
        temperatures (Sequence[float]): Retrieval temperature order.
        temperature_colors (dict[float, str]): Temperature color mapping.
        n_boot (int): Number of bootstrap resamples.
        random_state (int): Bootstrap and jitter seed.
        visual_dir (Path): Figure destination directory.
    """
    figure, axes = plt.subplots(2, 3, figsize=(14, 7.2))
    positions = temperature_positions(temperatures)
    plot_block_metric(
        axis=axes[0, 0],
        block_summary=block_summary,
        metric='frequency_score',
        temperatures=temperatures,
        temperature_colors=temperature_colors,
        ylabel='Frequency Score',
        ribbon=True,
    )
    axes[0, 0].set_title('A. Rare vs frequent punishment')

    contrast_colors = {'b_minus_a': '#7F3C8D', 'd_minus_c': '#11A579'}
    contrast_labels = {'b_minus_a': 'B - A', 'd_minus_c': 'D - C'}
    for metric in ('b_minus_a', 'd_minus_c'):
        summary = summarize_agent_metrics(
            table=agent_table,
            group_columns=['retrieval_temperature'],
            metric_columns=[metric],
            n_boot=n_boot,
            random_state=random_state,
        ).sort_values('retrieval_temperature')
        estimate = summary[f'{metric}_mean'].to_numpy()
        x_positions = [positions[temperature] for temperature in summary['retrieval_temperature']]
        standard_error = summary[f'{metric}_bootstrap_se'].to_numpy()
        axes[0, 1].plot(x_positions, estimate, color=contrast_colors[metric], marker='o', label=contrast_labels[metric])
        axes[0, 1].fill_between(x_positions, estimate - standard_error, estimate + standard_error, color=contrast_colors[metric], alpha=0.16)
    axes[0, 1].axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.5)
    axes[0, 1].set_xticks(list(positions.values()), [f'{temperature:g}' for temperature in temperatures])
    axes[0, 1].set_xlabel('Retrieval temperature')
    axes[0, 1].set_ylabel('Choice-count contrast')
    axes[0, 1].set_title('B. Within-value frequency contrasts')
    axes[0, 1].legend(frameon=False)
    style_axis(axes[0, 1])

    lag_summary = summarize_agent_metrics(
        table=agent_table,
        group_columns=['retrieval_temperature'],
        metric_columns=['mean_sampled_episodic_lag'],
        n_boot=n_boot,
        random_state=random_state,
    ).sort_values('retrieval_temperature')
    rng = np.random.default_rng(random_state)
    for index, temperature in enumerate(temperatures):
        values = agent_table.loc[agent_table['retrieval_temperature'] == temperature, 'mean_sampled_episodic_lag']
        axes[0, 2].scatter(
            np.full(len(values), positions[temperature]) + rng.uniform(-0.025, 0.025, len(values)),
            values,
            color=temperature_colors[temperature],
            alpha=0.25,
            s=14,
        )
    estimate = lag_summary['mean_sampled_episodic_lag_mean'].to_numpy()
    x_positions = [positions[temperature] for temperature in lag_summary['retrieval_temperature']]
    standard_error = lag_summary['mean_sampled_episodic_lag_bootstrap_se'].to_numpy()
    axes[0, 2].plot(x_positions, estimate, color='black', marker='o')
    axes[0, 2].fill_between(x_positions, estimate - standard_error, estimate + standard_error, color='black', alpha=0.16)
    axes[0, 2].set_xticks(list(positions.values()), [f'{temperature:g}' for temperature in temperatures])
    axes[0, 2].set_xlabel('Retrieval temperature')
    axes[0, 2].set_ylabel('Mean sampled episodic lag')
    axes[0, 2].set_title('C. Realized episodic retrieval')
    style_axis(axes[0, 2])
    plot_block_metric(axis=axes[1, 0], block_summary=block_summary_10, metric='frequency_score', temperatures=temperatures, temperature_colors=temperature_colors, ylabel='Frequency Score', ribbon=True)
    axes[1, 0].set_title('D. Rare vs frequent punishment (10-trial blocks)')
    axes[1, 1].set_visible(False)
    axes[1, 2].set_visible(False)
    figure.tight_layout()
    save_figure(figure=figure, savepath=visual_dir / 'figure_3_igt_frequency_and_retrieval_mechanism')


def plot_figure_4(
    *,
    block_summary: pd.DataFrame,
    agent_table: pd.DataFrame,
    temperatures: Sequence[float],
    temperature_colors: dict[float, str],
    n_boot: int,
    random_state: int,
    visual_dir: Path,
) -> None:
    """
    Plot a compact IGT behavioral summary across retrieval temperatures.

    Args:
        block_summary (pd.DataFrame): Block-level behavior summary.
        agent_table (pd.DataFrame): Whole-task agent table.
        temperatures (Sequence[float]): Retrieval temperature order.
        temperature_colors (dict[float, str]): Temperature color mapping.
        n_boot (int): Number of bootstrap resamples.
        random_state (int): Bootstrap seed.
        visual_dir (Path): Figure destination directory.
    """
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), squeeze=False)
    plot_block_metric(
        axis=axes[0, 0],
        block_summary=block_summary,
        metric='net_score',
        temperatures=temperatures,
        temperature_colors=temperature_colors,
        ylabel='Net Score',
        ribbon=True,
    )
    axes[0, 0].set_title('A. Advantageous-choice learning')
    axes[0, 0].legend(frameon=False, fontsize=8)

    plot_block_metric(
        axis=axes[0, 1],
        block_summary=block_summary,
        metric='frequency_score',
        temperatures=temperatures,
        temperature_colors=temperature_colors,
        ylabel='Frequency Score',
        ribbon=True,
    )
    axes[0, 1].set_title('B. Rare vs frequent punishment')

    positions = temperature_positions(temperatures)
    contrast_colors = {'b_minus_a': '#7F3C8D', 'd_minus_c': '#11A579'}
    contrast_labels = {'b_minus_a': 'B - A', 'd_minus_c': 'D - C'}
    for metric in ('b_minus_a', 'd_minus_c'):
        summary = summarize_agent_metrics(
            table=agent_table,
            group_columns=['retrieval_temperature'],
            metric_columns=[metric],
            n_boot=n_boot,
            random_state=random_state,
        ).sort_values('retrieval_temperature')
        estimate = summary[f'{metric}_mean'].to_numpy()
        x_positions = [positions[temperature] for temperature in summary['retrieval_temperature']]
        standard_error = summary[f'{metric}_bootstrap_se'].to_numpy()
        axes[0, 2].plot(
            x_positions,
            estimate,
            color=contrast_colors[metric],
            marker='o',
            label=contrast_labels[metric],
        )
        axes[0, 2].fill_between(
            x_positions,
            estimate - standard_error,
            estimate + standard_error,
            color=contrast_colors[metric],
            alpha=0.16,
        )
    axes[0, 2].axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.5)
    axes[0, 2].set_xticks(list(positions.values()), [f'{temperature:g}' for temperature in temperatures])
    axes[0, 2].set_xlabel('Retrieval temperature')
    axes[0, 2].set_ylabel('Choice-count contrast')
    axes[0, 2].set_title('C. Within-value frequency contrasts')
    axes[0, 2].legend(frameon=False)
    style_axis(axes[0, 2])
    figure.tight_layout()
    save_figure(figure=figure, savepath=visual_dir / 'figure_4_igt_behavioral_summary')


def run_analysis(
    *,
    input_dir: Path,
    output_dir: Path,
    n_boot: int,
    random_seed: int,
    save_figures: bool,
    deck_colors: dict[str, str] | None = None,
    configured_temperature_colors: dict[str, str] | None = None,
) -> None:
    """
    Build and export the complete temporal-only IGT analysis.

    Args:
        input_dir (Path): Simulation cell directory.
        output_dir (Path): Analysis table and figure directory.
        n_boot (int): Number of agent-level bootstrap resamples.
        random_seed (int): Bootstrap and jitter seed.
        save_figures (bool): Whether to write figure files.
        deck_colors (dict[str, str] | None): Optional deck palette.
        configured_temperature_colors (dict[str, str] | None): Optional temperature palette.
    """
    cells = load_simulation_cells(input_dir)
    trial_table = build_trial_table(cells)
    n_anneal_trials = cells[0]['config']['choice_policy']['n_anneal_trials']
    agent_block_table, agent_table = build_agent_tables(
        trial_table=trial_table,
        n_anneal_trials=n_anneal_trials,
    )
    trial_summary = build_trial_probability_summary(
        trial_table=trial_table,
        n_boot=n_boot,
        random_state=random_seed,
    )
    block_summary = build_block_behavior_summary(
        agent_block_table=agent_block_table,
        n_boot=n_boot,
        random_state=random_seed,
    )
    trial_table_10 = add_igt_blocks(table=trial_table, block_size=10)
    agent_block_table_10, _ = build_agent_tables(trial_table=trial_table_10, n_anneal_trials=n_anneal_trials)
    block_summary_10 = build_block_behavior_summary(agent_block_table=agent_block_table_10, n_boot=n_boot, random_state=random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_summary.to_csv(output_dir / 'igt_trial_probability_summary.csv', index=False)
    block_summary.to_csv(output_dir / 'igt_block_behavior_summary.csv', index=False)
    agent_table.to_csv(output_dir / 'igt_agent_behavior_summary.csv', index=False)

    if not save_figures:
        return

    visual_dir = output_dir / 'visuals'
    visual_dir.mkdir(parents=True, exist_ok=True)
    temperatures = sorted(trial_table['retrieval_temperature'].unique())
    resolved_deck_colors = {**DEFAULT_DECK_COLORS, **(deck_colors or {})}
    temperature_colors = resolve_temperature_colors(
        temperatures=temperatures,
        configured_colors=configured_temperature_colors,
    )
    plot_figure_1(
        trial_summary=trial_summary,
        deck_colors=resolved_deck_colors,
        temperature_colors=temperature_colors,
        visual_dir=visual_dir,
    )
    plot_figure_1b(
        trial_summary=trial_summary,
        deck_colors=resolved_deck_colors,
        visual_dir=visual_dir,
    )
    plot_figure_2(
        block_summary=block_summary,
        block_summary_10=block_summary_10,
        agent_table=agent_table,
        temperatures=temperatures,
        temperature_colors=temperature_colors,
        n_boot=n_boot,
        random_state=random_seed,
        visual_dir=visual_dir,
    )
    plot_figure_3(
        block_summary=block_summary,
        block_summary_10=block_summary_10,
        agent_table=agent_table,
        temperatures=temperatures,
        temperature_colors=temperature_colors,
        n_boot=n_boot,
        random_state=random_seed,
        visual_dir=visual_dir,
    )
    plot_figure_4(
        block_summary=block_summary,
        agent_table=agent_table,
        temperatures=temperatures,
        temperature_colors=temperature_colors,
        n_boot=n_boot,
        random_state=random_seed,
        visual_dir=visual_dir,
    )


def main(args: argparse.Namespace) -> None:
    """
    Resolve paths and run the configured temporal-only IGT analysis.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    config = load_analysis_config(args.config)
    input_dir = args.input_dir or filepaths.EXPERIMENTS_DIR / config['experiment_name']
    output_dir = args.output_dir or filepaths.OUTPUT_DIR / config['output_dir_name']
    run_analysis(
        input_dir=input_dir,
        output_dir=output_dir,
        n_boot=config['n_boot'],
        random_seed=config['random_seed'],
        save_figures=config['save_figures'] and not args.no_save_figures,
        deck_colors=config.get('deck_colors'),
        configured_temperature_colors=config.get('retrieval_temperature_colors'),
    )


if __name__ == '__main__':
    main(parse_args())
