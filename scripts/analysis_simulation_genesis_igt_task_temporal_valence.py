"""
Analyze temporal-valence GENESIS Iowa Gambling Task simulations.
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


DEFAULT_ANALYSIS_CONFIG_PATH = filepaths.ROOT / 'configs' / 'simulation_genesis_igt_task_models.yaml'
DECK_NAMES = ('A', 'B', 'C', 'D')
PROBABILITY_COLUMNS = {'A': 'p_a', 'B': 'p_b', 'C': 'p_c', 'D': 'p_d'}
CHOICE_COLUMNS = {'A': 'i_a', 'B': 'i_b', 'C': 'i_c', 'D': 'i_d'}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse temporal-valence IGT analysis arguments.

    Args:
        argv (list[str] | None): Optional CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Analyze GENESIS temporal-valence IGT simulations.')
    parser.add_argument('--config', type=Path, default=DEFAULT_ANALYSIS_CONFIG_PATH)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-save-figures', action='store_true')
    return parser.parse_args(argv)


def load_analysis_config(config_path: Path) -> dict[str, Any]:
    """
    Load temporal-valence IGT analysis controls.

    Args:
        config_path (Path): Analysis configuration file.

    Returns:
        dict[str, Any]: Parsed analysis controls.
    """
    return yaml.safe_load(config_path.read_text(encoding='utf-8'))


def fixed_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Remove temporal-valence grid fields from a resolved cell configuration.

    Args:
        config (dict[str, Any]): Resolved simulation configuration.

    Returns:
        dict[str, Any]: Fixed simulation fields.
    """
    return {
        key: value
        for key, value in config.items()
        if key not in {'temporal_weight', 'valence_weight'}
    }


def validate_cell(*, cell: dict[str, Any], reference_cell: dict[str, Any] | None) -> None:
    """
    Validate one temporal-valence IGT grid cell.

    Args:
        cell (dict[str, Any]): Loaded simulation payload.
        reference_cell (dict[str, Any] | None): First validated payload.

    Raises:
        ValueError: If the stored experiment contract is incompatible.
    """
    required_cell_keys = {
        'config',
        'query_valence',
        'temporal_weight',
        'valence_weight',
        'retrieval_temperature',
        'source_schedule_path',
        'reward_scaling_factor',
        'temporal_seed',
        'agent_runs',
    }
    missing_keys = required_cell_keys.difference(cell)
    if missing_keys:
        raise ValueError(f'Missing simulation payload fields: {sorted(missing_keys)}.')

    config = cell['config']
    if cell['query_valence'] not in {'positive', 'negative'}:
        raise ValueError('Unknown query valence condition.')
    if config['query_valence'] != cell['query_valence']:
        raise ValueError('Payload query valence does not match the resolved configuration.')
    if config['temporal_weight'] != cell['temporal_weight']:
        raise ValueError('Payload temporal weight does not match the resolved configuration.')
    if config['valence_weight'] != cell['valence_weight']:
        raise ValueError('Payload valence weight does not match the resolved configuration.')
    if not np.isclose(cell['valence_weight'], 1.0 - cell['temporal_weight']):
        raise ValueError('Temporal and valence weights must sum to one.')
    if cell['retrieval_temperature'] != 0.25 or config['retrieval_temperature'] != 0.25:
        raise ValueError('Temporal-valence IGT retrieval temperature must be fixed at 0.25.')
    if config['episodic_sampling']['sampling_budget'] != 20:
        raise ValueError('Temporal-valence IGT sampling budget must be fixed at 20.')
    if config['episodic_sampling']['sample_replacement'] is not True:
        raise ValueError('Temporal-valence IGT sampling must use replacement.')
    if config['episodic_sampling']['query_key_normalization'] != 'zscore':
        raise ValueError('Temporal-valence IGT requires zscore query-key normalization.')

    required_trial_keys = {
        'trial_id', 'chosen_deck', 'deck_draw_index', 'reward', 'cumulative_reward', 'action_values',
        'p_a', 'p_b', 'p_c', 'p_d', 'choice_temperature', 'query_valence', 'temporal_weight',
        'valence_weight', 'query_key_normalization', 'retrieval_temperature',
        'previous_experience_counts', 'retrieved_episode_indices', 'retrieval_metadata',
    }
    agent_ids = []
    for agent_run in cell['agent_runs']:
        agent_ids.append(agent_run['agent_id'])
        simulation_log = agent_run['simulation_log']
        if len(simulation_log) != config['task']['n_trials']:
            raise ValueError('Agent trial count does not match the resolved configuration.')
        for trial_id, trial in enumerate(simulation_log):
            missing_trial_keys = required_trial_keys.difference(trial)
            if missing_trial_keys:
                raise ValueError(f'Missing trial fields: {sorted(missing_trial_keys)}.')
            if trial['trial_id'] != trial_id:
                raise ValueError('Trial IDs must be contiguous and zero-based.')
            if trial['query_valence'] != cell['query_valence']:
                raise ValueError('Trial query valence does not match its cell.')
            if trial['temporal_weight'] != cell['temporal_weight']:
                raise ValueError('Trial temporal weight does not match its cell.')
            if trial['valence_weight'] != cell['valence_weight']:
                raise ValueError('Trial valence weight does not match its cell.')
            if trial['query_key_normalization'] != 'zscore':
                raise ValueError('Trial normalization does not match the experiment contract.')
            if trial['retrieval_temperature'] != 0.25:
                raise ValueError('Trial retrieval temperature does not match the experiment contract.')
            if set(trial['action_values']) != set(DECK_NAMES):
                raise ValueError('Action values must contain all four decks.')
            if set(trial['retrieved_episode_indices']) != set(DECK_NAMES):
                raise ValueError('Retrieved episode indices must contain all four decks.')
            if set(trial['retrieval_metadata']) != set(DECK_NAMES):
                raise ValueError('Retrieval metadata must contain all four decks.')

    if len(agent_ids) != config['n_agents']:
        raise ValueError('Agent-run count does not match the resolved configuration.')
    if reference_cell is None:
        return
    if fixed_config(config) != fixed_config(reference_cell['config']):
        raise ValueError('Simulation cells have incompatible fixed simulation settings.')
    if cell['query_valence'] != reference_cell['query_valence']:
        raise ValueError('One analysis input must contain exactly one query-valence condition.')
    for key in ('source_schedule_path', 'reward_scaling_factor', 'temporal_seed'):
        if cell[key] != reference_cell[key]:
            raise ValueError(f'Simulation cells have incompatible {key}.')
    reference_ids = [agent_run['agent_id'] for agent_run in reference_cell['agent_runs']]
    if agent_ids != reference_ids:
        raise ValueError('Simulation cells do not preserve shared agent IDs and order.')


def load_simulation_cells(input_dir: Path) -> list[dict[str, Any]]:
    """
    Load, validate, and temporal-weight-sort simulation cells.

    Args:
        input_dir (Path): Directory containing saved `.npy` cells.

    Returns:
        list[dict[str, Any]]: Compatible cells in temporal-weight order.

    Raises:
        ValueError: If no cells are found or cells are incompatible.
    """
    simulation_paths = list(input_dir.glob('*_temporal_weight-*.npy'))
    if not simulation_paths:
        raise ValueError(f'No temporal-valence IGT simulation files found in {input_dir}.')
    cells = [np.load(path, allow_pickle=True).item() for path in simulation_paths]
    cells.sort(key=lambda cell: float(cell['temporal_weight']))
    reference_cell = None
    for cell in cells:
        validate_cell(cell=cell, reference_cell=reference_cell)
        if reference_cell is None:
            reference_cell = cell
    weights = [cell['temporal_weight'] for cell in cells]
    if len(weights) != len(set(weights)):
        raise ValueError('Each temporal weight must have exactly one simulation cell.')
    return cells


def reward_valence(reward: float) -> str:
    """
    Map one observed outcome to its stored valence category.

    Args:
        reward (float): Observed scaled IGT outcome.

    Returns:
        str: Positive, negative, or zero valence label.
    """
    if reward > 0.0:
        return 'positive'
    if reward < 0.0:
        return 'negative'
    return 'zero'


def build_trial_table(cells: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten histories and compute realized sampled-memory mechanism measures.

    Args:
        cells (Sequence[dict[str, Any]]): Validated simulation payloads.

    Returns:
        pd.DataFrame: One row per temporal weight, agent, and trial.
    """
    rows = []
    for cell in cells:
        for agent_run in cell['agent_runs']:
            history = agent_run['simulation_log']
            for trial in history:
                sampled_indices = [
                    episode_index
                    for deck in DECK_NAMES
                    for episode_index in trial['retrieved_episode_indices'][deck]
                ]
                sampled_valences = [reward_valence(history[index]['reward']) for index in sampled_indices]
                candidate_valences = [
                    reward_valence(history[index]['reward'])
                    for deck in DECK_NAMES
                    for index in trial['retrieval_metadata'][deck]['candidate_indices']
                ]
                query_valence = trial['query_valence']
                row = {
                    'query_valence': query_valence,
                    'temporal_weight': float(trial['temporal_weight']),
                    'valence_weight': float(trial['valence_weight']),
                    'agent_id': agent_run['agent_id'],
                    'trial_id': trial['trial_id'],
                    'trial_number': trial['trial_id'] + 1,
                    'chosen_deck': trial['chosen_deck'],
                    'reward': trial['reward'],
                    'cumulative_reward': trial['cumulative_reward'],
                    'choice_temperature': trial['choice_temperature'],
                    'retrieved_episode_indices': trial['retrieved_episode_indices'],
                    'mean_sampled_episodic_lag': (
                        float(np.mean([trial['trial_id'] - index for index in sampled_indices]))
                        if sampled_indices else np.nan
                    ),
                    'sampled_positive_proportion': (
                        sampled_valences.count('positive') / len(sampled_valences) if sampled_valences else np.nan
                    ),
                    'sampled_negative_proportion': (
                        sampled_valences.count('negative') / len(sampled_valences) if sampled_valences else np.nan
                    ),
                    'sampled_zero_proportion': (
                        sampled_valences.count('zero') / len(sampled_valences) if sampled_valences else np.nan
                    ),
                    'query_valence_available': bool(query_valence in candidate_valences),
                    'sampled_query_valence_proportion': (
                        sampled_valences.count(query_valence) / len(sampled_valences) if sampled_valences else np.nan
                    ),
                    'block_id': trial['trial_id'] // 20 + 1,
                }
                row.update({f'q_{deck.lower()}': trial['action_values'][deck] for deck in DECK_NAMES})
                row.update({column: trial[column] for column in PROBABILITY_COLUMNS.values()})
                row.update({CHOICE_COLUMNS[deck]: int(trial['chosen_deck'] == deck) for deck in DECK_NAMES})
                rows.append(row)
    return pd.DataFrame(rows).sort_values(['temporal_weight', 'agent_id', 'trial_id']).reset_index(drop=True)


def build_agent_tables(
    *,
    trial_table: pd.DataFrame,
    n_anneal_trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute block and whole-task behavioral and mechanism summaries.

    Args:
        trial_table (pd.DataFrame): Flattened IGT trial records.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Agent-block and whole-task agent tables.
    """
    group_columns = ['query_valence', 'temporal_weight', 'agent_id']
    count_columns = list(CHOICE_COLUMNS.values())
    block_table = trial_table.groupby(group_columns + ['block_id'], sort=True)[count_columns].sum().reset_index()
    block_table['net_score'] = block_table['i_c'] + block_table['i_d'] - block_table['i_a'] - block_table['i_b']
    block_table['frequency_score'] = block_table['i_b'] + block_table['i_d'] - block_table['i_a'] - block_table['i_c']
    aggregations = {column: 'sum' for column in count_columns}
    aggregations.update({
        'reward': 'sum',
        'cumulative_reward': 'last',
    })
    agent_table = trial_table.groupby(group_columns, sort=True).agg(aggregations).reset_index()
    agent_table = agent_table.rename(columns={
        'i_a': 'total_a_choices', 'i_b': 'total_b_choices', 'i_c': 'total_c_choices', 'i_d': 'total_d_choices',
        'reward': 'total_reward', 'cumulative_reward': 'final_cumulative_reward',
    })
    lag_table = (
        trial_table[trial_table['trial_id'] >= n_anneal_trials]
        .groupby(group_columns, sort=True)['mean_sampled_episodic_lag']
        .mean()
        .reset_index()
    )
    agent_table = agent_table.merge(lag_table, on=group_columns)
    agent_table['total_net_score'] = (
        agent_table['total_c_choices'] + agent_table['total_d_choices']
        - agent_table['total_a_choices'] - agent_table['total_b_choices']
    )
    agent_table['total_frequency_score'] = (
        agent_table['total_b_choices'] + agent_table['total_d_choices']
        - agent_table['total_a_choices'] - agent_table['total_c_choices']
    )
    agent_table['b_minus_a'] = agent_table['total_b_choices'] - agent_table['total_a_choices']
    agent_table['d_minus_c'] = agent_table['total_d_choices'] - agent_table['total_c_choices']
    return block_table, agent_table


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
    random_generator = np.random.default_rng(random_state)
    sample_indices = random_generator.integers(0, len(array), size=(n_boot, len(array)))
    bootstrap_means = array[sample_indices].mean(axis=1)
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(array.mean()), float(ci_low), float(ci_high)


def bootstrap_mean_se(*, values: Sequence[float], n_boot: int, random_state: int) -> float:
    """
    Estimate the bootstrap standard error of an agent-level mean.

    Args:
        values (Sequence[float]): Independent agent-level observations.
        n_boot (int): Number of bootstrap resamples.
        random_state (int): Deterministic bootstrap seed.

    Returns:
        float: Bootstrap standard error.
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return np.nan
    random_generator = np.random.default_rng(random_state)
    sample_indices = random_generator.integers(0, len(array), size=(n_boot, len(array)))
    return float(array[sample_indices].mean(axis=1).std(ddof=1))


def build_trial_probability_summary(
    *,
    trial_table: pd.DataFrame,
    n_boot: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Summarize stored deck-choice probabilities across agents.

    Args:
        trial_table (pd.DataFrame): Flattened IGT trial records.

    Returns:
        pd.DataFrame: Weight, trial, and deck probability summaries.
    """
    rows = []
    group_index = 0
    for group_key, group in trial_table.groupby(['query_valence', 'temporal_weight', 'trial_number'], sort=True):
        for deck in DECK_NAMES:
            mean, ci_low, ci_high = bootstrap_mean_ci(
                values=group[PROBABILITY_COLUMNS[deck]],
                n_boot=n_boot,
                random_state=random_seed + group_index,
            )
            rows.append({
                'query_valence': group_key[0],
                'temporal_weight': group_key[1],
                'trial_number': group_key[2],
                'deck': deck,
                'mean_choice_probability': mean,
                'choice_probability_ci_low': ci_low,
                'choice_probability_ci_high': ci_high,
                'choice_probability_bootstrap_se': bootstrap_mean_se(
                    values=group[PROBABILITY_COLUMNS[deck]],
                    n_boot=n_boot,
                    random_state=random_seed + group_index,
                ),
                'observed_choice_proportion': group[CHOICE_COLUMNS[deck]].mean(),
            })
            group_index += 1
    return pd.DataFrame(rows)


def build_block_behavior_summary(
    *,
    block_table: pd.DataFrame,
    n_boot: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Summarize block-level IGT behavioral measures across agents.

    Args:
        block_table (pd.DataFrame): One row per weight, agent, and block.
        n_boot (int): Number of bootstrap resamples.
        random_seed (int): Base bootstrap seed.

    Returns:
        pd.DataFrame: Temporal-weight block behavior summary.
    """
    metrics = ['net_score', 'frequency_score', 'i_a', 'i_b', 'i_c', 'i_d']
    rows = []
    for group_index, (group_key, group) in enumerate(
        block_table.groupby(['query_valence', 'temporal_weight', 'block_id'], sort=True),
    ):
        row = dict(zip(['query_valence', 'temporal_weight', 'block_id'], group_key))
        for metric_index, metric in enumerate(metrics):
            seed = random_seed + group_index * len(metrics) + metric_index
            mean, ci_low, ci_high = bootstrap_mean_ci(values=group[metric], n_boot=n_boot, random_state=seed)
            row[f'{metric}_mean'] = mean
            row[f'{metric}_ci_low'] = ci_low
            row[f'{metric}_ci_high'] = ci_high
            row[f'{metric}_bootstrap_se'] = bootstrap_mean_se(
                values=group[metric], n_boot=n_boot, random_state=seed,
            )
        rows.append(row)
    return pd.DataFrame(rows).rename(columns={'i_a_mean': 'a_choices_mean', 'i_b_mean': 'b_choices_mean',
                                               'i_c_mean': 'c_choices_mean', 'i_d_mean': 'd_choices_mean'})


def save_figure(*, figure: plt.Figure, savepath: Path) -> None:
    """
    Save one analysis figure as PDF and SVG.

    Args:
        figure (plt.Figure): Completed matplotlib figure.
        savepath (Path): Export path without an extension.
    """
    figure.savefig(savepath.with_suffix('.pdf'), bbox_inches='tight')
    figure.savefig(savepath.with_suffix('.svg'), bbox_inches='tight')
    plt.close(figure)


def centered_rolling_mean(*, values: pd.Series, window: int) -> pd.Series:
    """
    Smooth one plotted trajectory with a centered rolling mean.

    Args:
        values (pd.Series): Ordered trajectory values.
        window (int): Centered smoothing-window width.

    Returns:
        pd.Series: Smoothed values with the original index.
    """
    return values.rolling(window=window, center=True, min_periods=1).mean()


def style_axis(axis: Axes) -> None:
    """
    Apply the shared manuscript axis style.

    Args:
        axis (Axes): Axis to style.
    """
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.tick_params(axis='both', length=0)


def resolve_temporal_weight_colors(
    *,
    temporal_weights: Sequence[float],
    configured_colors: dict[str, str],
) -> dict[float, str]:
    """
    Resolve the required configured color for every observed temporal weight.

    Args:
        temporal_weights (Sequence[float]): Observed temporal-weight grid.
        configured_colors (dict[str, str]): YAML temporal-weight color mapping.

    Returns:
        dict[float, str]: Colors keyed by numeric temporal weight.

    Raises:
        ValueError: If an observed temporal weight lacks a configured color.
    """
    missing_weights = [weight for weight in temporal_weights if str(weight) not in configured_colors]
    if missing_weights:
        raise ValueError(f'Missing temporal-weight colors for: {missing_weights}.')
    return {weight: configured_colors[str(weight)] for weight in temporal_weights}


def resolve_deck_colors(*, configured_colors: dict[str, str]) -> dict[str, str]:
    """
    Resolve the required configured color for every IGT deck.

    Args:
        configured_colors (dict[str, str]): YAML deck color mapping.

    Returns:
        dict[str, str]: Colors keyed by IGT deck name.

    Raises:
        ValueError: If an IGT deck lacks a configured color.
    """
    missing_decks = set(DECK_NAMES).difference(configured_colors)
    if missing_decks:
        raise ValueError(f'Missing deck colors for: {sorted(missing_decks)}.')
    return {deck: configured_colors[deck] for deck in DECK_NAMES}


def resolve_contrast_colors(*, configured_colors: dict[str, str]) -> dict[str, str]:
    """
    Resolve configured colors for the two within-value choice contrasts.

    Args:
        configured_colors (dict[str, str]): YAML contrast color mapping.

    Returns:
        dict[str, str]: Colors keyed by contrast metric name.

    Raises:
        ValueError: If either Figure 3 contrast lacks a configured color.
    """
    contrast_metrics = ('b_minus_a', 'd_minus_c')
    missing_metrics = set(contrast_metrics).difference(configured_colors)
    if missing_metrics:
        raise ValueError(f'Missing contrast colors for: {sorted(missing_metrics)}.')
    return {metric: configured_colors[metric] for metric in contrast_metrics}


def temporal_weight_positions(temporal_weights: Sequence[float]) -> dict[float, int]:
    """
    Map ordered temporal weights to evenly spaced categorical plot positions.

    Args:
        temporal_weights (Sequence[float]): Temporal weights in display order.

    Returns:
        dict[float, int]: Evenly spaced position for every temporal weight.
    """
    return {weight: index for index, weight in enumerate(temporal_weights)}


def summarize_agent_metric(
    *,
    agent_table: pd.DataFrame,
    metric: str,
    n_boot: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Summarize one whole-task metric across agents by temporal weight.

    Args:
        agent_table (pd.DataFrame): Whole-task agent records.
        metric (str): Agent-level metric column.
        n_boot (int): Number of bootstrap resamples.
        random_seed (int): Base bootstrap seed.

    Returns:
        pd.DataFrame: One bootstrap summary row per temporal weight.
    """
    rows = []
    for index, (weight, group) in enumerate(agent_table.groupby('temporal_weight', sort=True)):
        mean, ci_low, ci_high = bootstrap_mean_ci(
            values=group[metric],
            n_boot=n_boot,
            random_state=random_seed + index,
        )
        rows.append({
            'temporal_weight': weight,
            f'{metric}_mean': mean,
            f'{metric}_ci_low': ci_low,
            f'{metric}_ci_high': ci_high,
            f'{metric}_bootstrap_se': bootstrap_mean_se(
                values=group[metric],
                n_boot=n_boot,
                random_state=random_seed + index,
            ),
        })
    return pd.DataFrame(rows)


def add_igt_blocks(*, table: pd.DataFrame, block_size: int) -> pd.DataFrame:
    """
    Assign consecutive IGT blocks to one trial table.

    Args:
        table (pd.DataFrame): Trial-level analysis table.
        block_size (int): Number of trials per block.

    Returns:
        pd.DataFrame: Trial table with a replacement block assignment.
    """
    result = table.copy()
    result['block_id'] = result['trial_id'] // block_size + 1
    return result


def remove_obsolete_outputs(*, output_dir: Path) -> None:
    """
    Remove analysis artifacts superseded by the aligned Figure 1–3 layout.

    Args:
        output_dir (Path): Condition-specific analysis output directory.
    """
    obsolete_paths = (
        output_dir / 'igt_retrieval_mechanism_summary.csv',
        output_dir / 'igt_validation_summary.csv',
        output_dir / 'visuals' / 'figure_2_igt_behavioral_scores.pdf',
        output_dir / 'visuals' / 'figure_3_igt_retrieved_valence_composition.pdf',
        output_dir / 'visuals' / 'figure_4_igt_retrieval_to_value_mechanism.pdf',
    )
    for obsolete_path in obsolete_paths:
        obsolete_path.unlink(missing_ok=True)


def plot_block_metric(
    *,
    axis: Axes,
    block_summary: pd.DataFrame,
    metric: str,
    temporal_weights: Sequence[float],
    temporal_weight_colors: dict[float, str],
    ylabel: str,
) -> None:
    """
    Plot one block-level IGT metric for every temporal weight.

    Args:
        axis (Axes): Destination axis.
        block_summary (pd.DataFrame): Bootstrap block summary.
        metric (str): Summary metric base name.
        temporal_weights (Sequence[float]): Temporal weights in display order.
        temporal_weight_colors (dict[float, str]): Configured weight colors.
        ylabel (str): Axis label.
    """
    for weight in temporal_weights:
        group = block_summary[block_summary['temporal_weight'] == weight].sort_values('block_id')
        estimate = group[f'{metric}_mean'].to_numpy()
        standard_error = group[f'{metric}_bootstrap_se'].to_numpy()
        axis.plot(
            group['block_id'],
            estimate,
            color=temporal_weight_colors[weight],
            marker='o',
            label=f'$w_T={weight:g}$',
        )
        axis.fill_between(
            group['block_id'],
            estimate - standard_error,
            estimate + standard_error,
            color=temporal_weight_colors[weight],
            alpha=0.16,
        )
    axis.axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.5)
    axis.set_xlabel('IGT block')
    axis.set_ylabel(ylabel)
    style_axis(axis)


def plot_figure_1(
    *,
    trial_summary: pd.DataFrame,
    deck_colors: dict[str, str],
    temporal_weight_colors: dict[float, str],
    visual_dir: Path,
) -> None:
    """
    Plot deck-fixed and temporal-weight-fixed choice-probability trajectories.

    Args:
        trial_summary (pd.DataFrame): Choice probability summary.
        deck_colors (dict[str, str]): Configured deck color mapping.
        temporal_weight_colors (dict[float, str]): Configured temporal-weight color mapping.
        visual_dir (Path): Figure destination directory.
    """
    temporal_weights = sorted(trial_summary['temporal_weight'].unique())
    n_columns = max(len(DECK_NAMES), len(temporal_weights))
    figure, axes = plt.subplots(2, n_columns, figsize=(3.1 * n_columns, 6.2), squeeze=False)
    for index, deck in enumerate(DECK_NAMES):
        axis = axes[0, index]
        for weight in temporal_weights:
            data = trial_summary[(trial_summary['temporal_weight'] == weight) & (trial_summary['deck'] == deck)]
            mean = centered_rolling_mean(values=data['mean_choice_probability'], window=5)
            standard_error = centered_rolling_mean(values=data['choice_probability_bootstrap_se'], window=5)
            axis.plot(data['trial_number'], mean, color=temporal_weight_colors[weight], label=f'$w_T={weight:g}$')
            axis.fill_between(
                data['trial_number'],
                mean - standard_error,
                mean + standard_error,
                color=temporal_weight_colors[weight],
                alpha=0.16,
            )
        axis.axhline(0.25, color='black', linewidth=1, linestyle=':', alpha=0.5)
        axis.axvline(10.5, color='black', linewidth=1, linestyle='--', alpha=0.35)
        axis.set_title(f'{chr(65 + index)}. Deck {deck}')
        axis.set_xlim(1, 100)
        axis.set_ylim(0, 1)
        axis.set_xlabel('Trial')
        axis.set_ylabel('Mean choice probability')
        style_axis(axis)
    for index, weight in enumerate(temporal_weights):
        axis = axes[1, index]
        for deck in DECK_NAMES:
            data = trial_summary[(trial_summary['temporal_weight'] == weight) & (trial_summary['deck'] == deck)]
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
        axis.set_title(f'{chr(65 + len(DECK_NAMES) + index)}. Deck comparison ($w_T={weight:g}$)')
        axis.set_xlim(1, 100)
        axis.set_ylim(0, 1)
        axis.set_xlabel('Trial')
        axis.set_ylabel('Mean choice probability')
        style_axis(axis)
    for index in range(len(DECK_NAMES), n_columns):
        axes[0, index].set_visible(False)
    for index in range(len(temporal_weights), n_columns):
        axes[1, index].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    save_figure(figure=figure, savepath=visual_dir / 'figure_1_igt_deck_choice_probability_trajectories')


def plot_figure_2(
    *,
    block_summary: pd.DataFrame,
    block_summary_10: pd.DataFrame,
    agent_table: pd.DataFrame,
    temporal_weights: Sequence[float],
    temporal_weight_colors: dict[float, str],
    n_boot: int,
    random_seed: int,
    visual_dir: Path,
) -> None:
    """
    Plot classical IGT Net Score learning and whole-task scores by temporal weight.

    Args:
        block_summary (pd.DataFrame): Twenty-trial block summary.
        block_summary_10 (pd.DataFrame): Ten-trial block summary.
        agent_table (pd.DataFrame): Whole-task agent summary.
        temporal_weights (Sequence[float]): Temporal weights in display order.
        temporal_weight_colors (dict[float, str]): Configured temporal-weight colors.
        n_boot (int): Number of bootstrap resamples.
        random_seed (int): Bootstrap and jitter seed.
        visual_dir (Path): Figure destination directory.
    """
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    plot_block_metric(
        axis=axes[0, 0],
        block_summary=block_summary,
        metric='net_score',
        temporal_weights=temporal_weights,
        temporal_weight_colors=temporal_weight_colors,
        ylabel='Net Score',
    )
    axes[0, 0].set_title('A. Advantageous-choice learning')
    axes[0, 0].legend(frameon=False, fontsize=8)
    positions = temporal_weight_positions(temporal_weights)
    random_generator = np.random.default_rng(random_seed)
    for weight in temporal_weights:
        values = agent_table.loc[agent_table['temporal_weight'] == weight, 'total_net_score']
        axes[0, 1].scatter(
            np.full(len(values), positions[weight]) + random_generator.uniform(-0.025, 0.025, len(values)),
            values,
            color=temporal_weight_colors[weight],
            alpha=0.25,
            s=14,
        )
    summary = summarize_agent_metric(
        agent_table=agent_table,
        metric='total_net_score',
        n_boot=n_boot,
        random_seed=random_seed,
    ).sort_values('temporal_weight')
    estimate = summary['total_net_score_mean'].to_numpy()
    standard_error = summary['total_net_score_bootstrap_se'].to_numpy()
    x_positions = [positions[weight] for weight in summary['temporal_weight']]
    axes[0, 1].plot(x_positions, estimate, color='black', marker='o', linewidth=1)
    axes[0, 1].fill_between(
        x_positions,
        estimate - standard_error,
        estimate + standard_error,
        color='black',
        alpha=0.16,
    )
    axes[0, 1].axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.5)
    axes[0, 1].set_xticks(list(positions.values()), [f'{weight:g}' for weight in temporal_weights])
    axes[0, 1].set_xlabel('Temporal weight')
    axes[0, 1].set_ylabel('Total Net Score')
    axes[0, 1].set_title('B. Overall advantageous choice')
    style_axis(axes[0, 1])
    plot_block_metric(
        axis=axes[1, 0],
        block_summary=block_summary_10,
        metric='net_score',
        temporal_weights=temporal_weights,
        temporal_weight_colors=temporal_weight_colors,
        ylabel='Net Score',
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
    temporal_weights: Sequence[float],
    temporal_weight_colors: dict[float, str],
    contrast_colors: dict[str, str],
    n_boot: int,
    random_seed: int,
    visual_dir: Path,
) -> None:
    """
    Plot frequency behavior, within-value contrasts, and episodic lag by temporal weight.

    Args:
        block_summary (pd.DataFrame): Twenty-trial block summary.
        block_summary_10 (pd.DataFrame): Ten-trial block summary.
        agent_table (pd.DataFrame): Whole-task agent summary.
        temporal_weights (Sequence[float]): Temporal weights in display order.
        temporal_weight_colors (dict[float, str]): Configured temporal-weight colors.
        contrast_colors (dict[str, str]): Configured within-value contrast colors.
        n_boot (int): Number of bootstrap resamples.
        random_seed (int): Bootstrap and jitter seed.
        visual_dir (Path): Figure destination directory.
    """
    figure, axes = plt.subplots(2, 3, figsize=(14, 7.2))
    plot_block_metric(
        axis=axes[0, 0],
        block_summary=block_summary,
        metric='frequency_score',
        temporal_weights=temporal_weights,
        temporal_weight_colors=temporal_weight_colors,
        ylabel='Frequency Score',
    )
    axes[0, 0].set_title('A. Rare vs frequent punishment')
    positions = temporal_weight_positions(temporal_weights)
    contrast_labels = {'b_minus_a': 'B - A', 'd_minus_c': 'D - C'}
    for metric in ('b_minus_a', 'd_minus_c'):
        summary = summarize_agent_metric(
            agent_table=agent_table,
            metric=metric,
            n_boot=n_boot,
            random_seed=random_seed,
        ).sort_values('temporal_weight')
        estimate = summary[f'{metric}_mean'].to_numpy()
        standard_error = summary[f'{metric}_bootstrap_se'].to_numpy()
        x_positions = [positions[weight] for weight in summary['temporal_weight']]
        axes[0, 1].plot(x_positions, estimate, color=contrast_colors[metric], marker='o', label=contrast_labels[metric])
        axes[0, 1].fill_between(
            x_positions,
            estimate - standard_error,
            estimate + standard_error,
            color=contrast_colors[metric],
            alpha=0.16,
        )
    axes[0, 1].axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.5)
    axes[0, 1].set_xticks(list(positions.values()), [f'{weight:g}' for weight in temporal_weights])
    axes[0, 1].set_xlabel('Temporal weight')
    axes[0, 1].set_ylabel('Choice-count contrast')
    axes[0, 1].set_title('B. Within-value frequency contrasts')
    axes[0, 1].legend(frameon=False)
    style_axis(axes[0, 1])
    lag_summary = summarize_agent_metric(
        agent_table=agent_table,
        metric='mean_sampled_episodic_lag',
        n_boot=n_boot,
        random_seed=random_seed,
    ).sort_values('temporal_weight')
    random_generator = np.random.default_rng(random_seed)
    for weight in temporal_weights:
        values = agent_table.loc[agent_table['temporal_weight'] == weight, 'mean_sampled_episodic_lag']
        axes[0, 2].scatter(
            np.full(len(values), positions[weight]) + random_generator.uniform(-0.025, 0.025, len(values)),
            values,
            color=temporal_weight_colors[weight],
            alpha=0.25,
            s=14,
        )
    estimate = lag_summary['mean_sampled_episodic_lag_mean'].to_numpy()
    standard_error = lag_summary['mean_sampled_episodic_lag_bootstrap_se'].to_numpy()
    x_positions = [positions[weight] for weight in lag_summary['temporal_weight']]
    axes[0, 2].plot(x_positions, estimate, color='black', marker='o')
    axes[0, 2].fill_between(
        x_positions,
        estimate - standard_error,
        estimate + standard_error,
        color='black',
        alpha=0.16,
    )
    axes[0, 2].set_xticks(list(positions.values()), [f'{weight:g}' for weight in temporal_weights])
    axes[0, 2].set_xlabel('Temporal weight')
    axes[0, 2].set_ylabel('Mean sampled episodic lag')
    axes[0, 2].set_title('C. Realized episodic retrieval')
    style_axis(axes[0, 2])
    plot_block_metric(
        axis=axes[1, 0],
        block_summary=block_summary_10,
        metric='frequency_score',
        temporal_weights=temporal_weights,
        temporal_weight_colors=temporal_weight_colors,
        ylabel='Frequency Score',
    )
    axes[1, 0].set_title('D. Rare vs frequent punishment (10-trial blocks)')
    axes[1, 1].set_visible(False)
    axes[1, 2].set_visible(False)
    figure.tight_layout()
    save_figure(figure=figure, savepath=visual_dir / 'figure_3_igt_frequency_and_retrieval_mechanism')


def run_analysis(
    *,
    input_dir: Path,
    output_dir: Path,
    n_boot: int,
    random_seed: int,
    save_figures: bool,
    deck_colors: dict[str, str] | None = None,
    configured_temporal_weight_colors: dict[str, str] | None = None,
    configured_contrast_colors: dict[str, str] | None = None,
) -> None:
    """
    Export temporal-valence IGT behavioral summaries.

    Args:
        input_dir (Path): Simulation cell directory.
        output_dir (Path): Analysis table and figure directory.
        n_boot (int): Number of agent-level bootstrap resamples.
        random_seed (int): Bootstrap and jitter seed.
        save_figures (bool): Whether to write figure files.
        deck_colors (dict[str, str] | None): Optional configured deck colors.
        configured_temporal_weight_colors (dict[str, str] | None): Optional configured weight colors.
        configured_contrast_colors (dict[str, str] | None): Optional configured contrast colors.
    """
    cells = load_simulation_cells(input_dir)
    trial_table = build_trial_table(cells)
    n_anneal_trials = cells[0]['config']['choice_policy']['n_anneal_trials']
    block_table, agent_table = build_agent_tables(
        trial_table=trial_table,
        n_anneal_trials=n_anneal_trials,
    )
    trial_summary = build_trial_probability_summary(
        trial_table=trial_table,
        n_boot=n_boot,
        random_seed=random_seed,
    )
    block_summary = build_block_behavior_summary(
        block_table=block_table,
        n_boot=n_boot,
        random_seed=random_seed,
    )
    trial_table_10 = add_igt_blocks(table=trial_table, block_size=10)
    block_table_10, _ = build_agent_tables(
        trial_table=trial_table_10,
        n_anneal_trials=n_anneal_trials,
    )
    block_summary_10 = build_block_behavior_summary(
        block_table=block_table_10,
        n_boot=n_boot,
        random_seed=random_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_obsolete_outputs(output_dir=output_dir)
    trial_summary.to_csv(output_dir / 'igt_trial_probability_summary.csv', index=False)
    block_summary.to_csv(output_dir / 'igt_block_behavior_summary.csv', index=False)
    agent_table.to_csv(output_dir / 'igt_agent_behavior_summary.csv', index=False)
    if not save_figures:
        return
    visual_dir = output_dir / 'visuals'
    visual_dir.mkdir(parents=True, exist_ok=True)
    temporal_weights = sorted(trial_table['temporal_weight'].unique())
    resolved_deck_colors = resolve_deck_colors(configured_colors=deck_colors or {})
    temporal_weight_colors = resolve_temporal_weight_colors(
        temporal_weights=temporal_weights,
        configured_colors=configured_temporal_weight_colors or {},
    )
    contrast_colors = resolve_contrast_colors(configured_colors=configured_contrast_colors or {})
    plot_figure_1(
        trial_summary=trial_summary,
        deck_colors=resolved_deck_colors,
        temporal_weight_colors=temporal_weight_colors,
        visual_dir=visual_dir,
    )
    plot_figure_2(
        block_summary=block_summary,
        block_summary_10=block_summary_10,
        agent_table=agent_table,
        temporal_weights=temporal_weights,
        temporal_weight_colors=temporal_weight_colors,
        n_boot=n_boot,
        random_seed=random_seed,
        visual_dir=visual_dir,
    )
    plot_figure_3(
        block_summary=block_summary,
        block_summary_10=block_summary_10,
        agent_table=agent_table,
        temporal_weights=temporal_weights,
        temporal_weight_colors=temporal_weight_colors,
        contrast_colors=contrast_colors,
        n_boot=n_boot,
        random_seed=random_seed,
        visual_dir=visual_dir,
    )


def main(args: argparse.Namespace) -> None:
    """
    Resolve paths and run temporal-valence IGT analysis.

    Args:
        args (argparse.Namespace): Parsed analysis arguments.
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
        configured_temporal_weight_colors=config.get('temporal_weight_colors'),
        configured_contrast_colors=config.get('contrast_colors'),
    )


if __name__ == '__main__':
    main(parse_args())
