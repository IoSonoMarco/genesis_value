"""
Add TD baselines to the completed temporal-only temperature figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path.cwd()))

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_hex, to_rgba
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from scripts import analysis_simulation_genesis_reward_bandit_task_temporal_only as budget_analysis
from scripts import analysis_simulation_genesis_reward_bandit_task_temporal_only_replacement as temperature_analysis
from scripts.simulation_genesis_reward_bandit_task_td import (
    DEFAULT_CONFIG_PATH,
    OUTPUT_SUBDIR,
    SOURCE_CONFIG_PATH,
    SOURCE_SUBDIR,
    format_float,
    load_yaml,
    validate_td_config,
)
from utils import filepaths
from utils.simulation_config import load_simulation_config


ANALYSIS_CONFIG_PATH = (
    filepaths.ROOT / 'configs' / 'analysis_simulation_genesis_reward_bandit_task_temporal_only.yaml'
)
TD_LINE_STYLES = ('--', ':', '-.')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse TD-enriched analysis arguments.

    Args:
        argv (list[str] | None): Optional command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Add TD models to temporal-only temperature figures.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--rebuild-cache', action='store_true')
    parser.add_argument('--no-save-figures', action='store_true')
    return parser.parse_args(argv)


def td_label(learning_rate: float) -> str:
    """
    Format a TD model label.

    Args:
        learning_rate (float): TD learning rate.

    Returns:
        str: Display label.
    """
    return f'TD α={format_float(learning_rate)}'


def resolve_td_colors(learning_rates: Sequence[float]) -> dict[str, str]:
    """
    Build an ordered black-to-gray TD palette.

    Args:
        learning_rates (Sequence[float]): Ordered TD learning rates.

    Returns:
        dict[str, str]: Colors keyed by TD display label.
    """
    if len(learning_rates) == 1:
        shades = [0.08]
    else:
        shades = np.linspace(0.08, 0.5, len(learning_rates))
    return {
        td_label(learning_rate): to_hex([shade, shade, shade])
        for learning_rate, shade in zip(learning_rates, shades)
    }


def load_td_records(
    *,
    td_dir: Path,
    learning_rates: Sequence[float],
    source_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """
    Load the configured TD records in learning-rate order.

    Args:
        td_dir (Path): Dedicated TD record directory.
        learning_rates (Sequence[float]): Ordered configured learning rates.
        source_config (dict[str, Any]): Existing temperature simulation configuration.

    Returns:
        tuple[list[dict[str, Any]], str]: Ordered records and common source signature.

    Raises:
        ValueError: If records are missing, duplicated, stale, or incomplete.
    """
    records_by_rate = {}
    for path in sorted(td_dir.glob('*.npy')):
        record = np.load(path, allow_pickle=True).item()
        config = record.get('config', {})
        if config.get('model') != 'TD_RL':
            continue
        rate = float(config['learning_rate'])
        if rate in records_by_rate:
            raise ValueError(f'Duplicate TD record for learning rate {rate:g}.')
        records_by_rate[rate] = record
    if set(records_by_rate) != set(learning_rates):
        raise ValueError('TD records do not exactly match the configured learning-rate grid.')

    signatures = set()
    ordered_records = []
    for learning_rate in learning_rates:
        record = records_by_rate[learning_rate]
        signatures.add(record['source_signature'])
        if len(record['environments']) != source_config['n_environments']:
            raise ValueError('A TD record has an incompatible environment count.')
        for environment in record['environments']:
            if len(environment['agent_runs']) != source_config['n_agent_simulations']:
                raise ValueError('A TD environment has an incompatible agent count.')
            if len(environment['schedule']) != 130:
                raise ValueError('A TD environment has an incompatible trial count.')
            if any(len(agent['simulation_log']) != 130 for agent in environment['agent_runs']):
                raise ValueError('A TD agent history is incomplete.')
        ordered_records.append(record)
    if len(signatures) != 1:
        raise ValueError('TD records were generated from different source experiments.')
    return ordered_records, signatures.pop()


def flatten_td_records(
    *,
    records: Sequence[dict[str, Any]],
    learning_rates: Sequence[float],
) -> pd.DataFrame:
    """
    Flatten TD nested histories into the existing trial-table contract.

    Args:
        records (Sequence[dict[str, Any]]): Ordered TD records.
        learning_rates (Sequence[float]): Ordered learning rates.

    Returns:
        pd.DataFrame: Trial-level TD table.
    """
    rows = []
    for record, learning_rate in zip(records, learning_rates):
        label = td_label(learning_rate)
        for environment in record['environments']:
            p_win_left = [trial['p_win_left'] for trial in environment['schedule']]
            for agent in environment['agent_runs']:
                frame = pd.DataFrame(agent['simulation_log']).drop(columns='metadata')
                frame['p_win_left'] = p_win_left
                frame['agent_seed'] = agent['agent_seed']
                frame['environment_seed'] = environment['environment_seed']
                frame['agent_simulation_id'] = agent['agent_simulation_id']
                frame['sampling_budget'] = label
                frame['sampling_temperature'] = np.nan
                frame['environment_id'] = environment['environment_id']
                rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def validate_base_table(
    *,
    frame: pd.DataFrame,
    source_config: dict[str, Any],
) -> None:
    """
    Validate the completed base analysis table before adding TD.

    Args:
        frame (pd.DataFrame): Existing temperature trial table.
        source_config (dict[str, Any]): Existing simulation configuration.

    Raises:
        ValueError: If the base table is incomplete or contains the wrong conditions.
    """
    expected_rows = (
        source_config['n_environments']
        * source_config['n_agent_simulations']
        * len(source_config['sampling_temperature'])
        * 130
    )
    if len(frame) != expected_rows:
        raise ValueError(f'Base full_table.csv has {len(frame)} rows; expected {expected_rows}.')
    temperatures = set(frame.sampling_temperature.dropna().astype(float).unique())
    if temperatures != set(map(float, source_config['sampling_temperature'])):
        raise ValueError('Base full_table.csv does not contain the configured temperature grid.')
    if frame.environment_id.nunique() != source_config['n_environments']:
        raise ValueError('Base full_table.csv has an incompatible environment count.')
    if frame.agent_simulation_id.nunique() != source_config['n_agent_simulations']:
        raise ValueError('Base full_table.csv has an incompatible agent count.')


def combined_cache_identity(
    *,
    source_signature: str,
    source_config: dict[str, Any],
    learning_rates: Sequence[float],
    base_table_path: Path,
) -> str:
    """
    Hash the inputs controlling the combined analysis cache.

    Args:
        source_signature (str): Common TD source identity.
        source_config (dict[str, Any]): Existing simulation configuration.
        learning_rates (Sequence[float]): Ordered TD learning rates.
        base_table_path (Path): Existing base trial table.

    Returns:
        str: SHA-256 cache identity.
    """
    stat = base_table_path.stat()
    payload = json.dumps({
        'source_signature': source_signature,
        'source_config': source_config,
        'learning_rates': list(learning_rates),
        'base_table_size': stat.st_size,
        'base_table_mtime_ns': stat.st_mtime_ns,
    }, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def build_combined_table(
    *,
    base_table_path: Path,
    td_records: Sequence[dict[str, Any]],
    learning_rates: Sequence[float],
    source_config: dict[str, Any],
    output_dir: Path,
    cache_identity: str,
    rebuild_cache: bool,
) -> pd.DataFrame:
    """
    Build or load the combined temperature-plus-TD trial table.

    Args:
        base_table_path (Path): Existing temperature trial table.
        td_records (Sequence[dict[str, Any]]): Ordered TD records.
        learning_rates (Sequence[float]): Ordered TD learning rates.
        source_config (dict[str, Any]): Existing simulation configuration.
        output_dir (Path): TD analysis output directory.
        cache_identity (str): Current source/cache identity.
        rebuild_cache (bool): Whether to rebuild a compatible cache.

    Returns:
        pd.DataFrame: Combined trial-level table.
    """
    table_path = output_dir / 'combined_full_table.csv'
    metadata_path = output_dir / 'cache_metadata.json'
    if table_path.exists() and metadata_path.exists() and not rebuild_cache:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('cache_identity') != cache_identity:
            raise ValueError('The TD analysis cache is stale; use --rebuild-cache.')
        return pd.read_csv(table_path, dtype={'sampling_budget': str})

    base = pd.read_csv(base_table_path, dtype={'sampling_budget': str})
    validate_base_table(frame=base, source_config=source_config)
    base['sampling_budget'] = base.sampling_temperature.map(temperature_analysis.format_temperature)
    td = flatten_td_records(records=td_records, learning_rates=learning_rates)
    probability_lookup = (
        base[['environment_id', 'trial_id', 'p_win_left']]
        .drop_duplicates(['environment_id', 'trial_id'])
        .set_index(['environment_id', 'trial_id'])
        .p_win_left
    )
    td_probability_index = pd.MultiIndex.from_frame(td[['environment_id', 'trial_id']])
    td['p_win_left'] = probability_lookup.reindex(td_probability_index).to_numpy()
    combined = pd.concat([base, td], ignore_index=True, sort=False)
    combined.to_csv(table_path, index=False)
    metadata_path.write_text(
        json.dumps({'cache_identity': cache_identity}, indent=2),
        encoding='utf-8',
    )
    return combined


def condition_labels(
    *,
    temperatures: Sequence[float],
    learning_rates: Sequence[float],
) -> tuple[list[str], list[str], list[str]]:
    """
    Build ordered temperature, TD, and combined labels.

    Args:
        temperatures (Sequence[float]): Ordered retrieval temperatures.
        learning_rates (Sequence[float]): Ordered TD learning rates.

    Returns:
        tuple[list[str], list[str], list[str]]: Temperature, TD, and combined labels.
    """
    temperature_labels = [temperature_analysis.format_temperature(value) for value in temperatures]
    td_labels = [td_label(value) for value in learning_rates]
    return temperature_labels, td_labels, [*temperature_labels, *td_labels]


def save_td_example_plot(
    *,
    savepath: Path,
    p_win_left: pd.Series,
    trial_ids: np.ndarray,
    curves: Sequence[Mapping[str, Any]],
    condition_order: Sequence[str],
    temperature_labels: Sequence[str],
    td_labels: Sequence[str],
    colors: Mapping[str, str],
    reward_ylim: tuple[float, float],
    save_figures: bool,
) -> None:
    """
    Save an exact temperature example layout with TD overlays.

    Args:
        savepath (Path): PDF destination.
        p_win_left (pd.Series): Environment LEFT win probabilities.
        trial_ids (np.ndarray): Trial identifiers.
        curves (Sequence[Mapping[str, Any]]): Ordered condition curves.
        condition_order (Sequence[str]): Ordered condition labels.
        temperature_labels (Sequence[str]): Temperature labels.
        td_labels (Sequence[str]): TD labels.
        colors (Mapping[str, str]): Condition colors.
        reward_ylim (tuple[float, float]): Shared reward-axis range.
        save_figures (bool): Whether to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    budget_analysis.plot_example_environment_panel(
        axes=axes,
        p_win_left=p_win_left,
        trial_ids=trial_ids,
        curves=curves,
        budget_colors=colors,
        reward_ylim=reward_ylim,
    )
    for axis in axes:
        for line, condition in zip(axis.lines, condition_order):
            if condition in td_labels:
                td_index = td_labels.index(condition)
                line.set_linestyle(TD_LINE_STYLES[td_index % len(TD_LINE_STYLES)])
                line.set_linewidth(2.4)
    handles = [
        Line2D([], [], color=colors[label], linewidth=2, label=f'T={label}')
        for label in temperature_labels
    ] + [
        Line2D(
            [],
            [],
            color=colors[label],
            linestyle=TD_LINE_STYLES[index % len(TD_LINE_STYLES)],
            linewidth=2.4,
            label=label,
        )
        for index, label in enumerate(td_labels)
    ]
    fig.legend(
        handles=handles,
        title='Model condition',
        loc='upper center',
        bbox_to_anchor=(0.5, 1.13),
        ncol=min(4, len(handles)),
        frameon=False,
    )
    fig.tight_layout(pad=3, rect=[0, 0.08, 1, 1])
    if save_figures:
        fig.savefig(savepath, dpi=800, bbox_inches='tight')
    plt.close(fig)


def plot_td_correlation_top_axis(
    *,
    axis: Axes,
    td_stats: pd.DataFrame,
    td_labels: Sequence[str],
    td_colors: Mapping[str, str],
) -> Axes:
    """
    Add categorical TD correlations on a shared-y top x-axis.

    Args:
        axis (Axes): Bottom temperature-correlation axis.
        td_stats (pd.DataFrame): TD correlation rows.
        td_labels (Sequence[str]): Ordered TD labels.
        td_colors (Mapping[str, str]): TD colors.

    Returns:
        Axes: Configured top TD axis.
    """
    top_axis = axis.twiny()
    positions = np.arange(len(td_labels))
    top_axis.set_xlim(-0.5, len(td_labels) - 0.5)
    top_axis.set_xticks(positions, td_labels)
    top_axis.set_xlabel('TD learning-rate model', fontsize=12, labelpad=8)
    top_axis.spines['top'].set_visible(True)
    top_axis.spines['bottom'].set_visible(False)
    top_axis.spines['left'].set_visible(False)
    top_axis.spines['right'].set_visible(False)
    top_axis.tick_params(axis='x', size=4)

    for position, label in enumerate(td_labels):
        row = td_stats[td_stats.sampling_budget == label].iloc[0]
        y_value = row.pearson_r
        ci_low = row.pearson_ci_low
        ci_high = row.pearson_ci_high
        top_axis.scatter(position, y_value, color=td_colors[label], s=48, zorder=4)
        if np.isfinite(ci_low) and np.isfinite(ci_high):
            top_axis.errorbar(
                position,
                y_value,
                yerr=[[y_value - ci_low], [ci_high - y_value]],
                color=td_colors[label],
                ecolor=to_rgba(td_colors[label], alpha=0.45),
                elinewidth=7,
                capsize=0,
                zorder=3,
            )
    return top_axis


def save_td_correlation_plot(
    *,
    savepath: Path,
    correlation_stats: pd.DataFrame,
    temperature_labels: Sequence[str],
    td_labels: Sequence[str],
    temperature_colors: Mapping[str, str],
    td_colors: Mapping[str, str],
    save_figures: bool,
) -> None:
    """
    Save the temperature correlation plot with a categorical TD top axis.

    Args:
        savepath (Path): PDF destination.
        correlation_stats (pd.DataFrame): Combined correlation statistics.
        temperature_labels (Sequence[str]): Ordered temperature labels.
        td_labels (Sequence[str]): Ordered TD labels.
        temperature_colors (Mapping[str, str]): Temperature colors.
        td_colors (Mapping[str, str]): TD colors.
        save_figures (bool): Whether to save the figure.
    """
    temperature_stats = correlation_stats[
        correlation_stats.sampling_budget.isin(temperature_labels)
    ].set_index('sampling_budget').loc[list(temperature_labels)].reset_index()
    td_stats = correlation_stats[
        correlation_stats.sampling_budget.isin(td_labels)
    ].set_index('sampling_budget').loc[list(td_labels)].reset_index()
    fig, axis = plt.subplots(1, 1, figsize=(4.2, 6.0))
    budget_analysis.plot_reward_correlation_panel(
        ax=axis,
        df_reward_corr_stats=temperature_stats,
        budget_colors=temperature_colors,
    )
    axis.set_xlabel('Softmax temperature', fontsize=12)
    plot_td_correlation_top_axis(
        axis=axis,
        td_stats=td_stats,
        td_labels=td_labels,
        td_colors=td_colors,
    )
    y_candidates = correlation_stats[['pearson_r', 'pearson_ci_low']].to_numpy(dtype=float)
    finite = y_candidates[np.isfinite(y_candidates)]
    axis.set_ylim(max(-1.0, finite.min() - 0.05), 1.0)
    fig.tight_layout()
    if save_figures:
        fig.savefig(savepath, dpi=800, bbox_inches='tight')
    plt.close(fig)


def run_analysis(args: argparse.Namespace) -> None:
    """
    Produce the three TD-enriched temperature figures.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
    """
    td_config = load_yaml(args.config)
    learning_rates = validate_td_config(td_config)
    source_config = load_simulation_config(SOURCE_CONFIG_PATH)
    analysis_config = budget_analysis.load_analysis_config(ANALYSIS_CONFIG_PATH)
    save_figures = bool(analysis_config['save_figures']) and not args.no_save_figures
    td_dir = filepaths.EXPERIMENTS_DIR / source_config['experiment_name'] / OUTPUT_SUBDIR
    td_records, source_signature = load_td_records(
        td_dir=td_dir,
        learning_rates=learning_rates,
        source_config=source_config,
    )
    base_output_dir = (
        filepaths.OUTPUT_DIR
        / analysis_config['output_dir_name']
        / SOURCE_SUBDIR
    )
    base_table_path = base_output_dir / 'full_table.csv'
    if not base_table_path.is_file():
        raise FileNotFoundError('Run the original temperature analysis before the TD-enriched analysis.')
    output_dir = base_output_dir / OUTPUT_SUBDIR
    visual_dir = base_output_dir / 'visuals' / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    cache_identity = combined_cache_identity(
        source_signature=source_signature,
        source_config=source_config,
        learning_rates=learning_rates,
        base_table_path=base_table_path,
    )
    combined = build_combined_table(
        base_table_path=base_table_path,
        td_records=td_records,
        learning_rates=learning_rates,
        source_config=source_config,
        output_dir=output_dir,
        cache_identity=cache_identity,
        rebuild_cache=args.rebuild_cache,
    )
    env_score = budget_analysis.build_env_score_table(
        df_full=combined,
        savepath=output_dir / 'env_score-vs-policy-table.csv',
        rebuild_cache=args.rebuild_cache,
    )
    reward_analysis = budget_analysis.build_reward_analysis_table(env_score)
    temperature_labels, td_labels, condition_order = condition_labels(
        temperatures=source_config['sampling_temperature'],
        learning_rates=learning_rates,
    )
    temperature_colors = budget_analysis.resolve_budget_colors(
        budget_order=temperature_labels,
        configured_colors={},
    )
    td_colors = resolve_td_colors(learning_rates)
    colors = {**temperature_colors, **td_colors}

    env_scores = reward_analysis[['env_id', 'env_score']].drop_duplicates('env_id')
    example_curves = {}
    selections = {}
    for label, example_config in analysis_config['example_environments'].items():
        environment_id = temperature_analysis.select_replacement_example_environment(
            label=label,
            df_env_scores=env_scores,
            score_window=example_config['score_window'],
            seed=example_config['seed'],
        )
        selections[label] = environment_id
        example_curves[label] = budget_analysis.compute_example_environment_curves(
            df_full=combined,
            env_id=environment_id,
            budget_order=condition_order,
        )
    reward_ylim = temperature_analysis.resolve_shared_reward_ylim(
        example_curves=[curves[2] for curves in example_curves.values()],
    )
    for label, curves in example_curves.items():
        save_td_example_plot(
            savepath=visual_dir / f'sampling-temperature_rewards_env-score-{label}.pdf',
            p_win_left=curves[0],
            trial_ids=curves[1],
            curves=curves[2],
            condition_order=condition_order,
            temperature_labels=temperature_labels,
            td_labels=td_labels,
            colors=colors,
            reward_ylim=reward_ylim,
            save_figures=save_figures,
        )

    correlation_stats = budget_analysis.compute_reward_correlation_stats(
        df_reward_analysis=reward_analysis,
        budget_order=condition_order,
    )
    correlation_stats.to_csv(output_dir / 'reward_envscore_stats_by_temperature_and_td.csv', index=False)
    save_td_correlation_plot(
        savepath=visual_dir / 'reward_envscore_correlation_by_temperature.pdf',
        correlation_stats=correlation_stats,
        temperature_labels=temperature_labels,
        td_labels=td_labels,
        temperature_colors=temperature_colors,
        td_colors=td_colors,
        save_figures=save_figures,
    )
    pd.DataFrame([
        {'example': label, 'environment_id': environment_id}
        for label, environment_id in selections.items()
    ]).to_csv(output_dir / 'example_environment_selection.csv', index=False)
    np.save(
        output_dir / 'analysis_metadata.npy',
        {
            'source_signature': source_signature,
            'source_config': source_config,
            'td_config': td_config,
            'condition_order': condition_order,
            'cache_identity': cache_identity,
        },
        allow_pickle=True,
    )
    plt.close('all')
    print(f'Saved TD-enriched figures to {visual_dir}')


def main(argv: list[str] | None = None) -> None:
    """
    Run TD-enriched analysis from command-line arguments.

    Args:
        argv (list[str] | None): Optional command-line arguments.
    """
    run_analysis(parse_args(argv))


if __name__ == '__main__':
    main()
