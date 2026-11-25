"""
Analyze replacement-sampled GENESIS reminder-biased retrieval by temperature.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path.cwd()))

from scripts import analysis_simulation_genesis_reward_bandit_task as budget_analysis
from utils import filepaths
from utils.simulation_config import get_replacement_output_dir, load_simulation_config


DEFAULT_ANALYSIS_CONFIG_PATH = (
    filepaths.ROOT / 'configs' / 'analysis_simulation_genesis_reward_bandit_task.yaml'
)
GROUP_COLUMNS = [
    'environment_id',
    'sampling_temperature',
    'temporal_weight',
    'sample_replacement',
]
DEFAULT_PLOT_AESTHETICS = {
    'sampling_temperature_colors': {
        '0.1': '#440154',
        '0.25': '#3B528B',
        '0.5': '#21918C',
        '1.0': '#5EC962',
        '2.0': '#FDE725',
    },
    'nonsignificant_color': '#FF0000',
    'nonsignificant_hatch': '///',
    'temporal_weight_cmap': 'viridis',
    'plot_temporal_weights': [0.0, 1.0],
    'panel_sizes': {
        '1b': [7.0, 5.0],
        '2': [12.0, 5.0],
        '3a': [4.0, 5.5],
        '3b': [4.0, 5.5],
    },
    'choice_marker_size': 7,
    'choice_errorbar_linewidth': 7,
    'choice_errorbar_alpha': 0.45,
    'choice_jitter': 0.035,
}


def format_temperature(temperature: float) -> str:
    """
    Format a softmax temperature for plot labels.

    Args:
        temperature (float): Softmax temperature.

    Returns:
        str: Display label.
    """
    return f'{temperature:.2g}' if temperature < 1 else f'{temperature:.1f}'


def resolve_analysis_config(config_path: Path) -> dict[str, Any]:
    """
    Load replacement analysis controls and linked simulation settings.

    Args:
        config_path (Path): Analysis YAML configuration path.

    Returns:
        dict[str, Any]: Analysis configuration enriched with simulation settings.
    """
    analysis_config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    simulation_config_path = Path(analysis_config['simulation_config_path'])
    if not simulation_config_path.is_absolute():
        simulation_config_path = filepaths.ROOT / simulation_config_path

    simulation_config = load_simulation_config(simulation_config_path)
    assert simulation_config['sample_replacement']
    temperatures = [float(value) for value in simulation_config['sampling_temperature']]
    plot_temperatures = [
        float(value)
        for value in analysis_config.get('plot_sampling_temperatures', temperatures)
    ]
    assert plot_temperatures
    assert set(plot_temperatures).issubset(temperatures)

    return {
        **analysis_config,
        'experiment_name': simulation_config['experiment_name'],
        'n_environments': simulation_config['n_environments'],
        'n_agent_simulations': simulation_config['n_agent_simulations'],
        'replacement_sampling_budget': simulation_config['replacement_sampling_budget'],
        'sample_replacement': True,
        'plot_sampling_temperatures': plot_temperatures,
        'grid': {
            'sampling_temperature': temperatures,
            'temporal_weight': [float(value) for value in simulation_config['grid']['temporal_weight']],
            'sample_replacement': [True],
        },
    }


def load_experiment_tables(
    *,
    experiment_dir: Path,
    expected_grid: dict[str, list[Any]],
    n_environments: int,
    n_agent_simulations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load replacement simulation outputs into reminder and reward tables.

    Args:
        experiment_dir (Path): Directory containing one NumPy file per grid cell.
        expected_grid (dict[str, list[Any]]): Expected temperature-by-weight grid.
        n_environments (int): Expected environment count per cell.
        n_agent_simulations (int): Expected agent count per environment.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Reminder-level and agent-reward tables.
    """
    reminder_rows = []
    reward_rows = []
    observed_cells = []

    for savepath in sorted(experiment_dir.glob('*.npy')):
        result = np.load(savepath, allow_pickle=True).item()
        config = result['config']
        temperature = float(config['sampling_temperature'])
        temporal_weight = float(config['temporal_weight'])
        sample_replacement = bool(config['sample_replacement'])
        observed_cells.append((temperature, temporal_weight, sample_replacement))
        environments = result['environments']
        assert len(environments) == n_environments

        for environment in environments:
            agent_runs = environment['agent_runs']
            assert len(agent_runs) == n_agent_simulations
            for agent_run in agent_runs:
                simulation_log = agent_run['simulation_log']
                identifiers = {
                    'environment_id': environment['environment_id'],
                    'agent_simulation_id': agent_run['agent_simulation_id'],
                    'sampling_budget': str(config['sampling_budget']),
                    'sampling_temperature': temperature,
                    'temporal_weight': temporal_weight,
                    'sample_replacement': sample_replacement,
                }
                reminder_log = [record for record in simulation_log if record['is_reminder_trial']]
                for record in reminder_log:
                    reminder_rows.append({
                        **identifiers,
                        **{key: value for key, value in record.items() if key != 'metadata'},
                    })
                reward_rows.append({
                    **identifiers,
                    'total_reward': float(np.sum([record['reward'] for record in simulation_log])),
                })

    expected_cells = {
        (float(temperature), float(weight), bool(replacement))
        for temperature in expected_grid['sampling_temperature']
        for weight in expected_grid['temporal_weight']
        for replacement in expected_grid['sample_replacement']
    }
    assert len(observed_cells) == len(expected_cells)
    assert set(observed_cells) == expected_cells

    return pd.DataFrame(reminder_rows), pd.DataFrame(reward_rows)


def compute_environment_lag_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate causal-chain lag slopes within every temperature cell.

    Args:
        df (pd.DataFrame): Reminder table with derived metrics.

    Returns:
        pd.DataFrame: Environment-by-cell lag slopes.
    """
    outcome_columns = {
        'accessibility_lag_slope': 'log_accessibility_enrichment',
        'capture_lag_slope': 'reminder_target_sampled',
        'value_alignment_lag_slope': 'signed_value_alignment',
        'policy_alignment_lag_slope': 'signed_policy_alignment',
    }
    rows = []
    for identifiers, group in df.groupby(GROUP_COLUMNS, observed=True):
        row = dict(zip(GROUP_COLUMNS, identifiers))
        for slope_name, outcome_name in outcome_columns.items():
            row[slope_name] = budget_analysis.fit_slope(
                x=group['lag_10'],
                y=group[outcome_name].astype(float),
            )
        rows.append(row)

    return pd.DataFrame(rows)


def compute_environment_cell_table(
    reminder_table: pd.DataFrame,
    reward_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate causal-chain outcomes to independent environment-level cells.

    Args:
        reminder_table (pd.DataFrame): Reminder table with derived metrics.
        reward_table (pd.DataFrame): Agent-level total rewards.

    Returns:
        pd.DataFrame: Environment-by-cell summary table.
    """
    rows = []
    for identifiers, group in reminder_table.groupby(GROUP_COLUMNS, observed=True):
        wins = group[group['reminder_target_reward'] > 0]
        losses = group[group['reminder_target_reward'] < 0]
        rows.append({
            **dict(zip(GROUP_COLUMNS, identifiers)),
            'accessibility_mean': float(group['log_accessibility_enrichment'].mean()),
            'capture_rate': float(group['reminder_target_sampled'].mean()),
            'value_alignment_mean': float(group['signed_value_alignment'].mean()),
            'policy_alignment_mean': float(group['signed_policy_alignment'].mean()),
            'repeat_win': float(wins['repeated_target_action'].mean()),
            'repeat_loss': float(losses['repeated_target_action'].mean()),
            'choice_interaction': float(
                wins['repeated_target_action'].mean() - losses['repeated_target_action'].mean()
            ),
        })
    environment_cells = pd.DataFrame(rows)
    environment_rewards = (
        reward_table
        .groupby(GROUP_COLUMNS, observed=True)['total_reward']
        .mean()
        .rename('total_reward')
        .reset_index()
    )
    return environment_cells.merge(environment_rewards, on=GROUP_COLUMNS, how='left')


def compute_temperature_contrasts(
    slope_table: pd.DataFrame,
    environment_cells: pd.DataFrame,
    *,
    grid: dict[str, list[Any]],
    n_boot: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Compute temporal-weight and temperature contrasts.

    Args:
        slope_table (pd.DataFrame): Environment-level lag slopes.
        environment_cells (pd.DataFrame): Environment-level causal-chain outcomes.
        grid (dict[str, list[Any]]): Replacement simulation grid.
        n_boot (int): Number of bootstrap resamples.
        random_state (int): Bootstrap seed.

    Returns:
        pd.DataFrame: Planned contrast estimates and confidence intervals.
    """
    rows = []
    temperatures = [float(value) for value in grid['sampling_temperature']]
    temporal_weights = [float(value) for value in grid['temporal_weight']]
    metric_tables = [
        (
            slope_table,
            [
                'accessibility_lag_slope',
                'capture_lag_slope',
                'value_alignment_lag_slope',
                'policy_alignment_lag_slope',
            ],
        ),
        (environment_cells, ['value_alignment_mean', 'policy_alignment_mean', 'choice_interaction']),
    ]
    for table, metrics in metric_tables:
        for metric in metrics:
            collapsed = (
                table
                .groupby(['environment_id', 'temporal_weight'], observed=True)[metric]
                .mean()
                .unstack('temporal_weight')
            )
            for first_index, first_weight in enumerate(temporal_weights[:-1]):
                for second_weight in temporal_weights[first_index + 1:]:
                    rows.append(budget_analysis.summarize_contrast(
                        contrast=f'{metric}_weight-{first_weight}-minus-{second_weight}',
                        values=collapsed[first_weight] - collapsed[second_weight],
                        n_boot=n_boot,
                        random_state=random_state + len(rows),
                    ))
                    effects = {}
                    for temperature in temperatures:
                        temperature_table = table[table['sampling_temperature'] == temperature].pivot(
                            index='environment_id', columns='temporal_weight', values=metric
                        )
                        effects[temperature] = temperature_table[first_weight] - temperature_table[second_weight]
                        rows.append(budget_analysis.summarize_contrast(
                            contrast=(
                                f'{metric}_weight-{first_weight}-minus-{second_weight}_'
                                f'temperature-{format_temperature(temperature)}'
                            ),
                            values=effects[temperature],
                            n_boot=n_boot,
                            random_state=random_state + len(rows),
                        ))
                    for first_index, first_temperature in enumerate(temperatures[:-1]):
                        for second_temperature in temperatures[first_index + 1:]:
                            rows.append(budget_analysis.summarize_contrast(
                                contrast=(
                                    f'{metric.removesuffix("_mean")}_temperature-'
                                    f'{format_temperature(first_temperature)}-'
                                    f'minus-{format_temperature(second_temperature)}_'
                                    f'weight-{first_weight}-minus-{second_weight}'
                                ),
                                values=effects[first_temperature] - effects[second_temperature],
                                n_boot=n_boot,
                                random_state=random_state + len(rows),
                            ))
    return pd.DataFrame(rows)


def compute_lag_band_stats(
    reminder_table: pd.DataFrame,
    *,
    n_boot: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Summarize causal-chain metrics in prespecified reminder-lag bands.

    Args:
        reminder_table (pd.DataFrame): Reminder records with derived metrics.
        n_boot (int): Number of bootstrap resamples.
        random_state (int): Bootstrap seed.

    Returns:
        pd.DataFrame: Lag-band estimates and confidence intervals.
    """
    metric_columns = [
        'log_accessibility_enrichment',
        'normalized_target_rank',
        'reminder_target_sampled',
        'signed_value_alignment',
        'signed_policy_alignment',
        'repeated_target_action',
    ]
    environment_bands = (
        reminder_table
        .groupby([*GROUP_COLUMNS, 'lag_band'], observed=True)[metric_columns]
        .mean()
        .reset_index()
    )
    return budget_analysis.summarize_grouped_metric(
        environment_bands,
        group_columns=['sampling_temperature', 'temporal_weight', 'sample_replacement', 'lag_band'],
        metric_columns=metric_columns,
        n_boot=n_boot,
        random_state=random_state,
    )


def compute_continuous_lag_stats(
    reminder_table: pd.DataFrame,
    *,
    n_boot: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Summarize target retrieval advantage at every observed reminder lag.

    Args:
        reminder_table (pd.DataFrame): Reminder records with derived metrics.
        n_boot (int): Number of bootstrap resamples.
        random_state (int): Bootstrap seed.

    Returns:
        pd.DataFrame: Exact-lag target retrieval summaries.
    """
    environment_lags = (
        reminder_table
        .groupby([*GROUP_COLUMNS, 'reminder_lag'], observed=True)['log_accessibility_enrichment']
        .mean()
        .reset_index()
    )
    return budget_analysis.summarize_grouped_metric(
        environment_lags,
        group_columns=['sampling_temperature', 'temporal_weight', 'sample_replacement', 'reminder_lag'],
        metric_columns=['log_accessibility_enrichment'],
        n_boot=n_boot,
        random_state=random_state,
    )


def decaying_exponential(
    temporal_weight: np.ndarray,
    baseline: float,
    amplitude: float,
    rate: float,
) -> np.ndarray:
    """
    Evaluate a decaying exponential value-reconstruction curve.

    Args:
        temporal_weight (np.ndarray): Temporal-weight coordinates.
        baseline (float): Asymptotic value alignment.
        amplitude (float): Initial alignment above the asymptote.
        rate (float): Decay rate.

    Returns:
        np.ndarray: Value-alignment estimates.
    """
    return baseline + amplitude * np.exp(-rate * temporal_weight)


def fit_decaying_exponential(
    *,
    temporal_weights: np.ndarray,
    estimates: np.ndarray,
) -> np.ndarray:
    """
    Fit a constrained decaying exponential to value-alignment estimates.

    Args:
        temporal_weights (np.ndarray): Temporal-weight coordinates.
        estimates (np.ndarray): Value-alignment estimates at each coordinate.

    Returns:
        np.ndarray: Baseline, amplitude, and rate parameters.
    """
    parameters, _ = curve_fit(
        decaying_exponential,
        temporal_weights,
        estimates,
        p0=[estimates[-1], estimates[0] - estimates[-1], 1.0],
        bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf]),
        maxfev=10000,
    )

    return parameters


def compute_exponential_curves(
    *,
    environment_cells: pd.DataFrame,
    plot_temperatures: Sequence[float],
    temporal_weights: Sequence[float],
) -> pd.DataFrame:
    """
    Fit decaying-exponential value-alignment curves.

    Args:
        environment_cells (pd.DataFrame): Environment-by-grid-cell causal-chain outcomes.
        plot_temperatures (Sequence[float]): Temperatures included in the fitted curves.
        temporal_weights (Sequence[float]): Temporal-weight grid used for curve fitting.

    Returns:
        pd.DataFrame: Dense fitted curves.
    """
    weights = np.asarray(temporal_weights, dtype=float)
    curve_weights = np.linspace(weights.min(), weights.max(), num=100)
    curves = []

    for temperature in plot_temperatures:
        environment_values = (
            environment_cells[environment_cells['sampling_temperature'] == temperature]
            .pivot(
                index='environment_id',
                columns='temporal_weight',
                values='value_alignment_mean',
            )
            .reindex(columns=weights)
            .to_numpy()
        )
        parameters = fit_decaying_exponential(
            temporal_weights=weights,
            estimates=environment_values.mean(axis=0),
        )
        curves.append(pd.DataFrame({
            'sampling_temperature': temperature,
            'temporal_weight': curve_weights,
            'estimate': decaying_exponential(curve_weights, *parameters),
        }))

    return pd.concat(curves, ignore_index=True)


def plot_causal_chain_panels(
    *,
    continuous_lag_stats: pd.DataFrame,
    cell_stats: pd.DataFrame,
    environment_cells: pd.DataFrame,
    grid: dict[str, list[Any]],
    plot_aesthetics: dict[str, Any],
    plot_temperatures: Sequence[float],
    n_boot: int,
    random_state: int,
    visual_dir: Path,
) -> None:
    """
    Export the selected replacement causal-chain panels.

    Args:
        continuous_lag_stats (pd.DataFrame): Exact-lag accessibility summaries.
        cell_stats (pd.DataFrame): Grid-cell causal-chain summaries.
        environment_cells (pd.DataFrame): Environment-by-grid-cell causal-chain outcomes.
        grid (dict[str, list[Any]]): Replacement simulation grid.
        plot_aesthetics (dict[str, Any]): Figure styling controls.
        plot_temperatures (Sequence[float]): Temperature levels included in figures.
        n_boot (int): Number of environment bootstrap resamples.
        random_state (int): Bootstrap seed.
        visual_dir (Path): Directory for standalone panel PDFs.
    """
    aesthetics = {**DEFAULT_PLOT_AESTHETICS, **plot_aesthetics}
    temporal_weights = [float(value) for value in grid['temporal_weight']]
    panel_temporal_weights = budget_analysis.get_plot_temporal_weights(
        grid={'temporal_weight': temporal_weights},
        plot_aesthetics=aesthetics,
    )
    temperature_colors = {
        temperature: aesthetics['sampling_temperature_colors'][format_temperature(temperature)]
        for temperature in plot_temperatures
    }
    temperature_jitters = budget_analysis.get_sampling_budget_jitters(
        sampling_budgets=[format_temperature(value) for value in plot_temperatures],
        jitter=aesthetics['choice_jitter'],
    )
    temporal_colors = budget_analysis.get_temporal_weight_colors(
        temporal_weights=temporal_weights,
        cmap_name=aesthetics['temporal_weight_cmap'],
    )
    reference_temperature = plot_temperatures[0]

    fig, axis = plt.subplots(figsize=aesthetics['panel_sizes']['1b'])
    for temporal_weight, temporal_color in zip(temporal_weights, temporal_colors):
        group = continuous_lag_stats[
            (continuous_lag_stats['sampling_temperature'] == reference_temperature)
            & (continuous_lag_stats['temporal_weight'] == temporal_weight)
        ].sort_values('reminder_lag')
        axis.plot(group['reminder_lag'], group['estimate'], color=temporal_color, label=f'w={temporal_weight}')
        axis.fill_between(group['reminder_lag'], group['ci_low'], group['ci_high'], color=temporal_color, alpha=0.08)
    axis.axhline(0.0, color='black', linewidth=1, alpha=0.4)
    axis.set_xlabel('Reminder lag (trials)')
    axis.set_ylabel('Target retrieval advantage vs. uniform')
    axis.set_title('1b. Target retrieval across lag')
    axis.legend(frameon=False)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(visual_dir / 'reminder_retrieval_causal_chain_panel_1b.pdf', bbox_inches='tight')
    plt.close(fig)

    value_stats = cell_stats[cell_stats['metric'] == 'value_alignment_mean']
    exponential_curves = compute_exponential_curves(
        environment_cells=environment_cells,
        plot_temperatures=plot_temperatures,
        temporal_weights=temporal_weights,
    )
    fig, axes = plt.subplots(ncols=2, figsize=aesthetics['panel_sizes']['2'], sharey=True)
    connected_axis, fitted_axis = axes
    for temperature in plot_temperatures:
        group = (
            value_stats[value_stats['sampling_temperature'] == temperature]
            .set_index('temporal_weight')
            .reindex(temporal_weights)
        )
        estimates = group['estimate'].to_numpy()
        temperature_jitter = temperature_jitters[format_temperature(temperature)]
        plotted_weights = np.asarray(temporal_weights) + temperature_jitter
        connected_axis.plot(
            plotted_weights,
            estimates,
            color=temperature_colors[temperature],
            linewidth=2,
        )
        connected_axis.errorbar(
            plotted_weights,
            estimates,
            yerr=[estimates - group['ci_low'].to_numpy(), group['ci_high'].to_numpy() - estimates],
            color=temperature_colors[temperature],
            marker='o',
            markersize=aesthetics['choice_marker_size'],
            linestyle='none',
            ecolor=to_rgba(temperature_colors[temperature], alpha=aesthetics['choice_errorbar_alpha']),
            elinewidth=aesthetics['choice_errorbar_linewidth'],
            capsize=0,
            label=f'T={format_temperature(temperature)}',
        )
        fitted_axis.errorbar(
            plotted_weights,
            estimates,
            yerr=[estimates - group['ci_low'].to_numpy(), group['ci_high'].to_numpy() - estimates],
            color=temperature_colors[temperature],
            marker='o',
            markersize=aesthetics['choice_marker_size'],
            linestyle='none',
            ecolor=to_rgba(temperature_colors[temperature], alpha=aesthetics['choice_errorbar_alpha']),
            elinewidth=aesthetics['choice_errorbar_linewidth'],
            capsize=0,
            label=f'T={format_temperature(temperature)}',
        )
        curve = exponential_curves[exponential_curves['sampling_temperature'] == temperature]
        fitted_axis.plot(
            curve['temporal_weight'] + temperature_jitter,
            curve['estimate'],
            color=temperature_colors[temperature],
            linewidth=2,
        )
    for axis, title in [
        (connected_axis, '2a. Value reconstruction'),
        (fitted_axis, '2b. Exponential reconstruction'),
    ]:
        axis.set_xlabel('Temporal weight')
        axis.set_xticks(temporal_weights, [f'{weight:g}' for weight in temporal_weights])
        axis.set_title(title)
        axis.legend(frameon=False)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    connected_axis.set_ylabel('Signed value alignment')
    fig.tight_layout()
    fig.savefig(visual_dir / 'reminder_retrieval_causal_chain_panel_2.pdf', bbox_inches='tight')
    plt.close(fig)

    for panel_name, metric, title, direction in [
        ('3a', 'repeat_win', '3a. Repeat after win', 1.0),
        ('3b', 'repeat_loss', '3b. Repeat after loss', -1.0),
    ]:
        fig, axis = plt.subplots(figsize=aesthetics['panel_sizes'][panel_name])
        for temperature in plot_temperatures:
            group = cell_stats[
                (cell_stats['sampling_temperature'] == temperature)
                & (cell_stats['metric'] == metric)
            ].set_index('temporal_weight').reindex(panel_temporal_weights)
            estimates = group['estimate'].to_numpy() - 0.5
            ci_low = group['ci_low'].to_numpy() - 0.5
            ci_high = group['ci_high'].to_numpy() - 0.5
            positions = np.arange(len(panel_temporal_weights)) + temperature_jitters[format_temperature(temperature)]
            axis.errorbar(
                estimates,
                positions,
                xerr=[estimates - ci_low, ci_high - estimates],
                color=temperature_colors[temperature],
                marker='o',
                markersize=aesthetics['choice_marker_size'],
                linestyle='--',
                ecolor=to_rgba(temperature_colors[temperature], alpha=aesthetics['choice_errorbar_alpha']),
                elinewidth=aesthetics['choice_errorbar_linewidth'],
                capsize=0,
                linewidth=1,
                label=f'T={format_temperature(temperature)}',
            )
            for position, low, high in zip(positions, ci_low, ci_high):
                if low < 0.0 < high:
                    edge = high if direction > 0.0 else low
                    axis.add_patch(Rectangle(
                        (min(0.0, edge), position - aesthetics['choice_jitter'] / 2),
                        abs(edge),
                        aesthetics['choice_jitter'],
                        facecolor='none',
                        edgecolor=aesthetics['nonsignificant_color'],
                        hatch=aesthetics['nonsignificant_hatch'],
                        linewidth=0.5,
                        zorder=4,
                    ))
        axis.axvline(0.0, color='dimgray', linestyle=':', linewidth=1)
        axis.set_ylim(len(panel_temporal_weights) - 0.5, -0.5)
        axis.set_xlabel('Δ P(repeat target action)')
        axis.yaxis.set_visible(False)
        axis.set_title(title)
        axis.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=temperature_colors[value],
                    marker='o',
                    label=f'T={format_temperature(value)}',
                )
                for value in plot_temperatures
            ],
            frameon=False,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.25),
            ncol=2,
        )
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['left'].set_visible(False)
        fig.tight_layout()
        fig.savefig(visual_dir / f'reminder_retrieval_causal_chain_panel_{panel_name}.pdf', bbox_inches='tight')
        plt.close(fig)


def run_analysis_tables(
    *,
    reminder_table: pd.DataFrame,
    reward_table: pd.DataFrame,
    output_dir: Path,
    n_boot: int,
    random_state: int,
    grid: dict[str, list[Any]],
    plot_aesthetics: dict[str, Any],
    plot_temperatures: Sequence[float],
) -> None:
    """
    Compute, export, and visualize replacement causal-chain analysis tables.

    Args:
        reminder_table (pd.DataFrame): Raw reminder-level records.
        reward_table (pd.DataFrame): Agent-level total rewards.
        output_dir (Path): Analysis output directory.
        n_boot (int): Number of environment bootstrap resamples.
        random_state (int): Bootstrap seed.
        grid (dict[str, list[Any]]): Replacement simulation grid.
        plot_aesthetics (dict[str, Any]): Figure styling controls.
        plot_temperatures (Sequence[float]): Temperatures included in figures.
    """
    visual_dir = output_dir / 'visuals'
    visual_dir.mkdir(parents=True, exist_ok=True)
    reminder_metrics = budget_analysis.add_reminder_metrics(reminder_table)
    lag_slopes = compute_environment_lag_slopes(reminder_metrics)
    environment_cells = compute_environment_cell_table(reminder_metrics, reward_table)
    continuous_lag_stats = compute_continuous_lag_stats(reminder_metrics, n_boot=n_boot, random_state=random_state)
    lag_band_stats = compute_lag_band_stats(reminder_metrics, n_boot=n_boot, random_state=random_state)
    planned_contrasts = compute_temperature_contrasts(
        lag_slopes,
        environment_cells,
        grid=grid,
        n_boot=n_boot,
        random_state=random_state,
    )
    cell_stats = budget_analysis.summarize_grouped_metric(
        environment_cells,
        group_columns=['sampling_temperature', 'temporal_weight', 'sample_replacement'],
        metric_columns=[
            'accessibility_mean',
            'capture_rate',
            'value_alignment_mean',
            'policy_alignment_mean',
            'repeat_win',
            'repeat_loss',
            'choice_interaction',
        ],
        n_boot=n_boot,
        random_state=random_state,
    )
    reward_summary = budget_analysis.summarize_grouped_metric(
        environment_cells,
        group_columns=['sampling_temperature', 'temporal_weight', 'sample_replacement'],
        metric_columns=['total_reward'],
        n_boot=n_boot,
        random_state=random_state,
    )
    lag_slopes.to_csv(output_dir / 'environment_lag_slopes.csv', index=False)
    planned_contrasts.to_csv(output_dir / 'planned_contrasts.csv', index=False)
    lag_band_stats.to_csv(output_dir / 'reminder_stats_by_lag_band.csv', index=False)
    continuous_lag_stats.to_csv(output_dir / 'reminder_stats_by_continuous_lag.csv', index=False)
    cell_stats.to_csv(output_dir / 'reminder_stats_by_grid_cell.csv', index=False)
    reward_summary.to_csv(output_dir / 'reward_summary.csv', index=False)
    plot_causal_chain_panels(
        continuous_lag_stats=continuous_lag_stats,
        cell_stats=cell_stats,
        environment_cells=environment_cells,
        grid=grid,
        plot_aesthetics=plot_aesthetics,
        plot_temperatures=plot_temperatures,
        n_boot=n_boot,
        random_state=random_state,
        visual_dir=visual_dir,
    )


def run_analysis(args: Any) -> None:
    """
    Load configured replacement simulation results and run the analysis.

    Args:
        args (Any): Parsed command-line arguments.
    """
    config = resolve_analysis_config(args.config)
    input_dir = args.input_dir or get_replacement_output_dir(
        root_dir=filepaths.EXPERIMENTS_DIR,
        output_name=config['experiment_name'],
        sample_replacement=True,
    )
    reminder_table, reward_table = load_experiment_tables(
        experiment_dir=input_dir,
        expected_grid=config['grid'],
        n_environments=config['n_environments'],
        n_agent_simulations=config['n_agent_simulations'],
    )
    output_dir = args.output_dir or get_replacement_output_dir(
        root_dir=filepaths.OUTPUT_DIR,
        output_name=config['output_dir_name'],
        sample_replacement=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_analysis_tables(
        reminder_table=reminder_table,
        reward_table=reward_table,
        output_dir=output_dir,
        n_boot=config['n_boot'],
        random_state=config['random_state'],
        grid=config['grid'],
        plot_aesthetics=config,
        plot_temperatures=config['plot_sampling_temperatures'],
    )


def main(argv: list[str] | None = None) -> None:
    """
    Run replacement analysis from command-line arguments.

    Args:
        argv (list[str] | None): Optional command-line argument list.
    """
    parser = budget_analysis.parse_args(argv)
    if parser.config == budget_analysis.DEFAULT_ANALYSIS_CONFIG_PATH:
        parser.config = DEFAULT_ANALYSIS_CONFIG_PATH
    run_analysis(parser)


if __name__ == '__main__':
    main()
