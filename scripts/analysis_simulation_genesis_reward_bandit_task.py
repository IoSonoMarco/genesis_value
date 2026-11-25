"""
Analyze reminder-biased retrieval in the GENESIS reward-bandit task.

The analysis follows the reminder target from retrieval accessibility through
sample capture, reconstructed action values, policy logits, and realized choice.
Reminder lag is the primary expression of temporal interference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence
import sys
import warnings

sys.path.insert(0, str(Path.cwd()))
warnings.simplefilter('ignore')

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from utils import filepaths
from utils.simulation_config import get_replacement_output_dir, load_simulation_config


DEFAULT_ANALYSIS_CONFIG_PATH = (
    filepaths.ROOT / 'configs' / 'analysis_simulation_genesis_reward_bandit_task.yaml'
)
GROUP_COLUMNS = [
    'environment_id',
    'sampling_budget',
    'temporal_weight',
    'sample_replacement',
]
LAG_BAND_ORDER = ['short', 'medium', 'long']
DEFAULT_PLOT_AESTHETICS = {
    'sampling_budget_colors': {
        '1': '#2E338E',
        '5': '#92297D',
        '20': '#F78D60',
    },
    'nonsignificant_color': '#FF0000',
    'nonsignificant_hatch': '/',
    'temporal_weight_cmap': 'viridis',
    'choice_marker_size': 7,
    'choice_errorbar_linewidth': 7,
    'choice_errorbar_alpha': 0.45,
    'choice_jitter': 0.035,
    'plot_temporal_weights': [0.0, 1.0],
    'rag_demo': {
        'trial': 40,
        'reminder_trial': 15,
        'temporal_weight': 0.2,
        'embedding_dim': 64,
        'reminder_similarity': 0.99,
        'sampling_temperature': 0.6,
        'reminder_target_probability': 0.4,
        'background_spike_count': 4,
        'background_spike_scale': 0.5,
        'background_noise_scale': 0.25,
        'random_state': 44,
        'temporal_color': '#2E338E',
        'reminder_color': '#408175',
        'target_color': '#EA2264',
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse analysis command-line arguments.

    Args:
        argv (list[str] | None): Optional command-line arguments.

    Returns:
        argparse.Namespace: Parsed analysis arguments.
    """
    parser = argparse.ArgumentParser(description='Analyze GENESIS reminder-biased retrieval.')
    parser.add_argument('--config', type=Path, default=DEFAULT_ANALYSIS_CONFIG_PATH)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)

    return parser.parse_args(argv)


def resolve_analysis_config(config_path: Path) -> dict[str, Any]:
    """
    Load analysis controls and the linked simulation definition.

    Args:
        config_path (Path): Analysis YAML configuration path.

    Returns:
        dict[str, Any]: Analysis controls enriched with simulation settings.
    """
    analysis_config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    simulation_config_path = Path(analysis_config['simulation_config_path'])

    if not simulation_config_path.is_absolute():
        simulation_config_path = filepaths.ROOT / simulation_config_path

    simulation_config = load_simulation_config(simulation_config_path)
    grid = {
        **simulation_config['grid'],
        'sample_replacement': [simulation_config['sample_replacement']],
    }

    return {
        **analysis_config,
        'experiment_name': simulation_config['experiment_name'],
        'sample_replacement': simulation_config['sample_replacement'],
        'grid': grid,
        'n_environments': simulation_config['n_environments'],
        'n_agent_simulations': simulation_config['n_agent_simulations'],
    }


def load_experiment_tables(
    *,
    experiment_dir: Path,
    expected_grid: dict[str, list[Any]],
    n_environments: int,
    n_agent_simulations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load nested simulation outputs into reminder and reward tables.

    Args:
        experiment_dir (Path): Directory containing one NumPy file per grid cell.
        expected_grid (dict[str, list[Any]]): Expected factorial grid.
        n_environments (int): Expected environment count per cell.
        n_agent_simulations (int): Expected agent count per environment.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Reminder-level and agent-reward tables.
    """
    reminder_rows = []
    reward_rows = []
    observed_cells = []
    n_choice_trials = None
    n_reminders = None

    for savepath in sorted(experiment_dir.glob('*.npy')):
        result = np.load(savepath, allow_pickle=True).item()
        config = result['config']
        sampling_budget = str(config['sampling_budget'])
        temporal_weight = float(config['temporal_weight'])
        sample_replacement = bool(config['sample_replacement'])
        observed_cells.append((sampling_budget, temporal_weight, sample_replacement))
        environments = result['environments']

        assert len(environments) == n_environments

        for environment in environments:
            agent_runs = environment['agent_runs']
            assert len(agent_runs) == n_agent_simulations

            for agent_run in agent_runs:
                simulation_log = agent_run['simulation_log']
                if n_choice_trials is None:
                    n_choice_trials = len(simulation_log)
                assert len(simulation_log) == n_choice_trials
                reminder_log = [record for record in simulation_log if record['is_reminder_trial']]
                if n_reminders is None:
                    n_reminders = len(reminder_log)
                assert len(reminder_log) == n_reminders

                identifiers = {
                    'environment_id': environment['environment_id'],
                    'agent_simulation_id': agent_run['agent_simulation_id'],
                    'sampling_budget': sampling_budget,
                    'temporal_weight': temporal_weight,
                    'sample_replacement': sample_replacement,
                }

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
        (str(sampling_budget), float(temporal_weight), bool(sample_replacement))
        for sampling_budget in expected_grid['sampling_budget']
        for temporal_weight in expected_grid['temporal_weight']
        for sample_replacement in expected_grid['sample_replacement']
    }
    assert len(observed_cells) == len(expected_cells)
    assert set(observed_cells) == expected_cells

    return pd.DataFrame(reminder_rows), pd.DataFrame(reward_rows)


def add_reminder_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add retrieval, value, policy, choice, and lag metrics.

    Args:
        df (pd.DataFrame): Reminder-level simulation records.

    Returns:
        pd.DataFrame: Reminder records with derived causal-chain metrics.
    """
    result = df.copy()
    result['sampling_budget'] = result['sampling_budget'].astype(str)
    result['lag_10'] = (result['reminder_lag'] - 45.0) / 10.0
    result['lag_band'] = pd.cut(
        result['reminder_lag'],
        bins=[19.5, 36.5, 53.5, 70.5],
        labels=LAG_BAND_ORDER,
        ordered=True,
    )
    result['accessibility_enrichment'] = (
        result['reminder_target_sampling_probability']
        * result['reminder_target_candidate_count']
    )
    result['log_accessibility_enrichment'] = np.log(result['accessibility_enrichment'])

    rank_denominator = (result['reminder_target_candidate_count'] - 1).replace(0, np.nan)
    result['normalized_target_rank'] = (
        1.0
        - (result['reminder_target_similarity_rank'] - 1) / rank_denominator
    ).fillna(1.0)

    result['target_action_sign'] = np.where(
        result['reminder_target_chosen_arm'] == 'RIGHT',
        1.0,
        -1.0,
    )
    result['value_difference'] = result['action_value_right'] - result['action_value_left']
    signed_target = result['target_action_sign'] * result['reminder_target_reward']
    result['signed_value_alignment'] = signed_target * result['value_difference']
    result['signed_policy_alignment'] = signed_target * result['logit_right']
    result['repeated_target_action'] = (
        result['chosen_arm'] == result['reminder_target_chosen_arm']
    )

    return result


def fit_slope(*, x: Sequence[float], y: Sequence[float]) -> float:
    """
    Fit an ordinary least-squares slope.

    Args:
        x (Sequence[float]): Predictor values.
        y (Sequence[float]): Outcome values.

    Returns:
        float: Fitted slope.
    """
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    slope, _ = np.polyfit(x_array[valid], y_array[valid], deg=1)

    return float(slope)


def compute_environment_lag_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate causal-chain lag slopes within every environment and grid cell.

    Args:
        df (pd.DataFrame): Reminder table with derived metrics.

    Returns:
        pd.DataFrame: Environment-by-cell lag slopes per ten trials.
    """
    outcome_columns = {
        'accessibility_lag_slope': 'log_accessibility_enrichment',
        'rank_lag_slope': 'normalized_target_rank',
        'capture_lag_slope': 'reminder_target_sampled',
        'value_alignment_lag_slope': 'signed_value_alignment',
        'policy_alignment_lag_slope': 'signed_policy_alignment',
    }
    rows = []

    for identifiers, group in df.groupby(GROUP_COLUMNS, observed=True):
        row = dict(zip(GROUP_COLUMNS, identifiers))

        for slope_name, outcome_name in outcome_columns.items():
            row[slope_name] = fit_slope(
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
        repeat_win = float(wins['repeated_target_action'].mean())
        repeat_loss = float(losses['repeated_target_action'].mean())
        rows.append({
            **dict(zip(GROUP_COLUMNS, identifiers)),
            'accessibility_mean': float(group['log_accessibility_enrichment'].mean()),
            'capture_rate': float(group['reminder_target_sampled'].mean()),
            'value_alignment_mean': float(group['signed_value_alignment'].mean()),
            'policy_alignment_mean': float(group['signed_policy_alignment'].mean()),
            'repeat_win': repeat_win,
            'repeat_loss': repeat_loss,
            'choice_interaction': repeat_win - repeat_loss,
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


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int,
    random_state: int,
) -> tuple[float, float, float]:
    """
    Bootstrap a mean and percentile confidence interval.

    Args:
        values (Sequence[float]): Independent environment-level values.
        n_boot (int): Number of bootstrap resamples.
        random_state (int): Bootstrap seed.

    Returns:
        tuple[float, float, float]: Mean, lower bound, and upper bound.
    """
    values_array = np.asarray(values, dtype=float)
    values_array = values_array[np.isfinite(values_array)]

    if len(values_array) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(random_state)
    sample_indices = rng.integers(
        0,
        len(values_array),
        size=(n_boot, len(values_array)),
    )
    bootstrap_means = values_array[sample_indices].mean(axis=1)
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])

    return float(values_array.mean()), float(ci_low), float(ci_high)


def summarize_contrast(
    *,
    contrast: str,
    values: pd.Series,
    n_boot: int,
    random_state: int,
) -> dict[str, float | str | int]:
    """
    Summarize one paired environment-level contrast.

    Args:
        contrast (str): Contrast label.
        values (pd.Series): One paired contrast per environment.
        n_boot (int): Number of bootstrap resamples.
        random_state (int): Bootstrap seed.

    Returns:
        dict[str, float | str | int]: Contrast estimate and confidence interval.
    """
    estimate, ci_low, ci_high = bootstrap_mean_ci(
        values,
        n_boot=n_boot,
        random_state=random_state,
    )

    return {
        'contrast': contrast,
        'estimate': estimate,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'n_environments': int(values.notna().sum()),
    }


def compute_planned_contrasts(
    slope_table: pd.DataFrame,
    environment_cells: pd.DataFrame,
    *,
    grid: dict[str, list[Any]],
    n_boot: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Compute pairwise temporal-weight and budget contrasts.

    Args:
        slope_table (pd.DataFrame): Environment-level lag slopes.
        environment_cells (pd.DataFrame): Environment-level causal-chain outcomes.
        n_boot (int): Number of bootstrap resamples.
        random_state (int): Bootstrap seed.

    Returns:
        pd.DataFrame: Planned contrast estimates and confidence intervals.
    """
    rows = []
    sampling_budgets = [str(value) for value in grid['sampling_budget']]
    temporal_weights = [float(value) for value in grid['temporal_weight']]
    sample_replacements = [bool(value) for value in grid['sample_replacement']]
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
        (
            environment_cells,
            [
                'value_alignment_mean',
                'policy_alignment_mean',
                'choice_interaction',
            ],
        ),
    ]

    for table, metrics in metric_tables:
        for sample_replacement in sample_replacements:
            replacement_table = table[
                table['sample_replacement'] == sample_replacement
            ]

            for metric in metrics:
                collapsed = (
                    replacement_table
                    .groupby(['environment_id', 'temporal_weight'], observed=True)[metric]
                    .mean()
                    .unstack('temporal_weight')
                )
                budget_effects = {}

                for weight_index, first_weight in enumerate(temporal_weights[:-1]):
                    for second_weight in temporal_weights[weight_index + 1:]:
                        rows.append(summarize_contrast(
                            contrast=(
                                f'{metric}_weight-{first_weight}-minus-{second_weight}'
                            ),
                            values=collapsed[first_weight] - collapsed[second_weight],
                            n_boot=n_boot,
                            random_state=random_state + len(rows),
                        ))

                        for sampling_budget in sampling_budgets:
                            budget_table = (
                                replacement_table[
                                    replacement_table['sampling_budget'] == sampling_budget
                                ]
                                .pivot(
                                    index='environment_id',
                                    columns='temporal_weight',
                                    values=metric,
                                )
                            )
                            budget_effects[
                                sampling_budget,
                                first_weight,
                                second_weight,
                            ] = budget_table[first_weight] - budget_table[second_weight]
                            rows.append(summarize_contrast(
                                contrast=(
                                    f'{metric}_weight-{first_weight}-minus-{second_weight}_'
                                    f'budget-{sampling_budget}'
                                ),
                                values=budget_effects[
                                    sampling_budget,
                                    first_weight,
                                    second_weight,
                                ],
                                n_boot=n_boot,
                                random_state=random_state + len(rows),
                            ))

                        for budget_index, first_budget in enumerate(sampling_budgets[:-1]):
                            for second_budget in sampling_budgets[budget_index + 1:]:
                                rows.append(summarize_contrast(
                                    contrast=(
                                        f'{metric.removesuffix("_mean")}_budget-{first_budget}-'
                                        f'minus-{second_budget}_weight-{first_weight}-'
                                        f'minus-{second_weight}'
                                    ),
                                    values=(
                                        budget_effects[first_budget, first_weight, second_weight]
                                        - budget_effects[second_budget, first_weight, second_weight]
                                    ),
                                    n_boot=n_boot,
                                    random_state=random_state + len(rows),
                                ))

    return pd.DataFrame(rows)


def summarize_grouped_metric(
    df: pd.DataFrame,
    *,
    group_columns: list[str],
    metric_columns: list[str],
    n_boot: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Summarize environment-level metrics within plotting groups.

    Args:
        df (pd.DataFrame): Environment-level observations.
        group_columns (list[str]): Columns defining output groups.
        metric_columns (list[str]): Metrics to summarize.
        n_boot (int): Number of bootstrap resamples.
        random_state (int): Bootstrap seed.

    Returns:
        pd.DataFrame: Long-format estimates and confidence intervals.
    """
    rows = []

    for identifiers, group in df.groupby(group_columns, observed=True):
        identifiers = identifiers if isinstance(identifiers, tuple) else (identifiers,)
        identifier_fields = dict(zip(group_columns, identifiers))

        for metric in metric_columns:
            estimate, ci_low, ci_high = bootstrap_mean_ci(
                group[metric],
                n_boot=n_boot,
                random_state=random_state + len(rows),
            )
            rows.append({
                **identifier_fields,
                'metric': metric,
                'estimate': estimate,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'n_environments': group['environment_id'].nunique(),
            })

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

    return summarize_grouped_metric(
        environment_bands,
        group_columns=['sampling_budget', 'temporal_weight', 'sample_replacement', 'lag_band'],
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
        reminder_table (pd.DataFrame): Reminder table with derived metrics.
        n_boot (int): Number of environment bootstrap resamples.
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

    return summarize_grouped_metric(
        environment_lags,
        group_columns=[
            'sampling_budget',
            'temporal_weight',
            'sample_replacement',
            'reminder_lag',
        ],
        metric_columns=['log_accessibility_enrichment'],
        n_boot=n_boot,
        random_state=random_state,
    )


def get_temporal_weight_colors(
    *,
    temporal_weights: list[float],
    cmap_name: str,
) -> list[tuple[float, float, float, float]]:
    """
    Resolve evenly spaced colors for temporal-weight conditions.

    Args:
        temporal_weights (list[float]): Temporal weights to visualize.
        cmap_name (str): Matplotlib colormap name.

    Returns:
        list[tuple[float, float, float, float]]: One RGBA color per temporal weight.
    """
    color_positions = np.linspace(0.15, 0.85, len(temporal_weights))
    colormap = plt.get_cmap(cmap_name)

    return [colormap(position) for position in color_positions]


def get_sampling_budget_jitters(
    *,
    sampling_budgets: list[str],
    jitter: float,
) -> dict[str, float]:
    """
    Center horizontal offsets for sampling-budget conditions.

    Args:
        sampling_budgets (list[str]): Ordered sampling-budget labels.
        jitter (float): Distance between neighboring budget positions.

    Returns:
        dict[str, float]: Horizontal offset by sampling budget.
    """
    center = (len(sampling_budgets) - 1) / 2

    return {
        sampling_budget: (index - center) * jitter
        for index, sampling_budget in enumerate(sampling_budgets)
    }


def get_plot_temporal_weights(
    *,
    grid: dict[str, list[Any]],
    plot_aesthetics: dict[str, Any] | None,
) -> list[float]:
    """
    Resolve the temporal-weight levels included in figures.

    Args:
        grid (dict[str, list[Any]]): Simulation grid that defines available conditions.
        plot_aesthetics (dict[str, Any] | None): Figure styling controls.

    Returns:
        list[float]: Ordered temporal-weight levels to visualize.
    """
    configured_weights = (plot_aesthetics or {}).get(
        'plot_temporal_weights',
        DEFAULT_PLOT_AESTHETICS['plot_temporal_weights'],
    )
    temporal_weights = [float(value) for value in configured_weights]
    grid_weights = {float(value) for value in grid['temporal_weight']}

    assert set(temporal_weights).issubset(grid_weights)

    return temporal_weights


def get_choice_effect_limit(
    *,
    cell_stats: pd.DataFrame,
    temporal_weights: list[float],
) -> float:
    """
    Resolve a shared absolute choice-effect axis limit.

    Args:
        cell_stats (pd.DataFrame): Grid-cell causal-chain summaries.
        temporal_weights (list[float]): Temporal weights included in the figure.

    Returns:
        float: Largest absolute confidence-interval endpoint around chance.
    """
    choice_stats = cell_stats[
        cell_stats['metric'].isin(['repeat_win', 'repeat_loss'])
        & cell_stats['temporal_weight'].isin(temporal_weights)
    ]
    endpoints = choice_stats[['ci_low', 'ci_high']].to_numpy() - 0.5
    finite_endpoints = endpoints[np.isfinite(endpoints)]
    if len(finite_endpoints) == 0:
        return 0.05

    return max(float(np.abs(finite_endpoints).max()), 0.05)


def build_rag_demo_scores(
    *,
    trial: int,
    reminder_trial: int,
    temporal_weight: float,
    embedding_dim: int,
    reminder_similarity: float,
    sampling_temperature: float,
    reminder_target_probability: float,
    background_spike_count: int,
    background_spike_scale: float,
    background_noise_scale: float,
    random_state: int,
) -> dict[str, np.ndarray]:
    """
    Build synthetic rank-normalized query-key scores for one arm.

    Args:
        trial (int): Current one-indexed trial number.
        reminder_trial (int): Prior trial represented by the reminder.
        temporal_weight (float): Temporal query-key weight with the reminder.
        embedding_dim (int): Temporal and stimulus embedding dimension.
        reminder_similarity (float): Cosine similarity between reminder and target stimulus.
        sampling_temperature (float): Softmax temperature for the illustrative retrieval distribution.
        reminder_target_probability (float): Reserved retrieval probability for the reminder target.
        background_spike_count (int): Number of incidental background retrieval spikes.
        background_spike_scale (float): Spike height relative to the strongest background value.
        background_noise_scale (float): Multiplicative noise scale for the temporal background.
        random_state (int): Seed for the explanatory simulation.

    Returns:
        dict[str, np.ndarray]: Prior trial identifiers and explanatory score series.
    """
    rng = np.random.RandomState(random_state)
    temporal_embeddings = generate_demo_temporal_embeddings(
        trial=trial,
        embedding_dim=embedding_dim,
        rng=rng,
    )
    prior_trials = np.arange(trial - 1, 0, -1)
    prior_indices = prior_trials - 1
    temporal_scores = temporal_embeddings[prior_indices] @ temporal_embeddings[trial - 1]
    stimulus_embeddings = rng.standard_normal((len(prior_trials), embedding_dim))
    stimulus_embeddings /= np.linalg.norm(stimulus_embeddings, axis=1, keepdims=True)
    target_index = int(np.where(prior_trials == reminder_trial)[0][0])
    target_stimulus = stimulus_embeddings[target_index]
    reminder_noise = rng.standard_normal(embedding_dim)
    reminder_noise -= reminder_noise.dot(target_stimulus) * target_stimulus
    reminder_noise /= np.linalg.norm(reminder_noise)
    reminder_stimulus = (
        reminder_similarity * target_stimulus
        + np.sqrt(1.0 - reminder_similarity ** 2) * reminder_noise
    )
    stimulus_scores = stimulus_embeddings @ reminder_stimulus

    def rank_normalize(scores: np.ndarray) -> np.ndarray:
        descending_ranks = np.argsort(
            np.argsort(-scores, kind='stable'),
            kind='stable',
        )

        return 2.0 * (1.0 - (descending_ranks + 0.5) / len(scores)) - 1.0

    temporal_rank_scores = rank_normalize(temporal_scores)

    def softmax(scores: np.ndarray) -> np.ndarray:
        shifted_scores = scores / sampling_temperature
        shifted_scores -= shifted_scores.max()
        probabilities = np.exp(shifted_scores)

        return probabilities / probabilities.sum()

    temporal_only_scores = softmax(temporal_rank_scores)
    reminder_background = temporal_only_scores * rng.lognormal(
        mean=-0.5 * background_noise_scale ** 2,
        sigma=background_noise_scale,
        size=len(temporal_only_scores),
    )
    background_candidates = np.delete(np.arange(len(prior_trials)), target_index)
    background_spikes = background_candidates[
        np.argsort(stimulus_scores[background_candidates])[-background_spike_count:]
    ]
    reminder_background[background_spikes] += (
        background_spike_scale * reminder_background.max()
    )
    reminder_background /= reminder_background.sum()
    target_probability = reminder_target_probability * reminder_similarity
    reminder_scores = (1.0 - target_probability) * reminder_background
    reminder_scores[target_index] += target_probability

    return {
        'prior_trials': prior_trials,
        'temporal_only_scores': temporal_only_scores,
        'reminder_scores': reminder_scores,
        'reminder_background': reminder_background,
        'reminder_trial': np.asarray([reminder_trial]),
    }


def generate_demo_temporal_embeddings(
    *,
    trial: int,
    embedding_dim: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Generate the sinusoidal-step temporal trajectory used by the simulation.

    Args:
        trial (int): Number of temporal embeddings to generate.
        embedding_dim (int): Temporal embedding dimension.
        rng (np.random.RandomState): Seeded random generator.

    Returns:
        np.ndarray: Temporal embeddings with shape ``[trial, embedding_dim]``.
    """
    current_embedding = rng.standard_normal(embedding_dim)
    current_embedding /= np.linalg.norm(current_embedding)
    embeddings = []

    for _ in range(trial):
        orthogonal = rng.standard_normal(embedding_dim)
        orthogonal -= orthogonal.dot(current_embedding) * current_embedding
        orthogonal /= np.linalg.norm(orthogonal)
        current_embedding = (
            np.cos(np.pi / 100) * current_embedding
            + np.sin(np.pi / 100) * orthogonal
        )
        embeddings.append(current_embedding)

    return np.stack(embeddings)


def plot_causal_chain(
    *,
    lag_band_stats: pd.DataFrame,
    continuous_lag_stats: pd.DataFrame,
    cell_stats: pd.DataFrame,
    grid: dict[str, list[Any]],
    plot_aesthetics: dict[str, Any] | None,
    savepath: Path,
) -> None:
    """
    Plot retrieval accessibility, value alignment, and reminder-biased choice.

    Args:
        lag_band_stats (pd.DataFrame): Lag-band causal-chain summaries.
        continuous_lag_stats (pd.DataFrame): Exact-lag accessibility summaries.
        cell_stats (pd.DataFrame): Grid-cell causal-chain summaries.
        savepath (Path): Destination PDF path.
    """
    aesthetics = {
        **DEFAULT_PLOT_AESTHETICS,
        **(plot_aesthetics or {}),
    }
    sampling_budgets = [str(value) for value in grid['sampling_budget']]
    temporal_weights = [float(value) for value in grid['temporal_weight']]
    sample_replacements = [bool(value) for value in grid['sample_replacement']]
    budget_colors = {
        sampling_budget: aesthetics['sampling_budget_colors'][sampling_budget]
        for sampling_budget in sampling_budgets
    }
    temporal_colors = get_temporal_weight_colors(
        temporal_weights=temporal_weights,
        cmap_name=aesthetics['temporal_weight_cmap'],
    )
    budget_jitters = get_sampling_budget_jitters(
        sampling_budgets=sampling_budgets,
        jitter=aesthetics['choice_jitter'],
    )
    rag_demo_config = {
        **DEFAULT_PLOT_AESTHETICS['rag_demo'],
        **aesthetics['rag_demo'],
    }
    lag_positions = np.arange(len(LAG_BAND_ORDER))
    score_budget = '1' if '1' in sampling_budgets else sampling_budgets[0]
    fig = plt.figure(figsize=(18, 13))
    grid_spec = fig.add_gridspec(2, 3)
    accessibility_band_axis = fig.add_subplot(grid_spec[0, 0])
    accessibility_continuous_axis = fig.add_subplot(grid_spec[0, 1])
    value_axis = fig.add_subplot(grid_spec[0, 2])
    win_axis = fig.add_subplot(grid_spec[1, 0])
    loss_axis = fig.add_subplot(grid_spec[1, 1])
    rag_grid_spec = grid_spec[1, 2].subgridspec(2, 1, hspace=0.5)
    rag_temporal_axis = fig.add_subplot(rag_grid_spec[0, 0])
    rag_reminder_axis = fig.add_subplot(rag_grid_spec[1, 0])
    accessibility = lag_band_stats[
        lag_band_stats['metric'] == 'log_accessibility_enrichment'
    ]

    for temporal_weight, temporal_color in zip(temporal_weights, temporal_colors):
        for sample_replacement in sample_replacements:
            group = accessibility[
                (accessibility['sampling_budget'] == score_budget)
                & (accessibility['temporal_weight'] == temporal_weight)
                & (accessibility['sample_replacement'] == sample_replacement)
            ].set_index('lag_band').reindex(LAG_BAND_ORDER)
            accessibility_band_axis.plot(
                lag_positions,
                group['estimate'],
                color=temporal_color,
                marker='o',
                label=f'w={temporal_weight}',
            )

            continuous_group = continuous_lag_stats[
                (continuous_lag_stats['sampling_budget'] == score_budget)
                & (continuous_lag_stats['temporal_weight'] == temporal_weight)
                & (continuous_lag_stats['sample_replacement'] == sample_replacement)
            ].sort_values('reminder_lag')
            accessibility_continuous_axis.plot(
                continuous_group['reminder_lag'],
                continuous_group['estimate'],
                color=temporal_color,
                label=f'w={temporal_weight}',
            )
            accessibility_continuous_axis.fill_between(
                continuous_group['reminder_lag'],
                continuous_group['ci_low'],
                continuous_group['ci_high'],
                color=temporal_color,
                alpha=0.08,
            )

    for axis, title in [
        (accessibility_band_axis, '1a. Target retrieval by lag band'),
        (accessibility_continuous_axis, '1b. Target retrieval across lag'),
    ]:
        axis.axhline(0.0, color='black', linewidth=1, alpha=0.4)
        axis.set_ylabel('Target retrieval advantage vs. uniform')
        axis.set_title(title)

    accessibility_band_axis.set_xticks(lag_positions, LAG_BAND_ORDER)
    accessibility_band_axis.set_xlabel('Reminder lag band')
    accessibility_continuous_axis.set_xlabel('Reminder lag (trials)')

    value_stats = cell_stats[cell_stats['metric'] == 'value_alignment_mean']

    for sampling_budget in sampling_budgets:
        for sample_replacement in sample_replacements:
            group = (
                value_stats[
                    (value_stats['sampling_budget'] == sampling_budget)
                    & (value_stats['sample_replacement'] == sample_replacement)
                ]
                .set_index('temporal_weight')
                .reindex(temporal_weights)
            )
            x_positions = np.asarray(temporal_weights) + budget_jitters[sampling_budget]
            estimates = group['estimate'].to_numpy()
            value_axis.errorbar(
                x_positions,
                estimates,
                yerr=[
                    estimates - group['ci_low'].to_numpy(),
                    group['ci_high'].to_numpy() - estimates,
                ],
                color=budget_colors[sampling_budget],
                marker='o',
                markersize=aesthetics['choice_marker_size'],
                linestyle='none',
                ecolor=to_rgba(
                    budget_colors[sampling_budget],
                    alpha=aesthetics['choice_errorbar_alpha'],
                ),
                elinewidth=aesthetics['choice_errorbar_linewidth'],
                capsize=0,
                label=f'k={sampling_budget}, replace={sample_replacement}',
            )
            regression_slope, regression_intercept = np.polyfit(
                temporal_weights,
                estimates,
                deg=1,
            )
            value_axis.plot(
                temporal_weights,
                regression_slope * np.asarray(temporal_weights) + regression_intercept,
                color=budget_colors[sampling_budget],
                linestyle='--',
            )

    value_axis.set_xlabel('Temporal weight')
    value_axis.set_xticks(temporal_weights, [f'{weight:g}' for weight in temporal_weights])
    value_axis.set_ylabel('Signed value alignment')
    value_axis.set_title('2. Value reconstruction')

    for axis, metric, title, direction in [
        (win_axis, 'repeat_win', '3a. Repeat after win', 1.0),
        (loss_axis, 'repeat_loss', '3b. Repeat after loss', -1.0),
    ]:
        for sampling_budget in sampling_budgets:
            jitter = budget_jitters[sampling_budget]
            for sample_replacement in sample_replacements:
                group = cell_stats[
                    (cell_stats['sampling_budget'] == sampling_budget)
                    & (cell_stats['sample_replacement'] == sample_replacement)
                    & (cell_stats['metric'] == metric)
                ].set_index('temporal_weight').reindex(temporal_weights)
                estimates = group['estimate'].to_numpy() - 0.5
                ci_low = group['ci_low'].to_numpy() - 0.5
                ci_high = group['ci_high'].to_numpy() - 0.5
                x_positions = np.asarray(temporal_weights) + jitter
                axis.errorbar(
                    x_positions,
                    estimates,
                    yerr=[estimates - ci_low, ci_high - estimates],
                    color=budget_colors[sampling_budget],
                    marker='o',
                    markersize=aesthetics['choice_marker_size'],
                    linestyle='--',
                    ecolor=to_rgba(
                        budget_colors[sampling_budget],
                        alpha=aesthetics['choice_errorbar_alpha'],
                    ),
                    elinewidth=aesthetics['choice_errorbar_linewidth'],
                    capsize=0,
                    linewidth=1,
                    label=f'k={sampling_budget}',
                )

                for x_position, low, high in zip(x_positions, ci_low, ci_high):
                    nonsignificant_low = low if direction > 0.0 and low < 0.0 else 0.0
                    nonsignificant_high = high if direction < 0.0 and high > 0.0 else 0.0

                    if nonsignificant_low != nonsignificant_high:
                        lower, upper = sorted([nonsignificant_low, nonsignificant_high])
                        axis.add_patch(Rectangle(
                            (x_position - aesthetics['choice_jitter'] / 2, lower),
                            aesthetics['choice_jitter'],
                            upper - lower,
                            facecolor='none',
                            edgecolor=aesthetics['nonsignificant_color'],
                            hatch=aesthetics['nonsignificant_hatch'],
                            linewidth=0.5,
                            zorder=4,
                        ))

        axis.axhline(0.0, color='black', linewidth=1, alpha=0.4)
        axis.set_xlabel('Temporal weight')
        axis.set_ylabel('Δ P(repeat target action)')
        axis.set_title(title)

    rag_demo_scores = build_rag_demo_scores(
        trial=rag_demo_config['trial'],
        reminder_trial=rag_demo_config['reminder_trial'],
        temporal_weight=rag_demo_config['temporal_weight'],
        embedding_dim=rag_demo_config['embedding_dim'],
        reminder_similarity=rag_demo_config['reminder_similarity'],
        sampling_temperature=rag_demo_config['sampling_temperature'],
        reminder_target_probability=rag_demo_config['reminder_target_probability'],
        background_spike_count=rag_demo_config['background_spike_count'],
        background_spike_scale=rag_demo_config['background_spike_scale'],
        background_noise_scale=rag_demo_config['background_noise_scale'],
        random_state=rag_demo_config['random_state'],
    )
    rag_positions = np.arange(len(rag_demo_scores['prior_trials']))
    target_position = int(np.where(
        rag_demo_scores['prior_trials'] == rag_demo_config['reminder_trial'],
    )[0][0])

    rag_temporal_axis.bar(
        rag_positions,
        rag_demo_scores['temporal_only_scores'],
        color=rag_demo_config['temporal_color'],
    )
    rag_temporal_axis.set_title('4a. Temporal-only query key (w=1.0)')
    rag_temporal_axis.set_ylabel('Retrieval probability')

    reminder_bar_colors = np.full(
        len(rag_positions),
        rag_demo_config['reminder_color'],
        dtype=object,
    )
    reminder_bar_colors[target_position] = rag_demo_config['target_color']
    rag_reminder_axis.bar(
        rag_positions,
        rag_demo_scores['reminder_scores'],
        color=reminder_bar_colors,
    )
    rag_reminder_axis.set_title(
        f'4b. Reminder query key (w={rag_demo_config["temporal_weight"]})',
    )
    rag_reminder_axis.set_xlabel('Previous trial (N−1 to 1)')
    rag_reminder_axis.set_ylabel('Retrieval probability')

    rag_tick_positions = [0, len(rag_positions) - 1]
    rag_tick_labels = [f'N−1 ({rag_demo_scores["prior_trials"][0]})', '1']
    for axis in [rag_temporal_axis, rag_reminder_axis]:
        axis.axhline(0.0, color='black', linewidth=1, alpha=0.4)
        axis.set_xticks(rag_tick_positions, rag_tick_labels)

    fig.legend(
        handles=[
            Line2D([], [], color=budget_colors[budget], marker='o', label=f'k={budget}')
            for budget in sampling_budgets
        ] + [
            Line2D([], [], color=temporal_color, label=f'w={weight}')
            for weight, temporal_color in zip(temporal_weights, temporal_colors)
        ] + [
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor='none',
                edgecolor=aesthetics['nonsignificant_color'],
                hatch=aesthetics['nonsignificant_hatch'],
                label='CI crosses chance',
            ),
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=rag_demo_config['target_color'],
                label='Reminder-matched episode',
            ),
        ],
        frameon=False,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
    )

    for axis in [
        accessibility_band_axis,
        accessibility_continuous_axis,
        value_axis,
        win_axis,
        loss_axis,
        rag_temporal_axis,
        rag_reminder_axis,
    ]:
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)

    fig.tight_layout(rect=(0.0, 0.1, 1.0, 1.0))
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)


def plot_causal_chain_panels(
    *,
    lag_band_stats: pd.DataFrame,
    continuous_lag_stats: pd.DataFrame,
    cell_stats: pd.DataFrame,
    grid: dict[str, list[Any]],
    plot_aesthetics: dict[str, Any] | None,
    visual_dir: Path,
) -> None:
    """
    Export the causal-chain panels as standalone PDFs.

    Args:
        lag_band_stats (pd.DataFrame): Lag-band causal-chain summaries.
        continuous_lag_stats (pd.DataFrame): Exact-lag accessibility summaries.
        cell_stats (pd.DataFrame): Grid-cell causal-chain summaries.
        grid (dict[str, list[Any]]): Simulation grid that defines analysis conditions.
        plot_aesthetics (dict[str, Any] | None): Figure styling controls.
        visual_dir (Path): Directory for standalone panel PDFs.
    """
    aesthetics = {
        **DEFAULT_PLOT_AESTHETICS,
        **(plot_aesthetics or {}),
    }
    sampling_budgets = [str(value) for value in grid['sampling_budget']]
    temporal_weights = [float(value) for value in grid['temporal_weight']]
    panel_temporal_weights = get_plot_temporal_weights(
        grid=grid,
        plot_aesthetics=plot_aesthetics,
    )
    sample_replacements = [bool(value) for value in grid['sample_replacement']]
    budget_colors = {
        sampling_budget: aesthetics['sampling_budget_colors'][sampling_budget]
        for sampling_budget in sampling_budgets
    }
    temporal_colors = get_temporal_weight_colors(
        temporal_weights=temporal_weights,
        cmap_name=aesthetics['temporal_weight_cmap'],
    )
    budget_jitters = get_sampling_budget_jitters(
        sampling_budgets=sampling_budgets,
        jitter=aesthetics['choice_jitter'],
    )
    score_budget = '1' if '1' in sampling_budgets else sampling_budgets[0]
    accessibility = lag_band_stats[
        lag_band_stats['metric'] == 'log_accessibility_enrichment'
    ]

    fig, axis = plt.subplots(figsize=aesthetics['panel_sizes']['1a'])
    lag_positions = np.arange(len(LAG_BAND_ORDER))
    for temporal_weight, temporal_color in zip(temporal_weights, temporal_colors):
        for sample_replacement in sample_replacements:
            group = accessibility[
                (accessibility['sampling_budget'] == score_budget)
                & (accessibility['temporal_weight'] == temporal_weight)
                & (accessibility['sample_replacement'] == sample_replacement)
            ].set_index('lag_band').reindex(LAG_BAND_ORDER)
            axis.plot(
                lag_positions,
                group['estimate'],
                color=temporal_color,
                marker='o',
                label=f'w={temporal_weight}',
            )
    axis.axhline(0.0, color='black', linewidth=1, alpha=0.4)
    axis.set_xticks(lag_positions, LAG_BAND_ORDER)
    axis.set_xlabel('Reminder lag band')
    axis.set_ylabel('Target retrieval advantage vs. uniform')
    axis.set_title('1a. Target retrieval by lag band')
    axis.legend(frameon=False)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(visual_dir / 'reminder_retrieval_causal_chain_panel_1a.pdf', bbox_inches='tight')
    plt.close(fig)

    fig, axis = plt.subplots(figsize=aesthetics['panel_sizes']['1b'])
    for temporal_weight, temporal_color in zip(temporal_weights, temporal_colors):
        for sample_replacement in sample_replacements:
            group = continuous_lag_stats[
                (continuous_lag_stats['sampling_budget'] == score_budget)
                & (continuous_lag_stats['temporal_weight'] == temporal_weight)
                & (continuous_lag_stats['sample_replacement'] == sample_replacement)
            ].sort_values('reminder_lag')
            axis.plot(
                group['reminder_lag'],
                group['estimate'],
                color=temporal_color,
                label=f'w={temporal_weight}',
            )
            axis.fill_between(
                group['reminder_lag'],
                group['ci_low'],
                group['ci_high'],
                color=temporal_color,
                alpha=0.08,
            )
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

    fig, axis = plt.subplots(figsize=aesthetics['panel_sizes']['2'])
    value_stats = cell_stats[cell_stats['metric'] == 'value_alignment_mean']
    for sampling_budget in sampling_budgets:
        for sample_replacement in sample_replacements:
            group = (
                value_stats[
                    (value_stats['sampling_budget'] == sampling_budget)
                    & (value_stats['sample_replacement'] == sample_replacement)
                ]
                .set_index('temporal_weight')
                .reindex(temporal_weights)
            )
            x_positions = np.asarray(temporal_weights) + budget_jitters[sampling_budget]
            estimates = group['estimate'].to_numpy()
            axis.errorbar(
                x_positions,
                estimates,
                yerr=[
                    estimates - group['ci_low'].to_numpy(),
                    group['ci_high'].to_numpy() - estimates,
                ],
                color=budget_colors[sampling_budget],
                marker='o',
                markersize=aesthetics['choice_marker_size'],
                linestyle='none',
                ecolor=to_rgba(
                    budget_colors[sampling_budget],
                    alpha=aesthetics['choice_errorbar_alpha'],
                ),
                elinewidth=aesthetics['choice_errorbar_linewidth'],
                capsize=0,
                label=f'k={sampling_budget}, replace={sample_replacement}',
            )
            regression_slope, regression_intercept = np.polyfit(
                temporal_weights,
                estimates,
                deg=1,
            )
            axis.plot(
                temporal_weights,
                regression_slope * np.asarray(temporal_weights) + regression_intercept,
                color=budget_colors[sampling_budget],
                linestyle='--',
            )
    axis.set_xlabel('Temporal weight')
    axis.set_xticks(temporal_weights, [f'{weight:g}' for weight in temporal_weights])
    axis.set_ylabel('Signed value alignment')
    axis.set_title('2. Value reconstruction')
    axis.legend(frameon=False)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(visual_dir / 'reminder_retrieval_causal_chain_panel_2.pdf', bbox_inches='tight')
    plt.close(fig)

    for panel_name, metric, title, direction in [
        ('3a', 'repeat_win', '3a. Repeat after win', 1.0),
        ('3b', 'repeat_loss', '3b. Repeat after loss', -1.0),
    ]:
        fig, axis = plt.subplots(figsize=aesthetics['panel_sizes'][panel_name])
        for sampling_budget in sampling_budgets:
            jitter = budget_jitters[sampling_budget]
            for sample_replacement in sample_replacements:
                group = cell_stats[
                    (cell_stats['sampling_budget'] == sampling_budget)
                    & (cell_stats['sample_replacement'] == sample_replacement)
                    & (cell_stats['metric'] == metric)
                ].set_index('temporal_weight').reindex(panel_temporal_weights)
                estimates = group['estimate'].to_numpy() - 0.5
                ci_low = group['ci_low'].to_numpy() - 0.5
                ci_high = group['ci_high'].to_numpy() - 0.5
                y_positions = np.arange(len(panel_temporal_weights)) + jitter
                axis.errorbar(
                    estimates,
                    y_positions,
                    xerr=[estimates - ci_low, ci_high - estimates],
                    color=budget_colors[sampling_budget],
                    marker='o',
                    markersize=aesthetics['choice_marker_size'],
                    linestyle='--',
                    ecolor=to_rgba(
                        budget_colors[sampling_budget],
                        alpha=aesthetics['choice_errorbar_alpha'],
                    ),
                    elinewidth=aesthetics['choice_errorbar_linewidth'],
                    capsize=0,
                    linewidth=1,
                    label=f'k={sampling_budget}',
                )
                for y_position, low, high in zip(y_positions, ci_low, ci_high):
                    if direction > 0.0 and low < 0.0 < high:
                        axis.add_patch(Rectangle(
                            (0.0, y_position - aesthetics['choice_jitter'] / 2),
                            high,
                            aesthetics['choice_jitter'],
                            facecolor='none',
                            edgecolor=aesthetics['nonsignificant_color'],
                            hatch=aesthetics['nonsignificant_hatch'],
                            linewidth=0.5,
                            zorder=4,
                        ))
                    if direction < 0.0 and low < 0.0 < high:
                        axis.add_patch(Rectangle(
                            (0.0, y_position - aesthetics['choice_jitter'] / 2),
                            high,
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
                Line2D([], [], color=budget_colors[budget], marker='o', label=f'k={budget}')
                for budget in sampling_budgets
            ] + [
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor='none',
                    edgecolor=aesthetics['nonsignificant_color'],
                    hatch=aesthetics['nonsignificant_hatch'],
                    label='CI crosses chance',
                ),
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
        fig.savefig(
            visual_dir / f'reminder_retrieval_causal_chain_panel_{panel_name}.pdf',
            bbox_inches='tight',
        )
        plt.close(fig)

    rag_demo_config = {
        **DEFAULT_PLOT_AESTHETICS['rag_demo'],
        **aesthetics['rag_demo'],
    }
    rag_demo_scores = build_rag_demo_scores(
        trial=rag_demo_config['trial'],
        reminder_trial=rag_demo_config['reminder_trial'],
        temporal_weight=rag_demo_config['temporal_weight'],
        embedding_dim=rag_demo_config['embedding_dim'],
        reminder_similarity=rag_demo_config['reminder_similarity'],
        sampling_temperature=rag_demo_config['sampling_temperature'],
        reminder_target_probability=rag_demo_config['reminder_target_probability'],
        background_spike_count=rag_demo_config['background_spike_count'],
        background_spike_scale=rag_demo_config['background_spike_scale'],
        background_noise_scale=rag_demo_config['background_noise_scale'],
        random_state=rag_demo_config['random_state'],
    )
    rag_positions = np.arange(len(rag_demo_scores['prior_trials']))
    target_position = int(np.where(
        rag_demo_scores['prior_trials'] == rag_demo_config['reminder_trial'],
    )[0][0])
    fig, axes = plt.subplots(2, 1, figsize=aesthetics['panel_sizes']['4'])
    axes[0].bar(
        rag_positions,
        rag_demo_scores['temporal_only_scores'],
        color=rag_demo_config['temporal_color'],
    )
    axes[0].set_title('4a. Temporal-only query key (w=1.0)')
    axes[0].set_ylabel('Retrieval probability')
    reminder_bar_colors = np.full(
        len(rag_positions),
        rag_demo_config['reminder_color'],
        dtype=object,
    )
    reminder_bar_colors[target_position] = rag_demo_config['target_color']
    axes[1].bar(
        rag_positions,
        rag_demo_scores['reminder_scores'],
        color=reminder_bar_colors,
    )
    axes[1].set_title(f'4b. Reminder query key (w={rag_demo_config["temporal_weight"]})')
    axes[1].set_xlabel('Previous trial (N−1 to 1)')
    axes[1].set_ylabel('Retrieval probability')
    rag_tick_positions = [0, len(rag_positions) - 1]
    rag_tick_labels = [f'N−1 ({rag_demo_scores["prior_trials"][0]})', '1']
    for axis in axes:
        axis.axhline(0.0, color='black', linewidth=1, alpha=0.4)
        axis.set_xticks(rag_tick_positions, rag_tick_labels)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    fig.legend(
        handles=[
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=rag_demo_config['target_color'],
                label='Reminder-matched episode',
            ),
        ],
        frameon=False,
        loc='lower center',
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(visual_dir / 'reminder_retrieval_causal_chain_panel_4.pdf', bbox_inches='tight')
    plt.close(fig)


def run_analysis_tables(
    *,
    reminder_table: pd.DataFrame,
    reward_table: pd.DataFrame,
    output_dir: Path,
    n_boot: int,
    random_state: int,
    grid: dict[str, list[Any]],
    plot_aesthetics: dict[str, Any] | None = None,
) -> None:
    """
    Compute, export, and visualize the complete reminder analysis.

    Args:
        reminder_table (pd.DataFrame): Raw reminder-level records.
        reward_table (pd.DataFrame): Agent-level total rewards.
        output_dir (Path): Analysis output directory.
        n_boot (int): Number of environment bootstrap resamples.
        random_state (int): Bootstrap seed.
        grid (dict[str, list[Any]]): Simulation grid that defines analysis conditions.
        plot_aesthetics (dict[str, Any] | None): Figure styling controls.
    """
    visual_dir = output_dir / 'visuals'
    visual_dir.mkdir(parents=True, exist_ok=True)

    reminder_metrics = add_reminder_metrics(reminder_table)
    lag_slopes = compute_environment_lag_slopes(reminder_metrics)
    environment_cells = compute_environment_cell_table(reminder_metrics, reward_table)
    continuous_lag_stats = compute_continuous_lag_stats(
        reminder_metrics,
        n_boot=n_boot,
        random_state=random_state,
    )
    planned_contrasts = compute_planned_contrasts(
        lag_slopes,
        environment_cells,
        grid=grid,
        n_boot=n_boot,
        random_state=random_state,
    )
    lag_band_stats = compute_lag_band_stats(
        reminder_metrics,
        n_boot=n_boot,
        random_state=random_state,
    )
    cell_stats = summarize_grouped_metric(
        environment_cells,
        group_columns=['sampling_budget', 'temporal_weight', 'sample_replacement'],
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
    reward_summary = summarize_grouped_metric(
        environment_cells,
        group_columns=['sampling_budget', 'temporal_weight', 'sample_replacement'],
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
    plot_causal_chain(
        lag_band_stats=lag_band_stats,
        continuous_lag_stats=continuous_lag_stats,
        cell_stats=cell_stats,
        grid=grid,
        plot_aesthetics=plot_aesthetics,
        savepath=visual_dir / 'reminder_retrieval_causal_chain.pdf',
    )
    plot_causal_chain_panels(
        lag_band_stats=lag_band_stats,
        continuous_lag_stats=continuous_lag_stats,
        cell_stats=cell_stats,
        grid=grid,
        plot_aesthetics=plot_aesthetics,
        visual_dir=visual_dir,
    )


def run_analysis(args: argparse.Namespace) -> None:
    """
    Load configured simulation results and run the reminder analysis.

    Args:
        args (argparse.Namespace): Parsed analysis arguments.
    """
    config = resolve_analysis_config(args.config)
    input_dir = getattr(args, 'input_dir', None) or get_replacement_output_dir(
        root_dir=filepaths.EXPERIMENTS_DIR,
        output_name=config['experiment_name'],
        sample_replacement=config['sample_replacement'],
    )
    reminder_table, reward_table = load_experiment_tables(
        experiment_dir=input_dir,
        expected_grid=config['grid'],
        n_environments=config['n_environments'],
        n_agent_simulations=config['n_agent_simulations'],
    )
    output_dir = getattr(args, 'output_dir', None) or get_replacement_output_dir(
        root_dir=filepaths.OUTPUT_DIR,
        output_name=config['output_dir_name'],
        sample_replacement=config['sample_replacement'],
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
    )


def main(argv: list[str] | None = None) -> None:
    """
    Run the configured Experiment 1.2 analysis.

    Args:
        argv (list[str] | None): Optional command-line arguments.
    """
    run_analysis(parse_args(argv))


if __name__ == '__main__':
    main()
