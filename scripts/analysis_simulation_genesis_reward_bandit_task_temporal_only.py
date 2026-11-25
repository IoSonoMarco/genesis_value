"""
Analyze the temporal-only GENESIS reward-bandit simulation.

The analysis compresses agent trial logs into environment-level summaries, then
plots how episodic sampling budget changes reward collection as a function of
the temporal predictability of each reward environment. The script keeps cached
CSV tables for the expensive log-flattening steps and saves publication-ready
PDF figures for the separate panels and the unified multi-panel summary.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, str(Path.cwd()))
warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from scipy.signal import savgol_filter
from scipy.stats import linregress, pearsonr, spearmanr, ttest_rel
from sklearn.metrics import root_mean_squared_error

from utils import filepaths
from utils.metrics import early_to_late_direction_score
from utils.simulation_config import get_replacement_output_dir, load_simulation_config


DEFAULT_ANALYSIS_CONFIG_PATH = (
    filepaths.ROOT / 'configs' / 'analysis_simulation_genesis_reward_bandit_task_temporal_only.yaml'
)
DEFAULT_BUDGET_COLORS = (
    '#2E338E',
    '#92297D',
    '#EA2264',
    '#F78D60',
    '#86BCBD',
    '#408175',
)
DEFAULT_ENV_CLASS_LABELS = {
    'Low': 'Low predictability',
    'Mid': 'Mid predictability',
    'High': 'High predictability',
}
DEFAULT_ENV_CLASS_COLORS = {
    'Low': '#2E338E',
    'Mid': '#EA2264',
    'High': '#408175',
}
LOW_BUDGET_REFERENCE = ('1', '5')
MID_BUDGET_REFERENCE = ('10', '20')
LARGE_BUDGET_REFERENCE = ('50', 'all')
PROBABILITY_REFERENCE_COLOR = '#808080'
PROBABILITY_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)


def default_analysis_config() -> dict[str, Any]:
    """
    Build default controls for the temporal-only analysis.

    Returns:
        dict[str, Any]: Analysis defaults used when the YAML omits optional fields.
    """
    return {
        'experiment_name': 'simulation_genesis_reward_bandit_task_temporal_only',
        'output_dir_name': 'analysis_simulation_genesis_reward_bandit_task_temporal_only',
        'save_figures': True,
        'example_environments': {
            'low': {
                'score_window': [0.1, 0.2],
                'seed': 44,
            },
            'high': {
                'score_window': [0.75, 0.9],
                'seed': 45,
            },
        },
        'sampling_budget_colors': {
            str(index): color
            for index, color in zip(['1', '5', '10', '20', '50', 'all'], DEFAULT_BUDGET_COLORS)
        },
        'environment_classes': {
            'classes': ['Low', 'Mid', 'High'],
            'labels': dict(DEFAULT_ENV_CLASS_LABELS),
            'colors': dict(DEFAULT_ENV_CLASS_COLORS),
        },
    }


def merge_nested_config(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """
    Merge YAML overrides into nested default analysis controls.

    Args:
        defaults (Mapping[str, Any]): Default configuration tree.
        overrides (Mapping[str, Any]): User-provided YAML configuration tree.

    Returns:
        dict[str, Any]: Configuration tree with user values replacing defaults.
    """
    merged = dict(defaults)

    for key, value in overrides.items():
        default_value = merged.get(key)

        if isinstance(default_value, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_nested_config(default_value, value)
        else:
            merged[key] = value

    return merged


def load_analysis_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load analysis controls from YAML and fill omitted optional fields.

    Args:
        config_path (str | Path): YAML file with analysis-specific controls.

    Returns:
        dict[str, Any]: Complete analysis configuration.
    """
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    return merge_nested_config(default_analysis_config(), config)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse temporal-only analysis arguments.

    Args:
        argv (list[str] | None): Optional command-line argument list.

    Returns:
        argparse.Namespace: Parsed analysis arguments.
    """
    parser = argparse.ArgumentParser(description='Analyze temporal-only GENESIS reward-bandit outputs.')
    parser.add_argument('--config', type=Path, default=DEFAULT_ANALYSIS_CONFIG_PATH)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--rebuild-cache', action='store_true')
    parser.add_argument('--no-save-figures', action='store_true')

    return parser.parse_args(argv)


def discover_experiments(experiment_dir: Path) -> tuple[pd.DataFrame, Callable[..., dict[str, Any]]]:
    """
    Index simulation grid files and return a loader for matching grid cells.

    Args:
        experiment_dir (Path): Directory containing one `.npy` file per simulation grid cell.

    Returns:
        tuple[pd.DataFrame, Callable[..., dict[str, Any]]]: Grid metadata and a loader keyed by grid values.
    """
    rows = []

    for filepath in experiment_dir.iterdir():
        filename = filepath.name
        sampling_temperature_match = re.search(r'sampling_temperature-([^_]+)', filename)
        rows.append({
            'sampling_budget': filename.split('sampling_budget')[-1].split('_')[0][1:],
            'temporal_weight': float(filename.split('temporal_weight')[-1].split('_')[0][1:].replace('p', '.')),
            'sample_replacement': re.search(r'sample_replacement-(true|false)', filename).group(1) == 'true',
            'sampling_temperature': (
                np.nan
                if sampling_temperature_match is None
                else float(sampling_temperature_match.group(1).replace('p', '.'))
            ),
            'filepath': filepath,
        })

    df_experiments = pd.DataFrame(rows)

    def load_experiment(
        *,
        sampling_budget: str,
        temporal_weight: float = 1.0,
        sample_replacement: bool = False,
        sampling_temperature: float | None = None,
    ) -> dict[str, Any]:
        query = (
            (df_experiments.sampling_budget == str(sampling_budget))
            & (df_experiments.temporal_weight == temporal_weight)
            & (df_experiments.sample_replacement == sample_replacement)
        )
        if sampling_temperature is not None:
            query &= df_experiments.sampling_temperature == sampling_temperature
        filepath = df_experiments.loc[query, 'filepath'].iloc[0]
        data = np.load(filepath, allow_pickle=True).item()

        return data

    return df_experiments, load_experiment


def build_full_table(
    *,
    df_experiments: pd.DataFrame,
    load_experiment: Callable[..., dict[str, Any]],
    savepath: Path,
    rebuild_cache: bool,
) -> pd.DataFrame:
    """
    Flatten nested simulation logs into one trial-level table.

    Each simulation output stores environments, and each environment stores many
    agent runs. This table repeats the environment reward-probability trajectory
    across all runs and appends the agent seed, environment seed, environment id,
    and sampling-budget label to each trial row. The resulting table is the
    common source for all reward, policy-adherence, and example-environment plots.

    Args:
        df_experiments (pd.DataFrame): Experiment grid metadata.
        load_experiment (Callable[..., dict[str, Any]]): Loader returned by `discover_experiments`.
        savepath (Path): CSV cache path.
        rebuild_cache (bool): Whether to ignore and overwrite an existing cache.

    Returns:
        pd.DataFrame: Trial-level simulation log table.
    """
    if savepath.exists() and not rebuild_cache:
        df_full = pd.read_csv(savepath, dtype={'sampling_budget': str})
        return df_full

    rows = []

    for experiment in df_experiments.itertuples(index=False):
        data = np.load(experiment.filepath, allow_pickle=True).item()

        for environment in data['environments']:
            p_win_left = [trial['p_win_left'] for trial in environment['schedule']]

            for agent_run in environment['agent_runs']:
                agent_simulation_id, agent_seed, simulation_log = agent_run.values()
                df_run = pd.DataFrame(simulation_log)
                df_run['p_win_left'] = p_win_left
                df_run['agent_seed'] = agent_seed
                df_run['environment_seed'] = environment['environment_seed']
                df_run['agent_simulation_id'] = agent_simulation_id
                df_run['sampling_budget'] = experiment.sampling_budget
                df_run['sampling_temperature'] = experiment.sampling_temperature
                df_run['environment_id'] = environment['environment_id']
                rows.append(df_run)

    df_full = pd.concat(rows, axis=0).drop('metadata', axis=1).reset_index(drop=True)
    df_full.sampling_budget = df_full.sampling_budget.astype(str)
    df_full.to_csv(savepath, index=False)

    return df_full


def build_env_score_table(
    *,
    df_full: pd.DataFrame,
    savepath: Path,
    rebuild_cache: bool,
) -> pd.DataFrame:
    """
    Synthesize environment-level reward and policy summaries.

    The environment score is the average early-to-late direction score computed
    across early windows of 20, 30, and 40 trials. Larger values indicate that
    early reward-probability direction is more predictive of later reward
    direction. For each environment and sampling budget, total reward is the
    agent-averaged sum of rewards, while action-value adherence is the negative
    RMSE between the smoothed LEFT-minus-RIGHT value estimate and the true left
    win probability trajectory.

    Args:
        df_full (pd.DataFrame): Trial-level simulation log table.
        savepath (Path): CSV cache path.
        rebuild_cache (bool): Whether to ignore and overwrite an existing cache.

    Returns:
        pd.DataFrame: Environment-by-budget summary table.
    """
    if savepath.exists() and not rebuild_cache:
        df_env_score = pd.read_csv(savepath, dtype={'sampling_budget': str})
        return df_env_score

    rows = []

    for env_id in df_full.environment_id.unique():
        df_env = df_full[df_full.environment_id == env_id]
        assert df_env.groupby('trial_id').p_win_left.std().mean() == 0, 'problem with p_win_left consistency'

        p_win_left = df_env.groupby('trial_id').p_win_left.mean().values
        env_score = np.mean([early_to_late_direction_score(p_win_left, N=n_trials) for n_trials in [20, 30, 40]])

        for sampling_budget_label, sampling_budget in enumerate(df_env.sampling_budget.unique()):
            df_budget = df_env[df_env.sampling_budget == sampling_budget]
            n_trials = df_budget.trial_id.nunique()

            reward = df_budget.reward.values.reshape(-1, n_trials)
            total_reward = np.sum(reward, axis=-1).mean()

            action_values = (
                df_budget.action_value_left - df_budget.action_value_right
            ).values.reshape(-1, n_trials)
            action_values = np.array([
                pd.Series(agent_values).rolling(window=5).mean()
                for agent_values in action_values
            ]).mean(axis=0)

            valid = np.isfinite(action_values)
            action_value_adherence = -root_mean_squared_error(action_values[valid], p_win_left[valid])

            rows.append({
                'env_id': env_id,
                'env_score': env_score,
                'sampling_budget': sampling_budget,
                'sampling_budget_label': sampling_budget_label,
                'total_reward': total_reward,
                'action_value_adherence': action_value_adherence,
                'p_win_left': p_win_left,
            })

    df_env_score = pd.DataFrame(rows)
    df_env_score.sampling_budget = df_env_score.sampling_budget.astype(str)
    df_env_score.to_csv(savepath, index=False)

    return df_env_score


def select_example_environment(
    *,
    df_env_scores: pd.DataFrame,
    score_window: Sequence[float],
    seed: int,
) -> int:
    """
    Select one example environment from a configured score window.

    Args:
        df_env_scores (pd.DataFrame): Table with one row per environment and score.
        score_window (Sequence[float]): Inclusive lower and upper score bounds.
        seed (int): Random seed used to choose among candidate environments.

    Returns:
        int: Selected environment id.
    """
    low, high = score_window
    candidates = (
        df_env_scores
        .query('env_score >= @low & env_score <= @high')
        .sort_values('env_id')
        .env_id
        .to_numpy(dtype=int)
    )

    if len(candidates) == 0:
        raise ValueError(f'No environments found in score window [{low}, {high}].')

    rng = np.random.default_rng(seed)
    env_id = rng.choice(candidates)

    return int(env_id)


def resolve_budget_colors(
    *,
    budget_order: Sequence[str],
    configured_colors: Mapping[str, str],
) -> dict[str, str]:
    """
    Resolve a color for each sampling-budget level.

    Args:
        budget_order (Sequence[str]): Sampling-budget labels in plotting order.
        configured_colors (Mapping[str, str]): YAML palette keyed by sampling-budget label.

    Returns:
        dict[str, str]: Color palette keyed by sampling-budget label.
    """
    budget_colors = {}

    for index, sampling_budget in enumerate(budget_order):
        fallback_color = DEFAULT_BUDGET_COLORS[index % len(DEFAULT_BUDGET_COLORS)]
        budget_colors[str(sampling_budget)] = configured_colors.get(str(sampling_budget), fallback_color)

    return budget_colors


def bootstrap_corr_ci(
    *,
    x: Sequence[float],
    y: Sequence[float],
    n_boot: int = 5000,
    ci: int = 95,
    random_state: int = 123,
) -> tuple[float, float]:
    """
    Estimate a bootstrap confidence interval for Pearson correlation.

    Args:
        x (Sequence[float]): First variable.
        y (Sequence[float]): Second variable.
        n_boot (int): Number of bootstrap resamples.
        ci (int): Confidence interval width in percent.
        random_state (int): Seed for bootstrap resampling.

    Returns:
        tuple[float, float]: Lower and upper confidence bounds.
    """
    rng = np.random.default_rng(random_state)
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[valid]
    y_array = y_array[valid]

    if len(x_array) < 3 or np.std(x_array) == 0 or np.std(y_array) == 0:
        return np.nan, np.nan

    bootstraps = []
    n_samples = len(x_array)

    for _ in range(n_boot):
        sample_index = rng.integers(0, n_samples, size=n_samples)

        if np.std(x_array[sample_index]) == 0 or np.std(y_array[sample_index]) == 0:
            continue

        bootstraps.append(np.corrcoef(x_array[sample_index], y_array[sample_index])[0, 1])

    if len(bootstraps) == 0:
        return np.nan, np.nan

    alpha = (100 - ci) / 2
    low, high = np.percentile(bootstraps, [alpha, 100 - alpha])

    return float(low), float(high)


def sem(values: Sequence[float]) -> float:
    """
    Compute the standard error of finite values.

    Args:
        values (Sequence[float]): Input values.

    Returns:
        float: Standard error of the mean.
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]

    if len(array) <= 1:
        return np.nan

    return float(np.std(array, ddof=1) / np.sqrt(len(array)))


def build_reward_analysis_table(df_env_score: pd.DataFrame) -> pd.DataFrame:
    """
    Select the environment-score and reward columns used by reward plots.

    Args:
        df_env_score (pd.DataFrame): Environment-by-budget summary table.

    Returns:
        pd.DataFrame: Reward analysis table sorted by environment and budget.
    """
    df_reward_analysis = df_env_score[['env_id', 'env_score', 'sampling_budget', 'total_reward']]
    df_reward_analysis = df_reward_analysis.sort_values(['env_id', 'sampling_budget']).reset_index(drop=True)

    return df_reward_analysis


def compute_example_environment_curves(
    *,
    df_full: pd.DataFrame,
    env_id: int,
    budget_order: Sequence[str],
) -> tuple[pd.Series, np.ndarray, list[dict[str, Any]]]:
    """
    Compute smoothed reward and action-value curves for one environment.

    The cumulative reward curve is averaged across agent simulations for each
    sampling budget. The action-value curve tracks the rolling mean of the
    sampled LEFT-minus-RIGHT value estimate. Both curves share the environment's
    true left win-probability trajectory as a dashed reference in the plot.

    Args:
        df_full (pd.DataFrame): Trial-level simulation log table.
        env_id (int): Environment id to summarize.
        budget_order (Sequence[str]): Sampling-budget order for plotting.

    Returns:
        tuple[pd.Series, np.ndarray, list[dict[str, Any]]]: Win-probability trajectory, trial ids, and curves.
    """
    df_env = df_full[df_full.environment_id == env_id]
    assert df_env.groupby('trial_id').p_win_left.std().mean() == 0, 'problem with p_win_left consistency'

    p_win_left = df_env.groupby('trial_id').p_win_left.mean()
    trial_ids = df_env.trial_id.sort_values().unique()
    curves = []

    for sampling_budget in budget_order:
        df_budget = df_env[df_env.sampling_budget == sampling_budget]
        values = (
            df_budget.action_value_left - df_budget.action_value_right
        ).values.reshape(-1, len(trial_ids))
        values = np.array([
            pd.Series(agent_values).rolling(window=5).mean()
            for agent_values in values
        ])

        rewards = df_budget.reward.values.reshape(-1, len(trial_ids))
        rewards = savgol_filter(np.cumsum(rewards, axis=-1), 5, 1, axis=-1)

        curves.append({
            'sampling_budget': sampling_budget,
            'value_mean': values.mean(axis=0),
            'value_sem': values.std(axis=0) / np.sqrt(values.shape[0]),
            'reward_mean': rewards.mean(axis=0),
            'reward_sem': rewards.std(axis=0) / np.sqrt(rewards.shape[0]),
        })

    return p_win_left, trial_ids, curves


def style_axis(ax: Axes) -> None:
    """
    Apply the shared minimal plot style.

    Args:
        ax (Axes): Matplotlib axes to style.
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', size=0)


def plot_example_environment_panel(
    *,
    axes: Sequence[Axes],
    p_win_left: pd.Series,
    trial_ids: np.ndarray,
    curves: Sequence[Mapping[str, Any]],
    budget_colors: Mapping[str, str],
    reward_ylim: tuple[float, float] | None = None,
) -> None:
    """
    Plot cumulative reward and action-value tracking for one environment.

    Args:
        axes (Sequence[Axes]): Left reward axis and right action-value axis.
        p_win_left (pd.Series): True left win-probability trajectory.
        trial_ids (np.ndarray): Trial id values.
        curves (Sequence[Mapping[str, Any]]): Precomputed curves for each budget.
        budget_colors (Mapping[str, str]): Colors keyed by sampling-budget label.
        reward_ylim (tuple[float, float] | None): Optional cumulative-reward y-axis limits.
    """
    reward_ax, value_ax = axes
    n_nans = 0

    for curve in curves:
        sampling_budget = curve['sampling_budget']
        value_mean = curve['value_mean']
        value_sem = curve['value_sem']
        n_nans = max(n_nans, np.isnan(value_mean).sum())

        value_ax.plot(trial_ids, value_mean, color=budget_colors[sampling_budget], linewidth=2)
        value_ax.fill_between(
            trial_ids,
            value_mean - value_sem / 2,
            value_mean + value_sem / 2,
            alpha=0.2,
            color=budget_colors[sampling_budget],
        )

    value_reference_ax = value_ax.twinx()
    value_reference_ax.axhline(0.5, color=PROBABILITY_REFERENCE_COLOR, linewidth=0.75, alpha=0.5)
    value_reference_ax.plot(
        trial_ids,
        p_win_left,
        color=PROBABILITY_REFERENCE_COLOR,
        linestyle='dashed',
        alpha=0.5,
    )
    value_reference_ax.set_ylim(0.0, 1.0)
    value_reference_ax.set_yticks(PROBABILITY_TICKS)
    value_reference_ax.set_ylabel('P(Win Left)', color=PROBABILITY_REFERENCE_COLOR, fontsize=12)
    value_reference_ax.spines['right'].set_color(PROBABILITY_REFERENCE_COLOR)
    value_reference_ax.spines['top'].set_visible(False)
    value_reference_ax.spines['left'].set_visible(False)
    value_reference_ax.spines['bottom'].set_visible(False)
    value_reference_ax.tick_params(
        axis='y',
        colors=PROBABILITY_REFERENCE_COLOR,
        size=0,
        labelsize=8,
        labelrotation=90,
    )
    plt.setp(value_reference_ax.get_yticklabels(), horizontalalignment='center', verticalalignment='center')
    value_ax.set_ylabel('Action Value Left (Q(L) - Q(R))', fontsize=12)

    for curve in curves:
        sampling_budget = curve['sampling_budget']
        reward_mean = curve['reward_mean'].copy()
        reward_sem = curve['reward_sem'].copy()
        reward_mean[:n_nans] = np.nan
        reward_sem[:n_nans] = np.nan

        reward_ax.plot(trial_ids, reward_mean, color=budget_colors[sampling_budget], linewidth=2)
        reward_ax.fill_between(
            trial_ids,
            reward_mean - reward_sem / 10,
            reward_mean + reward_sem / 10,
            alpha=0.2,
            color=budget_colors[sampling_budget],
        )

    reward_reference_ax = reward_ax.twinx()
    reward_reference_ax.axhline(0.5, color=PROBABILITY_REFERENCE_COLOR, linewidth=0.75, alpha=0.5)
    reward_reference_ax.plot(
        trial_ids,
        p_win_left,
        color=PROBABILITY_REFERENCE_COLOR,
        linestyle='dashed',
        alpha=0.5,
    )
    reward_reference_ax.set_ylim(0.0, 1.0)
    reward_reference_ax.set_yticks(PROBABILITY_TICKS)
    reward_reference_ax.set_ylabel('P(Win Left)', color=PROBABILITY_REFERENCE_COLOR, fontsize=12)
    reward_reference_ax.spines['right'].set_color(PROBABILITY_REFERENCE_COLOR)
    reward_reference_ax.spines['top'].set_visible(False)
    reward_reference_ax.spines['left'].set_visible(False)
    reward_reference_ax.spines['bottom'].set_visible(False)
    reward_reference_ax.tick_params(
        axis='y',
        colors=PROBABILITY_REFERENCE_COLOR,
        size=0,
        labelsize=8,
        labelrotation=90,
    )
    plt.setp(reward_reference_ax.get_yticklabels(), horizontalalignment='center', verticalalignment='center')
    reward_ax.set_ylabel('Cumulative Reward', fontsize=12)
    if reward_ylim is not None:
        reward_ax.set_ylim(reward_ylim)

    for ax in axes:
        ax.set_xlabel('Trial', fontsize=12)
        style_axis(ax)


def add_budget_legend(
    *,
    fig: Figure,
    budget_order: Sequence[str],
    budget_colors: Mapping[str, str],
    y_anchor: float,
) -> None:
    """
    Add a shared sampling-budget legend to a figure.

    Args:
        fig (Figure): Figure receiving the legend.
        budget_order (Sequence[str]): Sampling-budget labels in legend order.
        budget_colors (Mapping[str, str]): Colors keyed by sampling-budget label.
        y_anchor (float): Legend y anchor in figure coordinates.
    """
    legend_handles = [
        Patch(facecolor=budget_colors[sampling_budget], edgecolor=budget_colors[sampling_budget], label=sampling_budget)
        for sampling_budget in budget_order
    ]

    fig.legend(
        handles=legend_handles,
        title='Sampling Budget',
        loc='upper center',
        bbox_to_anchor=(0.5, y_anchor),
        ncol=len(budget_order),
        frameon=False,
    )


def save_example_environment_plot(
    *,
    savepath: Path,
    p_win_left: pd.Series,
    trial_ids: np.ndarray,
    curves: Sequence[Mapping[str, Any]],
    budget_order: Sequence[str],
    budget_colors: Mapping[str, str],
    save_figures: bool,
    reward_ylim: tuple[float, float] | None = None,
) -> Figure:
    """
    Save a two-panel example-environment plot.

    Args:
        savepath (Path): PDF output path.
        p_win_left (pd.Series): True left win-probability trajectory.
        trial_ids (np.ndarray): Trial id values.
        curves (Sequence[Mapping[str, Any]]): Precomputed curves for each budget.
        budget_order (Sequence[str]): Sampling-budget order for legend entries.
        budget_colors (Mapping[str, str]): Colors keyed by sampling-budget label.
        save_figures (bool): Whether to save the figure.
        reward_ylim (tuple[float, float] | None): Optional cumulative-reward y-axis limits.

    Returns:
        Figure: Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    plot_example_environment_panel(
        axes=axes,
        p_win_left=p_win_left,
        trial_ids=trial_ids,
        curves=curves,
        budget_colors=budget_colors,
        reward_ylim=reward_ylim,
    )
    add_budget_legend(fig=fig, budget_order=budget_order, budget_colors=budget_colors, y_anchor=1.1)
    fig.tight_layout(pad=3, rect=[0, 0.08, 1, 1])

    if save_figures:
        fig.savefig(savepath, dpi=800, bbox_inches='tight')

    return fig


def compute_reward_correlation_stats(
    *,
    df_reward_analysis: pd.DataFrame,
    budget_order: Sequence[str],
) -> pd.DataFrame:
    """
    Correlate environment predictability with total reward for each budget.

    Args:
        df_reward_analysis (pd.DataFrame): Reward analysis table.
        budget_order (Sequence[str]): Sampling-budget order.

    Returns:
        pd.DataFrame: Correlation, bootstrap CI, and fitted-line statistics per budget.
    """
    rows = []

    for sampling_budget in budget_order:
        group = df_reward_analysis[df_reward_analysis.sampling_budget == sampling_budget]
        x = group['env_score'].to_numpy(dtype=float)
        y = group['total_reward'].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
            pearson_r, pearson_p = pearsonr(x, y)
            spearman_r, spearman_p = spearmanr(x, y)
            lr = linregress(x, y)
            ci_low, ci_high = bootstrap_corr_ci(x=x, y=y)
        else:
            pearson_r = np.nan
            pearson_p = np.nan
            spearman_r = np.nan
            spearman_p = np.nan
            lr = None
            ci_low = np.nan
            ci_high = np.nan

        rows.append({
            'sampling_budget': sampling_budget,
            'n_env': len(x),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'pearson_ci_low': ci_low,
            'pearson_ci_high': ci_high,
            'slope': np.nan if lr is None else lr.slope,
            'intercept': np.nan if lr is None else lr.intercept,
            'slope_p': np.nan if lr is None else lr.pvalue,
            'r_squared': pearson_r ** 2 if np.isfinite(pearson_r) else np.nan,
        })

    return pd.DataFrame(rows)


def plot_reward_correlation_panel(
    *,
    ax: Axes,
    df_reward_corr_stats: pd.DataFrame,
    budget_colors: Mapping[str, str],
) -> None:
    """
    Plot reward-environment correlation as a budget-ordered trend.

    Dots and confidence intervals use the sampling-budget palette. The fitted
    black line summarizes the monotonic trend across budget positions without
    implying that adjacent budget estimates are directly connected.

    Args:
        ax (Axes): Matplotlib axes.
        df_reward_corr_stats (pd.DataFrame): Correlation statistics per budget.
        budget_colors (Mapping[str, str]): Colors keyed by sampling-budget label.
    """
    xpos = np.arange(len(df_reward_corr_stats))

    for index, row in df_reward_corr_stats.iterrows():
        sampling_budget = row['sampling_budget']
        y_value = row['pearson_r']
        ci_low = row['pearson_ci_low']
        ci_high = row['pearson_ci_high']

        if np.isfinite(y_value):
            ax.scatter(
                index,
                y_value,
                color=budget_colors[sampling_budget],
                edgecolor=budget_colors[sampling_budget],
                s=42,
                zorder=3,
            )

        if np.isfinite(y_value) and np.isfinite(ci_low) and np.isfinite(ci_high):
            ax.errorbar(
                index,
                y_value,
                yerr=[[y_value - ci_low], [ci_high - y_value]],
                color=budget_colors[sampling_budget],
                ecolor=to_rgba(budget_colors[sampling_budget], alpha=0.45),
                elinewidth=7,
                capsize=0,
                zorder=2,
            )

    valid = np.isfinite(df_reward_corr_stats['pearson_r'].to_numpy(dtype=float))
    if valid.sum() >= 2:
        line_fit = linregress(xpos[valid], df_reward_corr_stats.loc[valid, 'pearson_r'])
        xline = np.linspace(xpos[valid].min(), xpos[valid].max(), 200)
        ax.plot(xline, line_fit.intercept + line_fit.slope * xline, color='black', linewidth=2)

    y_candidates = pd.concat([
        df_reward_corr_stats['pearson_r'],
        df_reward_corr_stats['pearson_ci_low'],
    ]).dropna()
    lower_ylim = max(-1.0, y_candidates.min() - 0.05)

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.4)
    ax.set_xticks(xpos)
    ax.set_xticklabels(df_reward_corr_stats['sampling_budget'])
    ax.set_xlabel('Sampling budget', fontsize=12)
    ax.set_ylabel('r(env score, total reward)', fontsize=12)
    ax.set_ylim(lower_ylim, 1.0)
    style_axis(ax)


def save_reward_correlation_plot(
    *,
    savepath: Path,
    df_reward_corr_stats: pd.DataFrame,
    budget_colors: Mapping[str, str],
    save_figures: bool,
) -> Figure:
    """
    Save the vertical reward-environment correlation plot.

    Args:
        savepath (Path): PDF output path.
        df_reward_corr_stats (pd.DataFrame): Correlation statistics per budget.
        budget_colors (Mapping[str, str]): Colors keyed by sampling-budget label.
        save_figures (bool): Whether to save the figure.

    Returns:
        Figure: Matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 6.0))
    plot_reward_correlation_panel(
        ax=ax,
        df_reward_corr_stats=df_reward_corr_stats,
        budget_colors=budget_colors,
    )
    fig.tight_layout()

    if save_figures:
        fig.savefig(savepath, dpi=800, bbox_inches='tight')

    return fig


def compute_budget_contrast(
    *,
    df_reward_analysis: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Sequence[str], Sequence[str], Sequence[str]]:
    """
    Compute low-minus-large budget reward advantage across environments.

    Args:
        df_reward_analysis (pd.DataFrame): Reward analysis table.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Sequence[str], Sequence[str], Sequence[str]]:
            Environment-level contrast table, single-row stats table, pivot table, and budget groups.
    """
    df_pivot = df_reward_analysis.pivot(index='env_id', columns='sampling_budget', values='total_reward')
    df_env_level = (
        df_reward_analysis[['env_id', 'env_score']]
        .drop_duplicates('env_id')
        .set_index('env_id')
    )
    df_contrast = df_env_level.copy()

    low_budgets = [budget for budget in LOW_BUDGET_REFERENCE if budget in df_pivot.columns]
    mid_budgets = [budget for budget in MID_BUDGET_REFERENCE if budget in df_pivot.columns]
    large_budgets = [budget for budget in LARGE_BUDGET_REFERENCE if budget in df_pivot.columns]

    df_contrast['low_budget_reward'] = df_pivot[low_budgets].mean(axis=1)
    df_contrast['mid_budget_reward'] = df_pivot[mid_budgets].mean(axis=1)
    df_contrast['large_budget_reward'] = df_pivot[large_budgets].mean(axis=1)
    df_contrast['low_minus_large_reward'] = (
        df_contrast['low_budget_reward'] - df_contrast['large_budget_reward']
    )

    if 'all' in df_pivot.columns:
        df_contrast['all_minus_low_reward'] = df_pivot['all'] - df_contrast['low_budget_reward']
    else:
        df_contrast['all_minus_low_reward'] = np.nan

    df_contrast = df_contrast.reset_index()
    x = df_contrast['env_score'].to_numpy(dtype=float)
    y = df_contrast['low_minus_large_reward'].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid]
    y_valid = y[valid]

    lr = linregress(x_valid, y_valid)
    pearson_r, pearson_p = pearsonr(x_valid, y_valid)
    spearman_r, spearman_p = spearmanr(x_valid, y_valid)

    df_contrast_stats = pd.DataFrame([{
        'contrast': 'mean(1,5) - mean(50,all)',
        'n_env': len(x_valid),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'slope': lr.slope,
        'intercept': lr.intercept,
        'slope_p': lr.pvalue,
        'r_squared': pearson_r ** 2,
    }])

    return df_contrast, df_contrast_stats, df_pivot, low_budgets, mid_budgets, large_budgets


def plot_budget_contrast_panel(
    *,
    ax: Axes,
    df_contrast: pd.DataFrame,
    df_contrast_stats: pd.DataFrame,
) -> None:
    """
    Plot whether low budgets outperform large budgets as predictability changes.

    Args:
        ax (Axes): Matplotlib axes.
        df_contrast (pd.DataFrame): Environment-level contrast table.
        df_contrast_stats (pd.DataFrame): Single-row fitted contrast statistics.
    """
    slope = df_contrast_stats.slope.iloc[0]
    intercept = df_contrast_stats.intercept.iloc[0]
    pearson_r = df_contrast_stats.pearson_r.iloc[0]
    pearson_p = df_contrast_stats.pearson_p.iloc[0]

    ax.scatter(
        df_contrast['env_score'],
        df_contrast['low_minus_large_reward'],
        color='black',
        alpha=0.7,
        s=35,
    )

    xline = np.linspace(df_contrast['env_score'].min(), df_contrast['env_score'].max(), 200)
    ax.plot(xline, intercept + slope * xline, color='#EA2264', linewidth=2.5)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Early-to-late direction score', fontsize=12)
    ax.set_ylabel('Reward advantage: mean(1,5) - mean(50,all)', fontsize=12)

    text = f'r = {pearson_r:.2f}\np = {pearson_p:.1e}\nslope = {slope:.2f}'
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
    style_axis(ax)


def save_budget_contrast_plot(
    *,
    savepath: Path,
    df_contrast: pd.DataFrame,
    df_contrast_stats: pd.DataFrame,
    save_figures: bool,
) -> Figure:
    """
    Save the low-minus-large budget advantage plot.

    Args:
        savepath (Path): PDF output path.
        df_contrast (pd.DataFrame): Environment-level contrast table.
        df_contrast_stats (pd.DataFrame): Single-row fitted contrast statistics.
        save_figures (bool): Whether to save the figure.

    Returns:
        Figure: Matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    plot_budget_contrast_panel(
        ax=ax,
        df_contrast=df_contrast,
        df_contrast_stats=df_contrast_stats,
    )
    fig.tight_layout()

    if save_figures:
        fig.savefig(savepath, dpi=800, bbox_inches='tight')

    return fig


def build_env_score_classes(
    *,
    df_reward_analysis: pd.DataFrame,
    class_names: Sequence[str],
    class_labels: Mapping[str, str],
) -> pd.DataFrame:
    """
    Assign environments to score quantile classes.

    Args:
        df_reward_analysis (pd.DataFrame): Reward analysis table.
        class_names (Sequence[str]): Ordered class keys, usually Low/Mid/High or Low/High.
        class_labels (Mapping[str, str]): Display labels keyed by class key.

    Returns:
        pd.DataFrame: One row per environment with ordered class labels.
    """
    labels = [class_labels[class_name] for class_name in class_names]
    df_env_bins = df_reward_analysis[['env_id', 'env_score']].drop_duplicates('env_id').copy()
    df_env_bins['env_score_class'] = pd.qcut(df_env_bins['env_score'], q=len(labels), labels=labels)
    df_env_bins['env_score_class'] = pd.Categorical(
        df_env_bins['env_score_class'],
        categories=labels,
        ordered=True,
    )

    return df_env_bins


def compute_env_class_stats(
    *,
    df_reward_analysis: pd.DataFrame,
    df_env_bins: pd.DataFrame,
    df_pivot: pd.DataFrame,
    budget_order: Sequence[str],
    low_budgets: Sequence[str],
    large_budgets: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Summarize rewards, winning budgets, and low-large contrasts by score class.

    Args:
        df_reward_analysis (pd.DataFrame): Reward analysis table.
        df_env_bins (pd.DataFrame): Environment score classes.
        df_pivot (pd.DataFrame): Environment-by-budget total reward table.
        budget_order (Sequence[str]): Sampling-budget order.
        low_budgets (Sequence[str]): Low-budget group.
        large_budgets (Sequence[str]): Large-budget group.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Reward stats, winner counts, and paired contrasts.
    """
    class_labels = df_env_bins['env_score_class'].cat.categories.tolist()
    df_class = df_reward_analysis.merge(
        df_env_bins[['env_id', 'env_score_class']],
        on='env_id',
        how='left',
    )
    df_class['env_score_class'] = pd.Categorical(
        df_class['env_score_class'],
        categories=class_labels,
        ordered=True,
    )

    df_class_stats = (
        df_class
        .groupby(['env_score_class', 'sampling_budget'], observed=True)
        .agg(
            mean_reward=('total_reward', 'mean'),
            sem_reward=('total_reward', sem),
            n_env=('env_id', 'nunique'),
        )
        .reset_index()
    )

    df_winners = (
        df_reward_analysis
        .sort_values(['env_id', 'total_reward'], ascending=[True, False])
        .groupby('env_id')
        .head(1)
        .merge(df_env_bins[['env_id', 'env_score_class']], on='env_id', how='left')
    )
    df_winner_counts = (
        df_winners
        .groupby(['env_score_class', 'sampling_budget'], observed=True)
        .size()
        .reset_index(name='winner_count')
    )

    full_index = pd.MultiIndex.from_product(
        [class_labels, budget_order],
        names=['env_score_class', 'sampling_budget'],
    )
    df_winner_counts = (
        df_winner_counts
        .set_index(['env_score_class', 'sampling_budget'])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    contrast_rows = []
    for env_score_class, group_ids in df_env_bins.groupby('env_score_class', observed=True):
        env_ids = group_ids['env_id'].values
        group_pivot = df_pivot.loc[df_pivot.index.isin(env_ids)].copy()
        low = group_pivot[low_budgets].mean(axis=1)
        large = group_pivot[large_budgets].mean(axis=1)
        diff = low - large

        if len(diff.dropna()) >= 2:
            tstat, pval = ttest_rel(low, large, nan_policy='omit')
        else:
            tstat, pval = np.nan, np.nan

        contrast_rows.append({
            'env_score_class': env_score_class,
            'n_env': len(diff.dropna()),
            'low_budget_mean': float(np.nanmean(low)),
            'large_budget_mean': float(np.nanmean(large)),
            'low_minus_large_mean': float(np.nanmean(diff)),
            'low_minus_large_sem': sem(diff),
            'paired_t': tstat,
            'paired_p': pval,
        })

    df_class_contrast_stats = pd.DataFrame(contrast_rows)

    return df_class_stats, df_winner_counts, df_class_contrast_stats


def plot_env_class_reward_panel(
    *,
    ax: Axes,
    df_class_stats: pd.DataFrame,
    budget_order: Sequence[str],
    class_labels: Sequence[str],
    class_colors: Mapping[str, str],
) -> None:
    """
    Plot mean total reward across sampling budgets for each environment class.

    Args:
        ax (Axes): Matplotlib axes.
        df_class_stats (pd.DataFrame): Reward summary by class and budget.
        budget_order (Sequence[str]): Sampling-budget order.
        class_labels (Sequence[str]): Ordered class display labels.
        class_colors (Mapping[str, str]): Colors keyed by class display label.
    """
    for env_score_class in class_labels:
        group = df_class_stats[df_class_stats.env_score_class == env_score_class].copy()
        group = group.set_index('sampling_budget').loc[budget_order].reset_index()
        xpos = np.arange(len(group))

        ax.plot(
            xpos,
            group['mean_reward'],
            marker='o',
            linewidth=2.5,
            color=class_colors[env_score_class],
            label=env_score_class,
        )
        ax.fill_between(
            xpos,
            group['mean_reward'] - group['sem_reward'],
            group['mean_reward'] + group['sem_reward'],
            color=class_colors[env_score_class],
            alpha=0.18,
        )

    ax.set_xticks(np.arange(len(budget_order)))
    ax.set_xticklabels(budget_order)
    ax.set_xlabel('Sampling budget', fontsize=12)
    ax.set_ylabel('Total reward', fontsize=12)
    ax.legend(frameon=False, title='Environment class')
    style_axis(ax)


def save_env_class_reward_plot(
    *,
    savepath: Path,
    df_class_stats: pd.DataFrame,
    budget_order: Sequence[str],
    class_labels: Sequence[str],
    class_colors: Mapping[str, str],
    save_figures: bool,
) -> Figure:
    """
    Save reward profiles by environment class and sampling budget.

    Args:
        savepath (Path): PDF output path.
        df_class_stats (pd.DataFrame): Reward summary by class and budget.
        budget_order (Sequence[str]): Sampling-budget order.
        class_labels (Sequence[str]): Ordered class display labels.
        class_colors (Mapping[str, str]): Colors keyed by class display label.
        save_figures (bool): Whether to save the figure.

    Returns:
        Figure: Matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
    plot_env_class_reward_panel(
        ax=ax,
        df_class_stats=df_class_stats,
        budget_order=budget_order,
        class_labels=class_labels,
        class_colors=class_colors,
    )
    fig.tight_layout()

    if save_figures:
        fig.savefig(savepath, dpi=800, bbox_inches='tight')

    return fig


def save_unified_plot(
    *,
    savepath: Path,
    low_example: tuple[pd.Series, np.ndarray, list[dict[str, Any]]],
    high_example: tuple[pd.Series, np.ndarray, list[dict[str, Any]]],
    df_reward_corr_stats: pd.DataFrame,
    df_contrast: pd.DataFrame,
    df_contrast_stats: pd.DataFrame,
    df_class_stats: pd.DataFrame,
    budget_order: Sequence[str],
    budget_colors: Mapping[str, str],
    class_labels: Sequence[str],
    class_colors: Mapping[str, str],
    save_figures: bool,
) -> Figure:
    """
    Save a unified five-panel summary figure.

    Row one combines the three statistical summaries. The correlation panel is
    intentionally narrower because it is a compact budget-ordered estimate, while
    the low-vs-large scatter and class-by-budget reward profile need more width.
    Rows two and three show the low- and high-score example environments across
    the full figure width.

    Args:
        savepath (Path): PDF output path.
        low_example (tuple[pd.Series, np.ndarray, list[dict[str, Any]]]): Low-score example curves.
        high_example (tuple[pd.Series, np.ndarray, list[dict[str, Any]]]): High-score example curves.
        df_reward_corr_stats (pd.DataFrame): Correlation statistics per budget.
        df_contrast (pd.DataFrame): Environment-level low-large contrast table.
        df_contrast_stats (pd.DataFrame): Single-row fitted contrast statistics.
        df_class_stats (pd.DataFrame): Reward summary by class and budget.
        budget_order (Sequence[str]): Sampling-budget order.
        budget_colors (Mapping[str, str]): Colors keyed by sampling-budget label.
        class_labels (Sequence[str]): Ordered class display labels.
        class_colors (Mapping[str, str]): Colors keyed by class display label.
        save_figures (bool): Whether to save the figure.

    Returns:
        Figure: Matplotlib figure.
    """
    fig = plt.figure(figsize=(13, 12))
    outer_grid = GridSpec(
        nrows=3,
        ncols=1,
        height_ratios=[1.05, 1.0, 1.0],
        hspace=0.48,
        figure=fig,
    )
    top_grid = GridSpecFromSubplotSpec(
        nrows=1,
        ncols=3,
        subplot_spec=outer_grid[0],
        width_ratios=[0.72, 1.35, 1.35],
        wspace=0.42,
    )

    correlation_ax = fig.add_subplot(top_grid[0])
    contrast_ax = fig.add_subplot(top_grid[1])
    class_ax = fig.add_subplot(top_grid[2])
    low_grid = GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=outer_grid[1], wspace=0.25)
    high_grid = GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=outer_grid[2], wspace=0.25)
    low_axes = [fig.add_subplot(low_grid[0]), fig.add_subplot(low_grid[1])]
    high_axes = [fig.add_subplot(high_grid[0]), fig.add_subplot(high_grid[1])]

    plot_reward_correlation_panel(
        ax=correlation_ax,
        df_reward_corr_stats=df_reward_corr_stats,
        budget_colors=budget_colors,
    )
    plot_budget_contrast_panel(
        ax=contrast_ax,
        df_contrast=df_contrast,
        df_contrast_stats=df_contrast_stats,
    )
    contrast_ax.set_ylabel('Reward advantage\nmean(1,5) - mean(50,all)', fontsize=11, labelpad=4)
    plot_env_class_reward_panel(
        ax=class_ax,
        df_class_stats=df_class_stats,
        budget_order=budget_order,
        class_labels=class_labels,
        class_colors=class_colors,
    )
    plot_example_environment_panel(
        axes=low_axes,
        p_win_left=low_example[0],
        trial_ids=low_example[1],
        curves=low_example[2],
        budget_colors=budget_colors,
    )
    plot_example_environment_panel(
        axes=high_axes,
        p_win_left=high_example[0],
        trial_ids=high_example[1],
        curves=high_example[2],
        budget_colors=budget_colors,
    )
    add_budget_legend(fig=fig, budget_order=budget_order, budget_colors=budget_colors, y_anchor=0.995)

    for panel_label, ax in zip(['A', 'B', 'C', 'D', 'E'], [
        correlation_ax,
        contrast_ax,
        class_ax,
        low_axes[0],
        high_axes[0],
    ]):
        ax.text(
            -0.12,
            1.08,
            panel_label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            va='top',
            ha='left',
        )

    if save_figures:
        fig.savefig(savepath, dpi=800, bbox_inches='tight')

    return fig


def resolve_class_plot_config(config: Mapping[str, Any]) -> tuple[list[str], list[str], dict[str, str]]:
    """
    Resolve ordered environment-class labels and colors.

    Args:
        config (Mapping[str, Any]): Complete analysis configuration.

    Returns:
        tuple[list[str], list[str], dict[str, str]]: Class keys, display labels, and display-label colors.
    """
    env_class_config = config['environment_classes']
    class_names = env_class_config['classes']
    class_labels = [env_class_config['labels'][class_name] for class_name in class_names]
    class_colors = {
        env_class_config['labels'][class_name]: env_class_config['colors'][class_name]
        for class_name in class_names
    }

    return class_names, class_labels, class_colors


def run_analysis_without_replacement(args: argparse.Namespace) -> None:
    """
    Run the complete temporal-only analysis pipeline.

    Args:
        args (argparse.Namespace): Parsed analysis arguments.
    """
    config = load_analysis_config(args.config)
    save_figures = bool(config['save_figures']) and not args.no_save_figures
    simulation_config_path = Path(config['simulation_config_path'])
    if not simulation_config_path.is_absolute():
        simulation_config_path = filepaths.ROOT / simulation_config_path
    simulation_config = load_simulation_config(simulation_config_path)

    output_dir = args.output_dir or get_replacement_output_dir(
        root_dir=filepaths.OUTPUT_DIR,
        output_name=config['output_dir_name'],
        sample_replacement=simulation_config['sample_replacement'],
    )
    visual_dir = output_dir / 'visuals'
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    experiment_dir = args.input_dir or get_replacement_output_dir(
        root_dir=filepaths.EXPERIMENTS_DIR,
        output_name=simulation_config['experiment_name'],
        sample_replacement=simulation_config['sample_replacement'],
    )
    df_experiments, load_experiment = discover_experiments(experiment_dir)
    df_full = build_full_table(
        df_experiments=df_experiments,
        load_experiment=load_experiment,
        savepath=output_dir / 'full_table.csv',
        rebuild_cache=args.rebuild_cache,
    )
    df_env_score = build_env_score_table(
        df_full=df_full,
        savepath=output_dir / 'env_score-vs-policy-table.csv',
        rebuild_cache=args.rebuild_cache,
    )
    budget_order = df_env_score.sampling_budget.unique().tolist()
    budget_colors = resolve_budget_colors(
        budget_order=budget_order,
        configured_colors=config['sampling_budget_colors'],
    )
    df_reward_analysis = build_reward_analysis_table(df_env_score)

    df_env_scores = df_reward_analysis[['env_id', 'env_score']].drop_duplicates('env_id')
    example_ids = {
        label: select_example_environment(
            df_env_scores=df_env_scores,
            score_window=example_config['score_window'],
            seed=example_config['seed'],
        )
        for label, example_config in config['example_environments'].items()
    }
    example_curves = {
        label: compute_example_environment_curves(
            df_full=df_full,
            env_id=env_id,
            budget_order=budget_order,
        )
        for label, env_id in example_ids.items()
    }

    for label, curves in example_curves.items():
        save_example_environment_plot(
            savepath=visual_dir / f'sampling-budget_rewards_env-score-{label}.pdf',
            p_win_left=curves[0],
            trial_ids=curves[1],
            curves=curves[2],
            budget_order=budget_order,
            budget_colors=budget_colors,
            save_figures=save_figures,
        )

    df_reward_corr_stats = compute_reward_correlation_stats(
        df_reward_analysis=df_reward_analysis,
        budget_order=budget_order,
    )
    df_reward_corr_stats.to_csv(output_dir / 'reward_envscore_stats_by_budget.csv', index=False)
    save_reward_correlation_plot(
        savepath=visual_dir / 'reward_envscore_correlation_by_budget.pdf',
        df_reward_corr_stats=df_reward_corr_stats,
        budget_colors=budget_colors,
        save_figures=save_figures,
    )

    df_contrast, df_contrast_stats, df_pivot, low_budgets, _, large_budgets = compute_budget_contrast(
        df_reward_analysis=df_reward_analysis,
    )
    df_contrast_stats.to_csv(output_dir / 'low_vs_large_budget_advantage_stats.csv', index=False)
    df_contrast.to_csv(output_dir / 'low_vs_large_budget_advantage_by_envscore.csv', index=False)
    save_budget_contrast_plot(
        savepath=visual_dir / 'low_vs_large_budget_advantage_by_envscore.pdf',
        df_contrast=df_contrast,
        df_contrast_stats=df_contrast_stats,
        save_figures=save_figures,
    )

    class_names, class_labels, class_colors = resolve_class_plot_config(config)
    df_env_bins = build_env_score_classes(
        df_reward_analysis=df_reward_analysis,
        class_names=class_names,
        class_labels=config['environment_classes']['labels'],
    )
    df_class_stats, df_winner_counts, df_class_contrast_stats = compute_env_class_stats(
        df_reward_analysis=df_reward_analysis,
        df_env_bins=df_env_bins,
        df_pivot=df_pivot,
        budget_order=budget_order,
        low_budgets=low_budgets,
        large_budgets=large_budgets,
    )
    df_class_stats.to_csv(output_dir / 'reward_by_envscore_tertile_stats.csv', index=False)
    df_winner_counts.to_csv(output_dir / 'winner_budget_counts_by_envscore_tertile.csv', index=False)
    df_class_contrast_stats.to_csv(
        output_dir / 'low_vs_large_budget_contrast_by_envscore_tertile.csv',
        index=False,
    )
    save_env_class_reward_plot(
        savepath=visual_dir / 'reward_by_envscore_tertile_and_budget.pdf',
        df_class_stats=df_class_stats,
        budget_order=budget_order,
        class_labels=class_labels,
        class_colors=class_colors,
        save_figures=save_figures,
    )
    save_unified_plot(
        savepath=visual_dir / 'temporal_only_analysis_unified.pdf',
        low_example=example_curves['low'],
        high_example=example_curves['high'],
        df_reward_corr_stats=df_reward_corr_stats,
        df_contrast=df_contrast,
        df_contrast_stats=df_contrast_stats,
        df_class_stats=df_class_stats,
        budget_order=budget_order,
        budget_colors=budget_colors,
        class_labels=class_labels,
        class_colors=class_colors,
        save_figures=save_figures,
    )
    plt.close('all')


def run_analysis(args: argparse.Namespace) -> None:
    """
    Dispatch the temporal-only analysis for the configured replacement regime.

    Args:
        args (argparse.Namespace): Parsed analysis arguments.
    """
    config = load_analysis_config(args.config)
    simulation_config_path = Path(config['simulation_config_path'])
    if not simulation_config_path.is_absolute():
        simulation_config_path = filepaths.ROOT / simulation_config_path
    simulation_config = load_simulation_config(simulation_config_path)

    if simulation_config['sample_replacement']:
        from scripts.analysis_simulation_genesis_reward_bandit_task_temporal_only_replacement import (
            run_analysis as run_replacement_analysis,
        )

        run_replacement_analysis(args)
        return

    run_analysis_without_replacement(args)


def main(argv: list[str] | None = None) -> None:
    """
    Run the temporal-only analysis from command-line arguments.

    Args:
        argv (list[str] | None): Optional command-line argument list.
    """
    args = parse_args(argv)
    run_analysis(args)


if __name__ == '__main__':
    main()
