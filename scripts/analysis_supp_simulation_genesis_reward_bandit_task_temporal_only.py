"""
Analyze the supplementary GENESIS-budget versus TD two-armed bandit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path.cwd()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

from utils import filepaths
from utils.metrics import early_to_late_direction_score


DEFAULT_CONFIG_PATH = (
    filepaths.ROOT / 'configs' / 'analysis_supp_simulation_genesis_reward_bandit_task_temporal_only.yaml'
)
MODEL_LABELS = {
    'GENESIS_1': 'GENESIS k=1',
    'GENESIS_3': 'GENESIS k=3',
    'GENESIS_5': 'GENESIS k=5',
    'GENESIS_10': 'GENESIS k=10',
    'GENESIS_20': 'GENESIS k=20',
    'GENESIS_50': 'GENESIS k=50',
    'GENESIS_ALL': 'GENESIS ALL',
    'TD_RL': 'TD RL',
}
LINE_STYLES = {
    'GENESIS_1': '-',
    'GENESIS_3': '-',
    'GENESIS_5': '-',
    'GENESIS_10': '-',
    'GENESIS_20': '-',
    'GENESIS_50': '-',
    'GENESIS_ALL': '--',
    'TD_RL': ':',
}
REWARD_RIBBON_SEM_DIVISOR = 10.0
VALUE_RIBBON_SEM_DIVISOR = 2.0
RIBBON_ALPHA = 0.08


def load_simulation_config(config_path: Path) -> dict[str, Any]:
    """
    Load a simulation YAML without importing model dependencies.

    Args:
        config_path (Path): Root-relative or absolute YAML path.

    Returns:
        dict[str, Any]: Parsed simulation configuration.
    """
    path = config_path if config_path.is_absolute() else filepaths.ROOT / config_path
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def model_ids_from_config(config: dict[str, Any]) -> list[str]:
    """
    Construct the configured model order.

    Args:
        config (dict[str, Any]): Simulation configuration.

    Returns:
        list[str]: Seven GENESIS identifiers followed by TD.
    """
    genesis_ids = [f'GENESIS_{str(value).upper()}' for value in config['genesis']['sampling_budgets']]
    return [*genesis_ids, 'TD_RL']


def config_fingerprint(config: dict[str, Any]) -> str:
    """
    Hash a resolved simulation configuration.

    Args:
        config (dict[str, Any]): Simulation configuration.

    Returns:
        str: Twelve-character fingerprint.
    """
    payload = json.dumps(config, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:12]


def validate_complete_record(record: dict[str, Any]) -> None:
    """
    Validate exact environment, agent, model, and trial completeness.

    Args:
        record (dict[str, Any]): Nested schema-v1 simulation record.

    Raises:
        ValueError: If combinations are missing or duplicated.
    """
    config = record['resolved_config']
    if record['config_fingerprint'] != config_fingerprint(config):
        raise ValueError('Resolved configuration does not match the saved fingerprint.')
    if record['actual_n_environments'] != config['n_environments']:
        raise ValueError('Actual environment count does not match the resolved configuration.')
    if record['actual_n_agent_simulations'] != config['n_agent_simulations']:
        raise ValueError('Actual agent count does not match the resolved configuration.')
    if len(record['environments']) != config['n_environments']:
        raise ValueError('The environment list is incomplete.')
    keys = []
    for environment in record['environments']:
        if len(environment['agent_records']) != config['n_agent_simulations']:
            raise ValueError('An environment has an incomplete agent set.')
        for agent in environment['agent_records']:
            histories = agent['model_histories']
            if [history['model_id'] for history in histories] != record['model_ids']:
                raise ValueError('A model history set is missing, duplicated, or out of order.')
            for history in histories:
                if len(history['trials']) != config['task']['n_choice_trials']:
                    raise ValueError('A model history has an incomplete trial sequence.')
                keys.extend(
                    (environment['environment_id'], agent['agent_id'], history['model_id'], trial['trial_id'])
                    for trial in history['trials']
                )
    expected = (
        config['n_environments']
        * config['n_agent_simulations']
        * len(record['model_ids'])
        * config['task']['n_choice_trials']
    )
    expected_runs = config['n_environments'] * config['n_agent_simulations'] * len(record['model_ids'])
    if record['expected_complete_runs'] != expected_runs or record['expected_trial_records'] != expected:
        raise ValueError('Saved expected counts do not match the resolved configuration.')
    if len(keys) != expected or len(set(keys)) != expected:
        raise ValueError('Simulation records are incomplete or contain duplicate keys.')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse supplementary analysis arguments.

    Args:
        argv (list[str] | None): Optional command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Analyze supplementary GENESIS-budget versus TD outputs.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--rebuild-cache', action='store_true')
    figures = parser.add_mutually_exclusive_group()
    figures.add_argument('--save-figures', action='store_true', dest='save_figures')
    figures.add_argument('--no-save-figures', action='store_false', dest='save_figures')
    parser.set_defaults(save_figures=None)
    return parser.parse_args(argv)


def load_analysis_config(config_path: Path) -> dict[str, Any]:
    """
    Load the analysis YAML.

    Args:
        config_path (Path): Root-relative or absolute analysis YAML path.

    Returns:
        dict[str, Any]: Parsed analysis configuration.
    """
    path = config_path if config_path.is_absolute() else filepaths.ROOT / config_path
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def is_hex_color(value: str) -> bool:
    """
    Return whether a value is a six-digit hexadecimal color.

    Args:
        value (str): Candidate color.

    Returns:
        bool: Whether the color has form ``#RRGGBB``.
    """
    if not isinstance(value, str) or len(value) != 7 or not value.startswith('#'):
        return False
    return all(character in '0123456789abcdefABCDEF' for character in value[1:])


def validate_analysis_config(config: dict[str, Any], model_ids: Sequence[str]) -> None:
    """
    Validate analysis controls against the resolved model set.

    Args:
        config (dict[str, Any]): Analysis configuration.
        model_ids (Sequence[str]): Ordered simulation model identifiers.

    Raises:
        ValueError: If an analysis control is invalid or misleading.
    """
    if config['n_boot'] <= 0:
        raise ValueError('n_boot must be positive.')
    for example in config['example_environments'].values():
        window = example['score_window']
        if len(window) != 2 or window[0] > window[1]:
            raise ValueError('Example score windows must be ordered pairs.')
    colors = config['model_colors']
    if list(colors) != list(model_ids):
        raise ValueError('model_colors keys must exactly match the ordered simulation models.')
    if not all(is_hex_color(color) for color in colors.values()):
        raise ValueError('Every configured model color must use #RRGGBB form.')
    classes = config['environment_classes']
    if set(classes['classes']) != set(classes['labels']) or set(classes['classes']) != set(classes['colors']):
        raise ValueError('Environment class names, labels, and colors must match.')
    if not all(is_hex_color(color) for color in classes['colors'].values()):
        raise ValueError('Every environment-class color must use #RRGGBB form.')
    plots = config['plots']
    if plots['cumulative_reward_smoothing_window'] <= 0 or plots['cumulative_reward_smoothing_window'] % 2 == 0:
        raise ValueError('Cumulative-reward smoothing must be a positive odd integer.')
    if plots['value_smoothing_window'] <= 0:
        raise ValueError('Value smoothing must be positive.')
    if plots['uncertainty'] != 'sem':
        raise ValueError('The supplementary primary figures require SEM uncertainty.')


def resolve_simulation_config_path(config: dict[str, Any]) -> Path:
    """
    Resolve the analysis input simulation YAML.

    Args:
        config (dict[str, Any]): Analysis configuration.

    Returns:
        Path: Existing absolute simulation YAML path.
    """
    path = Path(config['simulation_config_path'])
    path = path if path.is_absolute() else filepaths.ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def discover_record(input_dir: Path, simulation_config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """
    Find and validate the record matching the configured simulation.

    Args:
        input_dir (Path): Supplementary raw-record directory.
        simulation_config (dict[str, Any]): Simulation YAML configuration.

    Returns:
        tuple[Path, dict[str, Any]]: Selected path and loaded record.
    """
    expected_path = input_dir / f'simulation_{config_fingerprint(simulation_config)}.npy'
    candidates = sorted(input_dir.glob('simulation_*.npy'))
    if expected_path.is_file():
        selected = expected_path
    elif len(candidates) == 1:
        selected = candidates[0]
    else:
        raise ValueError('No unique simulation record matches the configured simulation.')
    record = np.load(selected, allow_pickle=True).item()
    validate_complete_record(record)
    if record['schema_version'] != simulation_config['output']['schema_version']:
        raise ValueError('Simulation schema version does not match the analysis configuration.')
    if record['experiment_name'] != simulation_config['experiment_name']:
        raise ValueError('Simulation experiment name does not match the analysis configuration.')
    if record['original_config'] != simulation_config:
        raise ValueError('Simulation records were generated from a different source YAML configuration.')
    return selected, record


def analysis_fingerprint(config: dict[str, Any], simulation_fingerprint: str) -> str:
    """
    Build a cache identity from analysis controls and simulation identity.

    Args:
        config (dict[str, Any]): Analysis configuration.
        simulation_fingerprint (str): Selected simulation fingerprint.

    Returns:
        str: Stable cache identity.
    """
    payload = json.dumps(
        {'analysis_config': config, 'simulation_fingerprint': simulation_fingerprint},
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:12]


def validate_or_write_cache_metadata(
    *,
    output_dir: Path,
    fingerprint: str,
    rebuild_cache: bool,
) -> None:
    """
    Prevent reuse of analysis caches from incompatible settings.

    Args:
        output_dir (Path): Analysis output directory.
        fingerprint (str): Current analysis cache fingerprint.
        rebuild_cache (bool): Whether replacement of stale caches is authorized.

    Raises:
        ValueError: If stale cache metadata exists without a rebuild request.
    """
    metadata_path = output_dir / 'cache_metadata.json'
    if metadata_path.exists():
        cached = json.loads(metadata_path.read_text(encoding='utf-8'))
        if cached['fingerprint'] != fingerprint and not rebuild_cache:
            raise ValueError('Analysis caches were generated from incompatible settings; use --rebuild-cache.')
    metadata_path.write_text(json.dumps({'fingerprint': fingerprint}, indent=2), encoding='utf-8')


def flatten_record(record: dict[str, Any]) -> pd.DataFrame:
    """
    Flatten a validated nested record into trial rows.

    Args:
        record (dict[str, Any]): Validated schema-v1 simulation record.

    Returns:
        pd.DataFrame: Complete trial-level table.
    """
    rows = []
    for environment in record['environments']:
        for agent in environment['agent_records']:
            for history in agent['model_histories']:
                for trial in history['trials']:
                    row = dict(trial)
                    row.update({
                        'environment_id': environment['environment_id'],
                        'environment_seed': environment['environment_seed'],
                        'agent_id': agent['agent_id'],
                    })
                    rows.append(row)
    df = pd.DataFrame(rows)
    key_columns = ['environment_id', 'agent_id', 'model_id', 'trial_id']
    if df.duplicated(key_columns).any():
        raise ValueError('Flattened records contain duplicate trial keys.')
    for column in ['retrieved_indices_left', 'retrieved_indices_right']:
        df[column] = df[column].map(lambda value: json.dumps(value) if isinstance(value, list) else value)
    return df.sort_values(key_columns).reset_index(drop=True)


def environment_scores(df_trials: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the early-to-late direction score once per environment.

    Args:
        df_trials (pd.DataFrame): Trial-level simulation records.

    Returns:
        pd.DataFrame: One predictability-score row per environment.
    """
    rows = []
    for environment_id, group in df_trials.groupby('environment_id', sort=True):
        probabilities = group.pivot_table(index='trial_id', values='p_win_left', aggfunc='first').p_win_left
        if group.groupby('trial_id').p_win_left.nunique().max() != 1:
            raise ValueError('An environment has inconsistent reward-probability trajectories.')
        score = np.nanmean([
            early_to_late_direction_score(probabilities.to_numpy(), N=window)
            for window in [20, 30, 40]
        ])
        rows.append({'environment_id': environment_id, 'environment_score': score})
    return pd.DataFrame(rows)


def build_agent_summary(
    df_trials: pd.DataFrame,
    *,
    n_warmup: int,
) -> pd.DataFrame:
    """
    Summarize each environment, agent, model, and trial window.

    Args:
        df_trials (pd.DataFrame): Trial-level records.
        n_warmup (int): Number of initial random-choice trials.

    Returns:
        pd.DataFrame: Agent-level reward, policy, and value-tracking metrics.
    """
    rows = []
    windows = {'all': lambda frame: frame, 'post_warmup': lambda frame: frame[frame.trial_id >= n_warmup]}
    for keys, group in df_trials.groupby(['environment_id', 'agent_id', 'model_id'], sort=False):
        environment_id, agent_id, model_id = keys
        for window_name, selector in windows.items():
            frame = selector(group)
            true_left = 2.0 * frame.p_win_left.to_numpy() - 1.0
            true_right = 2.0 * frame.p_win_right.to_numpy() - 1.0
            estimated_left = frame.action_value_left.to_numpy(dtype=float)
            estimated_right = frame.action_value_right.to_numpy(dtype=float)
            modeled = ~frame.is_random_warmup.to_numpy(dtype=bool)
            valid = modeled & np.isfinite(estimated_left) & np.isfinite(estimated_right)
            value_rmse = (
                float(np.sqrt(np.mean(np.concatenate([
                    (estimated_left[valid] - true_left[valid]) ** 2,
                    (estimated_right[valid] - true_right[valid]) ** 2,
                ]))))
                if valid.any()
                else np.nan
            )
            estimated_difference = estimated_left - estimated_right
            true_difference = true_left - true_right
            value_difference_rmse = (
                float(np.sqrt(np.mean((estimated_difference[valid] - true_difference[valid]) ** 2)))
                if valid.any()
                else np.nan
            )
            rows.append({
                'environment_id': environment_id,
                'agent_id': agent_id,
                'model_id': model_id,
                'trial_window': window_name,
                'n_trials': len(frame),
                'total_reward': frame.reward.sum(),
                'mean_reward': frame.reward.mean(),
                'reward_sd': frame.reward.std(ddof=1),
                'mean_p_right': frame.p_right.mean(),
                'value_rmse': value_rmse,
                'value_difference_rmse': value_difference_rmse,
            })
    return pd.DataFrame(rows)


def sem(values: pd.Series) -> float:
    """
    Compute the standard error of a numeric series.

    Args:
        values (pd.Series): Numeric observations.

    Returns:
        float: Sample standard error or NaN for fewer than two values.
    """
    values = values.dropna()
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan


def build_environment_summary(
    df_agents: pd.DataFrame,
    df_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate paired agents within each environment and model.

    Args:
        df_agents (pd.DataFrame): Agent-level metrics.
        df_scores (pd.DataFrame): Environment predictability scores.

    Returns:
        pd.DataFrame: Environment-by-model summary table.
    """
    metrics = ['total_reward', 'mean_reward', 'reward_sd', 'mean_p_right', 'value_rmse', 'value_difference_rmse']
    rows = []
    for keys, group in df_agents.groupby(['environment_id', 'model_id', 'trial_window'], sort=False):
        environment_id, model_id, trial_window = keys
        row = {
            'environment_id': environment_id,
            'model_id': model_id,
            'trial_window': trial_window,
            'n_agents': group.agent_id.nunique(),
        }
        for metric in metrics:
            row[f'{metric}_mean'] = group[metric].mean()
            row[f'{metric}_sem'] = sem(group[metric])
        rows.append(row)
    return pd.DataFrame(rows).merge(df_scores, on='environment_id', validate='many_to_one')


def bootstrap_mean_ci(values: Sequence[float], *, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    """
    Bootstrap a mean by resampling environment-level values.

    Args:
        values (Sequence[float]): One value per independent environment.
        n_boot (int): Number of bootstrap draws.
        rng (np.random.Generator): Analysis random generator.

    Returns:
        tuple[float, float]: Percentile 95% interval.
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    samples = rng.choice(array, size=(n_boot, len(array)), replace=True).mean(axis=1)
    return tuple(np.percentile(samples, [2.5, 97.5]))


def bootstrap_corr_ci(
    x: Sequence[float],
    y: Sequence[float],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """
    Bootstrap a Pearson correlation over environments.

    Args:
        x (Sequence[float]): Environment scores.
        y (Sequence[float]): Environment-level outcomes.
        n_boot (int): Number of bootstrap draws.
        rng (np.random.Generator): Analysis random generator.

    Returns:
        tuple[float, float]: Percentile 95% interval or NaNs.
    """
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if len(x_array) < 3 or np.std(x_array) == 0 or np.std(y_array) == 0:
        return np.nan, np.nan
    values = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(x_array), len(x_array))
        if np.std(x_array[indices]) == 0 or np.std(y_array[indices]) == 0:
            continue
        values.append(np.corrcoef(x_array[indices], y_array[indices])[0, 1])
    return tuple(np.percentile(values, [2.5, 97.5])) if values else (np.nan, np.nan)


def build_model_and_contrast_summaries(
    df_environment: pd.DataFrame,
    *,
    model_ids: Sequence[str],
    n_boot: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Summarize models and paired GENESIS-minus-TD contrasts.

    Args:
        df_environment (pd.DataFrame): Environment-by-model summaries.
        model_ids (Sequence[str]): Ordered model identifiers.
        n_boot (int): Environment bootstrap count.
        random_state (int): Analysis seed.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Model, contrast, and contrast-summary tables.
    """
    rng = np.random.default_rng(random_state)
    model_rows = []
    contrast_rows = []
    contrast_summary_rows = []
    for window in ['all', 'post_warmup']:
        frame = df_environment[df_environment.trial_window == window]
        for model_id in model_ids:
            group = frame[frame.model_id == model_id]
            reward_ci = bootstrap_mean_ci(group.total_reward_mean, n_boot=n_boot, rng=rng)
            correlation = (
                np.corrcoef(group.environment_score, group.total_reward_mean)[0, 1]
                if len(group) >= 2 and group.environment_score.std() > 0 and group.total_reward_mean.std() > 0
                else np.nan
            )
            correlation_ci = bootstrap_corr_ci(
                group.environment_score,
                group.total_reward_mean,
                n_boot=n_boot,
                rng=rng,
            )
            model_rows.append({
                'model_id': model_id,
                'trial_window': window,
                'n_environments': len(group),
                'mean_total_reward': group.total_reward_mean.mean(),
                'total_reward_ci_low': reward_ci[0],
                'total_reward_ci_high': reward_ci[1],
                'reward_predictability_r': correlation,
                'reward_predictability_ci_low': correlation_ci[0],
                'reward_predictability_ci_high': correlation_ci[1],
            })
        reward_pivot = frame.pivot(index='environment_id', columns='model_id', values='total_reward_mean')
        score_map = frame.drop_duplicates('environment_id').set_index('environment_id').environment_score
        for model_id in model_ids[:-1]:
            differences = reward_pivot[model_id] - reward_pivot['TD_RL']
            for environment_id, difference in differences.items():
                contrast_rows.append({
                    'environment_id': environment_id,
                    'environment_score': score_map.loc[environment_id],
                    'model_id': model_id,
                    'trial_window': window,
                    'genesis_minus_td_reward': difference,
                })
            ci = bootstrap_mean_ci(differences, n_boot=n_boot, rng=rng)
            contrast_summary_rows.append({
                'model_id': model_id,
                'trial_window': window,
                'n_environments': len(differences),
                'mean_genesis_minus_td_reward': differences.mean(),
                'contrast_ci_low': ci[0],
                'contrast_ci_high': ci[1],
            })
    df_models = pd.DataFrame(model_rows)
    df_contrasts = pd.DataFrame(contrast_rows)
    df_contrast_summary = pd.DataFrame(contrast_summary_rows)
    relative_columns = df_contrast_summary.rename(columns={
        'mean_genesis_minus_td_reward': 'mean_relative_to_td',
        'contrast_ci_low': 'relative_to_td_ci_low',
        'contrast_ci_high': 'relative_to_td_ci_high',
    })[[
        'model_id',
        'trial_window',
        'mean_relative_to_td',
        'relative_to_td_ci_low',
        'relative_to_td_ci_high',
    ]]
    df_models = df_models.merge(relative_columns, on=['model_id', 'trial_window'], how='left')
    td_rows = df_models.model_id == 'TD_RL'
    df_models.loc[
        td_rows,
        ['mean_relative_to_td', 'relative_to_td_ci_low', 'relative_to_td_ci_high'],
    ] = 0.0
    return df_models, df_contrasts, df_contrast_summary


def build_all_diagnostic(df_trials: pd.DataFrame) -> pd.DataFrame:
    """
    Compare decoded GENESIS ALL values with exact raw running means.

    Args:
        df_trials (pd.DataFrame): Trial-level records.

    Returns:
        pd.DataFrame: Per-agent decoder diagnostic.
    """
    frame = df_trials[(df_trials.model_id == 'GENESIS_ALL') & (~df_trials.is_random_warmup)].copy()
    frame['left_error'] = frame.action_value_left - frame.exact_sample_mean_left
    frame['right_error'] = frame.action_value_right - frame.exact_sample_mean_right
    rows = []
    for keys, group in frame.groupby(['environment_id', 'agent_id']):
        errors = np.concatenate([group.left_error.to_numpy(), group.right_error.to_numpy()])
        rows.append({
            'environment_id': keys[0],
            'agent_id': keys[1],
            'mean_absolute_decoder_error': np.mean(np.abs(errors)),
            'max_absolute_decoder_error': np.max(np.abs(errors)),
        })
    return pd.DataFrame(rows)


def select_example_environment(
    *,
    df_scores: pd.DataFrame,
    score_window: Sequence[float],
    seed: int,
    preferred_environment_id: int | None = None,
) -> tuple[int, str]:
    """
    Select an example using environment scores only.

    Args:
        df_scores (pd.DataFrame): One score row per environment.
        score_window (Sequence[float]): Inclusive selection interval.
        seed (int): Deterministic selection seed.
        preferred_environment_id (int | None): Eligible historical preferred environment.

    Returns:
        tuple[int, str]: Selected identifier and selection status.
    """
    low, high = score_window
    eligible = df_scores[df_scores.environment_score.between(low, high)].sort_values('environment_id')
    if preferred_environment_id is not None and preferred_environment_id in eligible.environment_id.values:
        return int(preferred_environment_id), 'preferred_eligible'
    if not eligible.empty:
        rng = np.random.default_rng(seed)
        return int(rng.choice(eligible.environment_id.to_numpy())), 'seeded_window_selection'
    distances = np.maximum(low - df_scores.environment_score, 0) + np.maximum(df_scores.environment_score - high, 0)
    closest = df_scores.assign(distance=distances).sort_values(['distance', 'environment_id']).iloc[0]
    return int(closest.environment_id), 'fallback_closest_score'


def compute_example_curves(
    df_trials: pd.DataFrame,
    *,
    environment_id: int,
    model_ids: Sequence[str],
    reward_window: int,
    value_window: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Build agent-first smoothed example trajectories.

    Args:
        df_trials (pd.DataFrame): Trial-level records.
        environment_id (int): Selected environment.
        model_ids (Sequence[str]): Ordered model identifiers.
        reward_window (int): Savitzky-Golay reward smoothing window.
        value_window (int): Trailing value smoothing window.

    Returns:
        tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]: Probability, trial ids, and model curves.
    """
    environment = df_trials[df_trials.environment_id == environment_id]
    trial_ids = np.sort(environment.trial_id.unique())
    probabilities = environment.groupby('trial_id').p_win_left.first().loc[trial_ids].to_numpy()
    curves = []
    for model_id in model_ids:
        model = environment[environment.model_id == model_id]
        rewards = []
        values = []
        for _, agent in model.groupby('agent_id', sort=True):
            agent = agent.sort_values('trial_id')
            cumulative = agent.reward.cumsum().to_numpy(dtype=float)
            rewards.append(savgol_filter(cumulative, reward_window, 1))
            value_difference = (agent.action_value_left - agent.action_value_right).mask(agent.is_random_warmup)
            values.append(value_difference.rolling(value_window, min_periods=value_window).mean().to_numpy())
        reward_array = np.vstack(rewards)
        value_array = np.vstack(values)
        value_frame = pd.DataFrame(value_array)
        value_counts = value_frame.count(axis=0).to_numpy()
        value_sem = value_frame.std(axis=0, ddof=1).to_numpy() / np.sqrt(value_counts)
        curves.append({
            'model_id': model_id,
            'reward_mean': np.nanmean(reward_array, axis=0),
            'reward_sem': np.nanstd(reward_array, axis=0, ddof=1) / np.sqrt(reward_array.shape[0]),
            'value_mean': value_frame.mean(axis=0).to_numpy(),
            'value_sem': value_sem,
            'n_agents': reward_array.shape[0],
        })
    return probabilities, trial_ids, curves


def shared_reward_ylim(curve_sets: Sequence[Sequence[dict[str, Any]]]) -> tuple[float, float]:
    """
    Compute a shared reward-axis interval including unscaled SEM.

    Args:
        curve_sets (Sequence[Sequence[dict[str, Any]]]): Low/high example curves.

    Returns:
        tuple[float, float]: Shared lower and upper limits.
    """
    lower = min(
        np.nanmin(curve['reward_mean'] - curve['reward_sem'] / REWARD_RIBBON_SEM_DIVISOR)
        for curves in curve_sets
        for curve in curves
    )
    upper = max(
        np.nanmax(curve['reward_mean'] + curve['reward_sem'] / REWARD_RIBBON_SEM_DIVISOR)
        for curves in curve_sets
        for curve in curves
    )
    padding = max((upper - lower) * 0.05, 1.0)
    return lower - padding, upper + padding


def style_axis(axis: Axes) -> None:
    """
    Apply the established minimal plot style.

    Args:
        axis (Axes): Matplotlib axis.
    """
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)


def save_example_plot(
    *,
    path: Path,
    probabilities: np.ndarray,
    trial_ids: np.ndarray,
    curves: Sequence[dict[str, Any]],
    model_ids: Sequence[str],
    colors: dict[str, str],
    n_warmup: int,
    reward_ylim: tuple[float, float] | None,
    title: str,
    pdf_metadata: dict[str, str],
    save_figures: bool,
) -> None:
    """
    Save one two-panel example-environment figure.

    Args:
        path (Path): PDF destination.
        probabilities (np.ndarray): True LEFT win probabilities.
        trial_ids (np.ndarray): Trial identifiers.
        curves (Sequence[dict[str, Any]]): Ordered model trajectories.
        model_ids (Sequence[str]): Ordered model identifiers.
        colors (dict[str, str]): Model colors.
        n_warmup (int): Random warm-up length.
        reward_ylim (tuple[float, float] | None): Shared reward limits.
        title (str): Figure title.
        pdf_metadata (dict[str, str]): Embedded PDF metadata.
        save_figures (bool): Whether to save the PDF.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for axis, metric, ylabel in [
        (axes[0], 'reward', 'Cumulative reward'),
        (axes[1], 'value', 'Q(LEFT) - Q(RIGHT)'),
    ]:
        probability_axis = axis.twinx()
        probability_axis.plot(trial_ids, probabilities, color='#808080', linewidth=1.5, alpha=0.65)
        probability_axis.axhline(0.5, color='#808080', linestyle='--', linewidth=0.8, alpha=0.5)
        probability_axis.set_ylim(0, 1)
        probability_axis.set_ylabel('P(Win LEFT)', color='#808080')
        probability_axis.tick_params(axis='y', colors='#808080', labelsize=8)
        for curve in curves:
            model_id = curve['model_id']
            mean = curve[f'{metric}_mean']
            uncertainty = curve[f'{metric}_sem']
            uncertainty_divisor = (
                REWARD_RIBBON_SEM_DIVISOR
                if metric == 'reward'
                else VALUE_RIBBON_SEM_DIVISOR
            )
            display_uncertainty = uncertainty / uncertainty_divisor
            axis.plot(
                trial_ids,
                mean,
                color=colors[model_id],
                linestyle=LINE_STYLES[model_id],
                linewidth=2.0 if model_id in {'GENESIS_ALL', 'TD_RL'} else 1.5,
            )
            axis.fill_between(
                trial_ids,
                mean - display_uncertainty,
                mean + display_uncertainty,
                color=colors[model_id],
                alpha=RIBBON_ALPHA,
                linewidth=0,
                zorder=1,
            )
        axis.axvspan(-0.5, n_warmup - 0.5, color='#D9D9D9', alpha=0.25)
        axis.axvline(n_warmup - 0.5, color='#777777', linestyle=':', linewidth=1)
        axis.set_xlabel('Trial')
        axis.set_ylabel(ylabel)
        style_axis(axis)
    if reward_ylim is not None:
        axes[0].set_ylim(reward_ylim)
    axes[0].text(0.02, 0.03, f'Random warm-up: trials 1-{n_warmup}', transform=axes[0].transAxes, fontsize=8)
    axes[0].text(0.98, 0.03, 'Display ribbon: SEM / 10', transform=axes[0].transAxes, ha='right', fontsize=8)
    axes[1].text(0.98, 0.03, 'Display ribbon: SEM / 2', transform=axes[1].transAxes, ha='right', fontsize=8)
    handles = [
        Line2D([0], [0], color=colors[model_id], linestyle=LINE_STYLES[model_id], label=MODEL_LABELS[model_id])
        for model_id in model_ids
    ]
    fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.suptitle(f'{title}\nGENESIS: retrieval temperature 0.25; ALL exhaustive/unweighted; TD RL baseline', y=1.15)
    fig.tight_layout()
    if save_figures:
        fig.savefig(path, dpi=800, bbox_inches='tight', metadata=pdf_metadata)
    plt.close(fig)


def save_budget_summary_plot(
    *,
    path: Path,
    df_models: pd.DataFrame,
    model_ids: Sequence[str],
    colors: dict[str, str],
    save_figures: bool,
) -> None:
    """
    Save post-warm-up mean reward by GENESIS budget with TD reference.

    Args:
        path (Path): PDF destination.
        df_models (pd.DataFrame): Model summary table.
        model_ids (Sequence[str]): Ordered models.
        colors (dict[str, str]): Model colors.
        save_figures (bool): Whether to save the figure.
    """
    frame = df_models[df_models.trial_window == 'post_warmup'].set_index('model_id').loc[list(model_ids)]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(model_ids) - 1)
    genesis = frame.loc[list(model_ids[:-1])]
    axis.errorbar(
        x,
        genesis.mean_total_reward,
        yerr=[
            genesis.mean_total_reward - genesis.total_reward_ci_low,
            genesis.total_reward_ci_high - genesis.mean_total_reward,
        ],
        color='#333333',
        marker='o',
        capsize=3,
    )
    for index, model_id in enumerate(model_ids[:-1]):
        axis.scatter(index, genesis.loc[model_id, 'mean_total_reward'], color=colors[model_id], s=60, zorder=3)
    td_reward = frame.loc['TD_RL', 'mean_total_reward']
    axis.axhline(td_reward, color=colors['TD_RL'], linestyle=':', label='TD RL mean')
    axis.set_xticks(x, [MODEL_LABELS[model_id].replace('GENESIS ', '') for model_id in model_ids[:-1]])
    axis.set_xlabel('GENESIS sampling budget')
    axis.set_ylabel('Post-warm-up total reward')
    axis.legend(frameon=False)
    style_axis(axis)
    fig.tight_layout()
    if save_figures:
        fig.savefig(path, dpi=800, bbox_inches='tight')
    plt.close(fig)


def save_contrast_plot(
    *,
    path: Path,
    df_contrasts: pd.DataFrame,
    model_ids: Sequence[str],
    colors: dict[str, str],
    save_figures: bool,
) -> None:
    """
    Save paired post-warm-up GENESIS-minus-TD differences by predictability.

    Args:
        path (Path): PDF destination.
        df_contrasts (pd.DataFrame): Paired environment contrasts.
        model_ids (Sequence[str]): Ordered models.
        colors (dict[str, str]): Model colors.
        save_figures (bool): Whether to save the figure.
    """
    frame = df_contrasts[df_contrasts.trial_window == 'post_warmup']
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True, sharey=True)
    for axis, model_id in zip(axes.flat, model_ids[:-1]):
        group = frame[frame.model_id == model_id]
        axis.scatter(group.environment_score, group.genesis_minus_td_reward, color=colors[model_id], alpha=0.75)
        axis.axhline(0, color='#555555', linestyle='--', linewidth=1)
        if len(group) >= 2 and group.environment_score.std() > 0:
            coefficients = np.polyfit(group.environment_score, group.genesis_minus_td_reward, 1)
            x_line = np.linspace(group.environment_score.min(), group.environment_score.max(), 100)
            axis.plot(x_line, np.polyval(coefficients, x_line), color=colors[model_id], linewidth=1.5)
        axis.set_title(MODEL_LABELS[model_id])
        style_axis(axis)
    axes.flat[-1].axis('off')
    fig.supxlabel('Early-to-late direction score')
    fig.supylabel('Post-warm-up reward: GENESIS - TD')
    fig.tight_layout()
    if save_figures:
        fig.savefig(path, dpi=800, bbox_inches='tight')
    plt.close(fig)


def build_tertile_summary(df_environment: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Summarize reward by predictability tertile when groups are estimable.

    Args:
        df_environment (pd.DataFrame): Environment-by-model table.

    Returns:
        tuple[pd.DataFrame, str]: Tertile table and status message.
    """
    scores = df_environment[['environment_id', 'environment_score']].drop_duplicates().sort_values('environment_id')
    if len(scores) < 6:
        return pd.DataFrame(), 'insufficient environments: fewer than six unique environments'
    try:
        scores['environment_class'] = pd.qcut(scores.environment_score, q=3, labels=['Low', 'Mid', 'High'])
    except ValueError:
        return pd.DataFrame(), 'insufficient environments: predictability scores do not form three tertiles'
    if scores.groupby('environment_class', observed=True).size().min() < 2:
        return pd.DataFrame(), 'insufficient environments: fewer than two environments in a tertile'
    frame = df_environment.merge(scores[['environment_id', 'environment_class']], on='environment_id')
    summary = (
        frame.groupby(['trial_window', 'environment_class', 'model_id'], observed=True)
        .total_reward_mean
        .agg(['count', 'mean', sem])
        .reset_index()
        .rename(columns={'count': 'n_environments', 'mean': 'mean_total_reward', 'sem': 'sem_total_reward'})
    )
    return summary, 'complete'


def write_or_load_table(
    *,
    path: Path,
    rebuild_cache: bool,
    builder: Any,
) -> pd.DataFrame:
    """
    Load a compatible CSV cache or build and save it.

    Args:
        path (Path): CSV path.
        rebuild_cache (bool): Whether to rebuild existing tables.
        builder (Any): Zero-argument table builder.

    Returns:
        pd.DataFrame: Loaded or newly built table.
    """
    if path.exists() and not rebuild_cache:
        return pd.read_csv(path)
    table = builder()
    table.to_csv(path, index=False)
    return table


def run_analysis(args: argparse.Namespace) -> None:
    """
    Run the complete supplementary analysis.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
    """
    analysis_config = load_analysis_config(args.config)
    simulation_config_path = resolve_simulation_config_path(analysis_config)
    simulation_config = load_simulation_config(simulation_config_path)
    expected_model_ids = model_ids_from_config(simulation_config)
    validate_analysis_config(analysis_config, expected_model_ids)
    input_dir = args.input_dir or (
        filepaths.EXPERIMENTS_DIR
        / simulation_config['experiment_name']
        / simulation_config['output']['sample_replacement_subdir']
    )
    if not input_dir.is_absolute():
        input_dir = filepaths.ROOT / input_dir
    record_path, record = discover_record(input_dir, simulation_config)
    resolved_config = record['resolved_config']
    model_ids = record['model_ids']
    if model_ids != model_ids_from_config(resolved_config):
        raise ValueError('Saved model order does not match the resolved simulation configuration.')
    validate_analysis_config(analysis_config, model_ids)
    output_dir = args.output_dir or (
        filepaths.OUTPUT_DIR
        / analysis_config['output_dir_name']
        / resolved_config['output']['sample_replacement_subdir']
    )
    if not output_dir.is_absolute():
        output_dir = filepaths.ROOT / output_dir
    visual_dir = output_dir / 'visuals'
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = analysis_fingerprint(analysis_config, record['config_fingerprint'])
    validate_or_write_cache_metadata(
        output_dir=output_dir,
        fingerprint=fingerprint,
        rebuild_cache=args.rebuild_cache,
    )
    save_figures = analysis_config['save_figures'] if args.save_figures is None else args.save_figures

    df_trials = write_or_load_table(
        path=output_dir / 'trial_level.csv',
        rebuild_cache=args.rebuild_cache,
        builder=lambda: flatten_record(record),
    )
    expected_rows = record['expected_trial_records']
    if len(df_trials) != expected_rows:
        raise ValueError(f'Trial table has {len(df_trials)} rows; expected {expected_rows}.')
    df_scores = environment_scores(df_trials)
    df_scores.to_csv(output_dir / 'environment_predictability.csv', index=False)
    df_agents = write_or_load_table(
        path=output_dir / 'agent_model_summary.csv',
        rebuild_cache=args.rebuild_cache,
        builder=lambda: build_agent_summary(
            df_trials,
            n_warmup=resolved_config['task']['n_random_warmup_trials'],
        ),
    )
    df_environment = build_environment_summary(df_agents, df_scores)
    df_environment.to_csv(output_dir / 'environment_by_model.csv', index=False)
    df_models, df_contrasts, df_contrast_summary = build_model_and_contrast_summaries(
        df_environment,
        model_ids=model_ids,
        n_boot=analysis_config['n_boot'],
        random_state=analysis_config['random_state'],
    )
    df_models.to_csv(output_dir / 'model_summary.csv', index=False)
    df_contrasts.to_csv(output_dir / 'genesis_vs_td_by_environment.csv', index=False)
    df_contrast_summary.to_csv(output_dir / 'genesis_vs_td_bootstrap_summary.csv', index=False)
    build_all_diagnostic(df_trials).to_csv(output_dir / 'genesis_all_exact_mean_diagnostic.csv', index=False)
    df_tertiles, tertile_status = build_tertile_summary(df_environment)
    if not df_tertiles.empty:
        df_tertiles.to_csv(output_dir / 'reward_by_predictability_tertile.csv', index=False)
    (output_dir / 'predictability_tertile_status.txt').write_text(tertile_status, encoding='utf-8')

    selections = {}
    example_curves = {}
    for label, example_config in analysis_config['example_environments'].items():
        environment_id, status = select_example_environment(
            df_scores=df_scores,
            score_window=example_config['score_window'],
            seed=example_config['seed'],
            preferred_environment_id=example_config.get('preferred_environment_id'),
        )
        score = float(df_scores.set_index('environment_id').loc[environment_id, 'environment_score'])
        probabilities, trial_ids, curves = compute_example_curves(
            df_trials,
            environment_id=environment_id,
            model_ids=model_ids,
            reward_window=analysis_config['plots']['cumulative_reward_smoothing_window'],
            value_window=analysis_config['plots']['value_smoothing_window'],
        )
        selections[label] = {
            'environment_id': environment_id,
            'environment_score': score,
            'selection_status': status,
        }
        example_curves[label] = (probabilities, trial_ids, curves)
    pd.DataFrame.from_dict(selections, orient='index').to_csv(output_dir / 'example_environment_selection.csv')
    reward_limits = (
        shared_reward_ylim([values[2] for values in example_curves.values()])
        if analysis_config['plots']['shared_reward_ylim']
        else None
    )
    for label, values in example_curves.items():
        selection = selections[label]
        filename = (
            f'sampling-temperature_rewards_env-score-{label}.pdf'
            if analysis_config['plots']['use_historical_filenames']
            else f'model-budget_rewards_env-score-{label}.pdf'
        )
        save_example_plot(
            path=visual_dir / filename,
            probabilities=values[0],
            trial_ids=values[1],
            curves=values[2],
            model_ids=model_ids,
            colors=analysis_config['model_colors'],
            n_warmup=resolved_config['task']['n_random_warmup_trials'],
            reward_ylim=reward_limits,
            title=(
                f'{label.title()} predictability environment {selection["environment_id"]} '
                f'(score={selection["environment_score"]:.3f})'
            ),
            pdf_metadata={
                'Title': f'Supplementary {label} predictability bandit example',
                'Subject': json.dumps({
                    **selection,
                    'n_agents': resolved_config['n_agent_simulations'],
                    'model_ids': model_ids,
                    'sampling_temperature': resolved_config['genesis']['sampling_temperature'],
                }),
            },
            save_figures=save_figures,
        )
    save_budget_summary_plot(
        path=visual_dir / 'reward_by_sampling_budget.pdf',
        df_models=df_models,
        model_ids=model_ids,
        colors=analysis_config['model_colors'],
        save_figures=save_figures,
    )
    save_contrast_plot(
        path=visual_dir / 'genesis_td_difference_by_predictability.pdf',
        df_contrasts=df_contrasts,
        model_ids=model_ids,
        colors=analysis_config['model_colors'],
        save_figures=save_figures,
    )
    metadata = {
        'analysis_config': analysis_config,
        'simulation_record': str(record_path),
        'simulation_fingerprint': record['config_fingerprint'],
        'analysis_fingerprint': fingerprint,
        'resolved_simulation_config': resolved_config,
        'schema_version': record['schema_version'],
        'model_ids': model_ids,
        'checkpoints': record['checkpoints'],
        'random_stream_policy': record['random_stream_policy'],
        'inference': 'Environment-cluster percentile bootstrap; confidence intervals only; no equivalence test.',
    }
    np.save(output_dir / 'analysis_metadata.npy', metadata, allow_pickle=True)
    print(f'Analyzed {record["expected_complete_runs"]} model histories in {output_dir}')


def main(argv: list[str] | None = None) -> None:
    """
    Run supplementary analysis from command-line arguments.

    Args:
        argv (list[str] | None): Optional command-line arguments.
    """
    run_analysis(parse_args(argv))


if __name__ == '__main__':
    main()
