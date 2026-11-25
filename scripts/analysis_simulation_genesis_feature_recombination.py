"""
Analyze alpha-zero GENESIS feature-recombination reward reconstruction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path.cwd()))

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import filepaths
from utils.simulation_config import get_replacement_output_dir, load_simulation_config
from models.utils import softmax


DEFAULT_CONFIG_PATH = filepaths.ROOT / 'configs' / 'analysis_simulation_genesis_reward_reconstruction.yaml'
DEFAULT_BUDGET_COLORS = ('#2E338E', '#92297D', '#EA2264', '#F78D60', '#86BCBD', '#408175')
EXPERIMENT_ONE_BUDGET_COLORS = {
    1: '#2E338E',
    5: '#92297D',
    10: '#EA2264',
    20: '#F78D60',
    50: '#86BCBD',
}
OBSOLETE_FIGURE_FILENAMES = (
    'feature_recombination_supp.pdf',
    'feature_recombination_panel_1.pdf',
    'feature_recombination_panel_2.pdf',
    'feature_recombination_panel_3.pdf',
    'feature_recombination_panel_4.pdf',
    'feature_recombination_panel_5.pdf',
    'feature_recombination_panel_6.pdf',
)
OBSOLETE_TABLE_FILENAMES = (
    'episode_sampling_probabilities.csv',
    'session_reward_baselines.csv',
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse alpha-zero feature-recombination analysis arguments.
    """
    parser = argparse.ArgumentParser(description='Analyze GENESIS feature reward reconstruction.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--n-boot', type=int)
    parser.add_argument('--random-state', type=int)

    return parser.parse_args(argv)


def load_analysis_config(config_path: Path) -> dict[str, Any]:
    """
    Load the alpha-zero analysis configuration.
    """
    return yaml.safe_load(config_path.read_text(encoding='utf-8'))


def load_simulation_records(input_dir: Path) -> list[dict[str, Any]]:
    """
    Load all supported session records from one experiment directory.
    """
    records = [
        np.load(input_path, allow_pickle=True).item()
        for input_path in sorted(input_dir.glob('001_session-*.npy'))
    ]
    if not records:
        raise FileNotFoundError(f'no session records found in {input_dir}')
    if any(record['schema_version'] not in (2, 3) for record in records):
        raise ValueError('analysis requires simulation schema version 2 or 3')

    return records


def build_retrieval_repetition_table(
    *,
    records: list[dict[str, Any]],
    selected_temporal_weight: float,
) -> pd.DataFrame:
    """
    Build alpha-zero repetition-level reconstructed-reward outcomes.
    """
    rows = []

    for record in records:
        session = record['session']
        cube_reward_target = float(session['reward_assignment']['cube_session_mean_reward'])
        metadata = pd.DataFrame(session['session_metadata']).sort_values('session_position')
        cue_feature = session['reward_targets']['cue_feature']
        cue_value = session['reward_targets']['cue_value']
        is_cube = metadata[cue_feature].to_numpy() == cue_value
        if 'population_id' in session:
            population_id = session['population_id']
        else:
            population_id = f"balanced_{len(session['session_metadata'])}"

        for grid_cell in record['retrieval_grid']:
            cell_config = grid_cell['config']
            temporal_weight = float(cell_config['temporal_weight'])
            if not np.isclose(temporal_weight, selected_temporal_weight):
                continue

            for repetition in grid_cell['repetitions']:
                decoded_reward = float(repetition['decoded_recombined_reward'])
                semantic_target_error = decoded_reward - cube_reward_target
                sampled_episode_indices = np.asarray(repetition['sampled_episode_indices'], dtype=int)
                cube_query_hit_count = int(is_cube[sampled_episode_indices].sum())
                rows.append({
                    'session_id': int(session['session_id']),
                    'session_seed': int(session['session_seed']),
                    'population_id': population_id,
                    'retrieval_scope': cell_config.get('retrieval_scope', 'all_session_items'),
                    'sampling_budget': int(cell_config['sampling_budget']),
                    'repetition_id': int(repetition['repetition_id']),
                    'retrieval_seed': int(repetition['retrieval_seed']),
                    'decoded_recombined_reward': decoded_reward,
                    'cube_session_mean_reward': cube_reward_target,
                    'semantic_target_error': semantic_target_error,
                    'semantic_absolute_error': abs(semantic_target_error),
                    'cube_query_hit_count': cube_query_hit_count,
                    'cube_query_hit_rate': cube_query_hit_count / len(sampled_episode_indices),
                    'requested_sampling_budget': int(repetition['requested_sampling_budget']),
                    'effective_sample_count': int(repetition['effective_sample_count']),
                })

    if not rows:
        raise ValueError(f'no records found for temporal weight {selected_temporal_weight:g}')

    return pd.DataFrame(rows).sort_values([
        'session_id',
        'sampling_budget',
        'repetition_id',
    ]).reset_index(drop=True)


def build_cube_session_reward_targets(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Build one realized cube-reward target per session.
    """
    rows = [
        {
            'session_id': int(record['session']['session_id']),
            'cube_session_mean_reward': float(record['session']['reward_assignment']['cube_session_mean_reward']),
        }
        for record in records
    ]

    return pd.DataFrame(rows).sort_values('session_id').reset_index(drop=True)


def build_query_sampling_probability_table(
    *,
    records: list[dict[str, Any]],
    selected_temporal_weight: float,
    session_id: int,
    sampling_temperature: float | None = None,
) -> pd.DataFrame:
    """
    Recompute one session's query-cue sampling distribution at its stored temperature.
    """
    record = next(record for record in records if record['session']['session_id'] == session_id)
    session = record['session']
    cue_feature = session['reward_targets']['cue_feature']
    cue_value = session['reward_targets']['cue_value']
    grid_cell = min(
        (
            grid_cell
            for grid_cell in record['retrieval_grid']
            if np.isclose(grid_cell['config']['temporal_weight'], selected_temporal_weight)
        ),
        key=lambda grid_cell: grid_cell['config']['sampling_budget'],
    )
    if sampling_temperature is None:
        sampling_temperature = float(record['config']['sampling_temperature'])
    query_key_match_scores = np.asarray(
        grid_cell['retrieval_distribution']['query_key_match_scores'],
        dtype=float,
    )
    sampling_probabilities = softmax(query_key_match_scores, sampling_temperature)
    metadata = pd.DataFrame(session['session_metadata']).sort_values('session_position')

    return pd.DataFrame({
        'session_id': int(session_id),
        'session_position': metadata['session_position'].to_numpy(dtype=int),
        'is_cube': (metadata[cue_feature].to_numpy() == cue_value),
        'sampling_probability': sampling_probabilities,
        'sampling_temperature': sampling_temperature,
    })


def build_session_budget_summary(*, retrieval_repetitions: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse retrieval repetitions to one row per session and sampling budget.
    """
    group_columns = ['session_id', 'population_id', 'retrieval_scope', 'sampling_budget']
    rows = []

    for identifiers, group in retrieval_repetitions.groupby(group_columns, observed=True):
        session_id, population_id, retrieval_scope, sampling_budget = identifiers
        rows.append({
            'session_id': int(session_id),
            'population_id': population_id,
            'retrieval_scope': retrieval_scope,
            'sampling_budget': int(sampling_budget),
            'n_repetitions': int(len(group)),
            'mean_decoded_recombined_reward': float(group['decoded_recombined_reward'].mean()),
            'cube_session_mean_reward': float(group['cube_session_mean_reward'].iloc[0]),
            'semantic_target_mae': float(group['semantic_absolute_error'].mean()),
            'mean_cube_query_hit_rate': float(group['cube_query_hit_rate'].mean()),
            'session_mean_reconstruction_mae': abs(
                float(group['decoded_recombined_reward'].mean())
                - float(group['cube_session_mean_reward'].iloc[0]),
            ),
            'mean_absolute_percentage_accuracy': 100.0 * (
                1.0
                - abs(
                    float(group['decoded_recombined_reward'].mean())
                    - float(group['cube_session_mean_reward'].iloc[0]),
                )
                / float(group['cube_session_mean_reward'].iloc[0])
            ),
        })

    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def bootstrap_mean_standard_error(
    values: pd.Series,
    *,
    n_boot: int,
    random_state: int,
) -> tuple[float, float]:
    """
    Calculate the mean and bootstrap standard error.
    """
    value_array = values.to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    sample_indices = rng.integers(0, len(value_array), size=(n_boot, len(value_array)))
    bootstrap_means = value_array[sample_indices].mean(axis=1)
    return float(value_array.mean()), float(bootstrap_means.std(ddof=1))


def build_across_session_budget_summary(
    *,
    session_budget_summary: pd.DataFrame,
    n_boot: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Aggregate each sampling budget over independent session-level estimates.
    """
    condition_count = session_budget_summary[['population_id', 'retrieval_scope']].drop_duplicates().shape[0]
    if condition_count != 1:
        raise ValueError('analysis requires one population and retrieval scope per run')

    rows = []
    metric_names = (
        'mean_decoded_recombined_reward',
        'semantic_target_mae',
        'mean_cube_query_hit_rate',
        'session_mean_reconstruction_mae',
        'mean_absolute_percentage_accuracy',
    )
    for budget_index, (sampling_budget, group) in enumerate(
        session_budget_summary.groupby('sampling_budget', observed=True),
    ):
        row = {
            'sampling_budget': int(sampling_budget),
            'n_sessions': int(group['session_id'].nunique()),
        }
        for metric_index, metric_name in enumerate(metric_names):
            mean, bootstrap_se = bootstrap_mean_standard_error(
                group[metric_name],
                n_boot=n_boot,
                random_state=random_state + budget_index * len(metric_names) + metric_index,
            )
            row[metric_name] = mean
            row[f'{metric_name}_bootstrap_se'] = bootstrap_se
        rows.append(row)

    return pd.DataFrame(rows).sort_values('sampling_budget').reset_index(drop=True)


def get_sampling_budget_colors(
    *,
    sampling_budgets: list[int],
    configured_colors: dict[str, str],
) -> dict[int, Any]:
    """
    Build stable colors for the sampling-budget grid.
    """
    return {
        sampling_budget: configured_colors.get(
            str(sampling_budget),
            EXPERIMENT_ONE_BUDGET_COLORS.get(
                sampling_budget,
                DEFAULT_BUDGET_COLORS[index % len(DEFAULT_BUDGET_COLORS)],
            ),
        )
        for index, sampling_budget in enumerate(sampling_budgets)
    }


def style_axis(axis: plt.Axes) -> None:
    """
    Apply shared figure styling.
    """
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)


def plot_ground_truth_normalized_accuracy(
    *,
    axis: plt.Axes,
    summary: pd.DataFrame,
    budget_colors: dict[int, Any],
) -> None:
    """
    Plot session-normalized reconstruction accuracy against simulated rewards.
    """
    axis.plot(
        summary['sampling_budget'],
        summary['mean_absolute_percentage_accuracy'],
        color='#4B5563',
        linewidth=1.5,
        label='Mean absolute percentage accuracy',
        zorder=1,
    )
    for row in summary.itertuples(index=False):
        axis.errorbar(
            row.sampling_budget,
            row.mean_absolute_percentage_accuracy,
            yerr=row.mean_absolute_percentage_accuracy_bootstrap_se,
            color=budget_colors[row.sampling_budget],
            fmt='none',
            elinewidth=4.0,
            capsize=0,
            alpha=0.35,
            zorder=2,
        )
        axis.scatter(
            row.sampling_budget,
            row.mean_absolute_percentage_accuracy,
            color=budget_colors[row.sampling_budget],
            s=55,
            zorder=3,
        )
    axis.axhline(
        100.0,
        color='#408175',
        linestyle='--',
        linewidth=1.5,
        label='Exact reconstruction',
    )
    axis.set_xticks(summary['sampling_budget'])
    axis.set_xlabel('Sampling budget k')
    axis.set_ylabel('Mean absolute percentage accuracy (%)')
    axis.set_title('Ground-truth-normalized reconstruction accuracy')
    axis.legend(frameon=False, fontsize=8)
    style_axis(axis)


def plot_semantic_target_mae(
    *,
    axis: plt.Axes,
    summary: pd.DataFrame,
    budget_colors: dict[int, Any],
) -> None:
    """
    Plot session-conditioned cube-reward reconstruction MAE.
    """
    axis.plot(
        summary['sampling_budget'],
        summary['semantic_target_mae'],
        color='#4B5563',
        linewidth=1.5,
        zorder=1,
    )
    for row in summary.itertuples(index=False):
        axis.errorbar(
            row.sampling_budget,
            row.semantic_target_mae,
            yerr=row.semantic_target_mae_bootstrap_se,
            color=budget_colors[row.sampling_budget],
            fmt='none',
            elinewidth=4.0,
            capsize=0,
            alpha=0.35,
            label=f'k={row.sampling_budget}',
            zorder=2,
        )
        axis.scatter(
            row.sampling_budget,
            row.semantic_target_mae,
            color=budget_colors[row.sampling_budget],
            s=55,
            zorder=3,
        )
    axis.set_xticks(summary['sampling_budget'])
    axis.set_xlabel('Sampling budget k')
    axis.set_ylabel('Mean absolute error')
    axis.set_title('Cube-reward reconstruction error')
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, fontsize=8)
    style_axis(axis)


def plot_cube_query_hit_rate(
    *,
    axis: plt.Axes,
    summary: pd.DataFrame,
    budget_colors: dict[int, Any],
) -> None:
    """
    Plot the observed cube-query hit rate at the simulation temperature.
    """
    hit_rate_percent = 100.0 * summary['mean_cube_query_hit_rate']
    hit_rate_se_percent = 100.0 * summary['mean_cube_query_hit_rate_bootstrap_se']
    lower_bound = float((hit_rate_percent - hit_rate_se_percent).min())
    upper_bound = float((hit_rate_percent + hit_rate_se_percent).max())
    padding = max((upper_bound - lower_bound) * 0.1, 0.01)
    axis.plot(
        summary['sampling_budget'],
        hit_rate_percent,
        color='#4B5563',
        linewidth=1.5,
        zorder=1,
    )
    for row in summary.itertuples(index=False):
        axis.errorbar(
            row.sampling_budget,
            100.0 * row.mean_cube_query_hit_rate,
            yerr=100.0 * row.mean_cube_query_hit_rate_bootstrap_se,
            color=budget_colors[row.sampling_budget],
            fmt='none',
            elinewidth=4.0,
            capsize=0,
            alpha=0.35,
            zorder=2,
        )
        axis.scatter(
            row.sampling_budget,
            100.0 * row.mean_cube_query_hit_rate,
            color=budget_colors[row.sampling_budget],
            s=55,
            zorder=3,
        )
    axis.set_xticks(summary['sampling_budget'])
    axis.set_xlabel('Sampling budget k')
    axis.set_ylabel('Mean cube-query hit rate (%)')
    axis.set_title('Cube-query hit rate (T=0.25)')
    axis.set_ylim(max(0.0, lower_bound - padding), min(100.0, upper_bound + padding))
    style_axis(axis)


def plot_session_cube_reward_distribution(
    *,
    axis: plt.Axes,
    cube_reward_targets: pd.DataFrame,
    summary: pd.DataFrame,
    budget_colors: dict[int, Any],
) -> None:
    """
    Plot the realized session cube-reward distribution and reconstructed means.
    """
    cube_rewards = cube_reward_targets['cube_session_mean_reward']
    axis.hist(cube_rewards, bins='auto', color='#A7D1C7', edgecolor='white', alpha=0.9, label='Session cube rewards')
    cube_reward_target = float(cube_rewards.mean())
    axis.axvline(
        cube_reward_target,
        color='#408175',
        linestyle='--',
        linewidth=1.5,
        label='Mean cube ground-truth reward',
    )
    for row in summary.itertuples(index=False):
        axis.axvline(
            row.mean_decoded_recombined_reward,
            color=budget_colors[row.sampling_budget],
            linewidth=2.0,
            label=f'Reconstructed reward, k={row.sampling_budget}',
        )
    axis.set_xlabel('Cube-shape reward')
    axis.set_ylabel('Sessions')
    axis.set_title('Session cube-reward distribution')
    axis.legend(frameon=False, fontsize=8)
    style_axis(axis)


def plot_session_reward_calibration(
    *,
    axes: np.ndarray,
    session_budget_summary: pd.DataFrame,
    budget_colors: dict[int, Any],
) -> None:
    """
    Compare each session's true and reconstructed cube reward by budget.
    """
    budgets = sorted(session_budget_summary['sampling_budget'].unique())
    all_rewards = pd.concat([
        session_budget_summary['cube_session_mean_reward'],
        session_budget_summary['mean_decoded_recombined_reward'],
    ])
    reward_range = float(all_rewards.max() - all_rewards.min())
    padding = max(reward_range * 0.08, 0.01)
    limits = (float(all_rewards.min()) - padding, float(all_rewards.max()) + padding)

    for axis, sampling_budget in zip(axes, budgets):
        group = session_budget_summary.loc[session_budget_summary['sampling_budget'] == sampling_budget]
        axis.scatter(
            group['cube_session_mean_reward'],
            group['mean_decoded_recombined_reward'],
            color=budget_colors[int(sampling_budget)],
            alpha=0.8,
        )
        axis.plot(limits, limits, color='#4B5563', linestyle='--', linewidth=1.2)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_xlabel('True cube reward')
        axis.set_ylabel('Mean reconstructed cube reward')
        axis.set_title(f'k={sampling_budget}')
        style_axis(axis)


def plot_query_sampling_probabilities(*, axis: plt.Axes, probabilities: pd.DataFrame) -> None:
    """
    Plot the cube-cue softmax distribution over all stored session episodes.
    """
    cube_probabilities = probabilities[probabilities['is_cube']]
    non_cube_probabilities = probabilities[~probabilities['is_cube']]
    axis.bar(
        non_cube_probabilities['session_position'],
        non_cube_probabilities['sampling_probability'],
        color='#D1D5DB',
        width=1.0,
        linewidth=0,
        label='Other shapes',
    )
    axis.bar(
        cube_probabilities['session_position'],
        cube_probabilities['sampling_probability'],
        color='#408175',
        width=1.0,
        linewidth=0,
        label='Cube items',
    )
    sampling_temperature = float(probabilities['sampling_temperature'].iloc[0])
    axis.set_xlabel('Stored episode position')
    axis.set_ylabel('Sampling probability')
    axis.set_title(f'Cube-cue retrieval probabilities (T={sampling_temperature:g})')
    axis.legend(frameon=False, fontsize=8, loc='lower right')
    style_axis(axis)


def save_figure(*, figure: plt.Figure, savepath: Path) -> None:
    """
    Save and close one analysis figure.
    """
    figure.savefig(savepath, bbox_inches='tight')
    plt.close(figure)


def generate_figures(
    *,
    summary: pd.DataFrame,
    session_budget_summary: pd.DataFrame,
    cube_reward_targets: pd.DataFrame,
    query_probabilities: pd.DataFrame,
    visual_dir: Path,
    analysis_config: dict[str, Any],
) -> None:
    """
    Generate the reconstruction summary with session-level calibration panels.
    """
    budget_colors = get_sampling_budget_colors(
        sampling_budgets=summary['sampling_budget'].tolist(),
        configured_colors=analysis_config.get('sampling_budget_colors', {}),
    )
    figure = plt.figure(figsize=(18, 18), layout='constrained')
    grid = figure.add_gridspec(nrows=4, ncols=2)
    top_left_axis = figure.add_subplot(grid[0, 0])
    top_right_axis = figure.add_subplot(grid[0, 1])
    bottom_left_axis = figure.add_subplot(grid[1, 0])
    bottom_right_axis = figure.add_subplot(grid[1, 1])
    hit_rate_axis = figure.add_subplot(grid[2, :])
    calibration_grid = grid[3, :].subgridspec(nrows=1, ncols=len(budget_colors))
    calibration_axes = np.asarray([figure.add_subplot(cell) for cell in calibration_grid])

    plot_ground_truth_normalized_accuracy(
        axis=top_left_axis,
        summary=summary,
        budget_colors=budget_colors,
    )
    plot_semantic_target_mae(axis=top_right_axis, summary=summary, budget_colors=budget_colors)
    plot_query_sampling_probabilities(axis=bottom_left_axis, probabilities=query_probabilities)
    plot_session_cube_reward_distribution(
        axis=bottom_right_axis,
        cube_reward_targets=cube_reward_targets,
        summary=summary,
        budget_colors=budget_colors,
    )
    plot_cube_query_hit_rate(axis=hit_rate_axis, summary=summary, budget_colors=budget_colors)
    plot_session_reward_calibration(
        axes=calibration_axes,
        session_budget_summary=session_budget_summary,
        budget_colors=budget_colors,
    )
    save_figure(figure=figure, savepath=visual_dir / 'feature_recombination_summary.pdf')


def remove_obsolete_artifacts(*, output_dir: Path, visual_dir: Path) -> None:
    """
    Remove figures and tables not produced by the focused analysis.
    """
    for filename in OBSOLETE_FIGURE_FILENAMES:
        (visual_dir / filename).unlink(missing_ok=True)
    for filename in OBSOLETE_TABLE_FILENAMES:
        (output_dir / filename).unlink(missing_ok=True)


def run_analysis(args: argparse.Namespace) -> None:
    """
    Export alpha-zero reward-reconstruction summaries and one combined figure.
    """
    analysis_config = load_analysis_config(args.config)
    if args.input_dir is None or args.output_dir is None:
        simulation_config_path = Path(analysis_config['simulation_config_path'])
        simulation_config = load_simulation_config(simulation_config_path)
    if args.input_dir is None:
        input_dir = get_replacement_output_dir(
            root_dir=filepaths.EXPERIMENTS_DIR,
            output_name=simulation_config['experiment_name'],
            sample_replacement=simulation_config['sample_replacement'],
        )
    else:
        input_dir = args.input_dir
    if args.output_dir is None:
        output_dir = get_replacement_output_dir(
            root_dir=filepaths.OUTPUT_DIR,
            output_name=analysis_config['output_dir_name'],
            sample_replacement=simulation_config['sample_replacement'],
        )
    else:
        output_dir = args.output_dir
    n_boot = args.n_boot if args.n_boot is not None else analysis_config['n_boot']
    random_state = args.random_state if args.random_state is not None else analysis_config['random_state']
    selected_temporal_weight = float(analysis_config['selected_temporal_weight'])
    records = load_simulation_records(input_dir)
    visual_dir = output_dir / 'visuals'
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    retrieval_repetitions = build_retrieval_repetition_table(
        records=records,
        selected_temporal_weight=selected_temporal_weight,
    )
    session_budget_summary = build_session_budget_summary(retrieval_repetitions=retrieval_repetitions)
    across_session_summary = build_across_session_budget_summary(
        session_budget_summary=session_budget_summary,
        n_boot=n_boot,
        random_state=random_state,
    )
    cube_reward_targets = build_cube_session_reward_targets(records)
    query_probabilities = build_query_sampling_probability_table(
        records=records,
        selected_temporal_weight=selected_temporal_weight,
        session_id=int(min(record['session']['session_id'] for record in records)),
    )
    temperature_one_query_probabilities = build_query_sampling_probability_table(
        records=records,
        selected_temporal_weight=selected_temporal_weight,
        session_id=int(min(record['session']['session_id'] for record in records)),
        sampling_temperature=1.0,
    )

    remove_obsolete_artifacts(output_dir=output_dir, visual_dir=visual_dir)
    retrieval_repetitions.to_csv(output_dir / 'retrieval_repetitions.csv', index=False)
    session_budget_summary.to_csv(output_dir / 'session_grid_summary.csv', index=False)
    across_session_summary.to_csv(output_dir / 'across_session_grid_summary.csv', index=False)
    cube_reward_targets.to_csv(output_dir / 'cube_session_reward_targets.csv', index=False)
    query_probabilities.to_csv(output_dir / 'query_sampling_probabilities.csv', index=False)
    generate_figures(
        summary=across_session_summary,
        session_budget_summary=session_budget_summary,
        cube_reward_targets=cube_reward_targets,
        query_probabilities=temperature_one_query_probabilities,
        visual_dir=visual_dir,
        analysis_config=analysis_config,
    )


def main(argv: list[str] | None = None) -> None:
    """
    Run the feature-recombination analysis.
    """
    run_analysis(parse_args(argv))


if __name__ == '__main__':
    main()
