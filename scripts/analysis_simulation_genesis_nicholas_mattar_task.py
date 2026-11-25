"""
Analyze GENESIS Nicholas-Mattar feature-choice simulations.
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


DEFAULT_CONFIG_PATH = filepaths.ROOT / 'configs' / 'analysis_simulation_genesis_nicholas_mattar_task.yaml'
DEFAULT_BUDGET_COLORS = ('#2E338E', '#92297D', '#EA2264', '#F78D60', '#86BCBD')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse Nicholas-Mattar analysis arguments.
    """
    parser = argparse.ArgumentParser(description='Analyze GENESIS Nicholas-Mattar feature choices.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--n-boot', type=int)
    parser.add_argument('--random-state', type=int)

    return parser.parse_args(argv)


def load_analysis_config(config_path: Path) -> dict[str, Any]:
    """
    Load one Nicholas-Mattar analysis configuration.
    """
    return yaml.safe_load(config_path.read_text(encoding='utf-8'))


def load_simulation_records(input_dir: Path) -> list[dict[str, Any]]:
    """
    Load saved Nicholas-Mattar session records.
    """
    records = [
        np.load(input_path, allow_pickle=True).item()
        for input_path in sorted(input_dir.glob('001_session-*.npy'))
    ]
    if not records:
        raise FileNotFoundError(f'no session records found in {input_dir}')
    if any(record['schema_version'] != 1 for record in records):
        raise ValueError('analysis requires simulation schema version 1')

    return records


def get_session_target_map(session: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Index configured target definitions by target name.
    """
    return {target['name']: target for target in session['targets']}


def build_choice_repetition_table(
    *,
    records: list[dict[str, Any]],
    selected_temporal_weight: float,
) -> pd.DataFrame:
    """
    Flatten paired cube-sphere reconstruction outcomes by repetition.
    """
    rows = []

    for record in records:
        session = record['session']
        reward_assignment = session['reward_assignment']
        true_reward_gap = float(reward_assignment['target_mean_gap'])

        for grid_cell in record['choice_grid']:
            config = grid_cell['config']
            if not np.isclose(float(config['temporal_weight']), selected_temporal_weight):
                continue

            for repetition in grid_cell['repetitions']:
                target_retrievals = repetition['target_retrievals']
                cube_reward = float(target_retrievals['cube']['decoded_recombined_reward'])
                sphere_reward = float(target_retrievals['sphere']['decoded_recombined_reward'])
                estimated_reward_gap = float(repetition['estimated_reward_gap'])
                rows.append({
                    'session_id': int(session['session_id']),
                    'session_seed': int(session['session_seed']),
                    'required_oracle_choice': session['required_oracle_choice'],
                    'oracle_choice': reward_assignment['oracle_choice'],
                    'sampling_budget': int(config['sampling_budget']),
                    'repetition_id': int(repetition['repetition_id']),
                    'cube_reconstructed_reward': cube_reward,
                    'sphere_reconstructed_reward': sphere_reward,
                    'true_reward_gap': true_reward_gap,
                    'estimated_reward_gap': estimated_reward_gap,
                    'gap_error': estimated_reward_gap - true_reward_gap,
                    'absolute_gap_error': abs(estimated_reward_gap - true_reward_gap),
                    'estimated_gap_confidence': abs(estimated_reward_gap),
                    'chosen_target': repetition['chosen_target'],
                    'is_correct': bool(repetition['is_correct']),
                })

    if not rows:
        raise ValueError(f'no records found for temporal weight {selected_temporal_weight:g}')

    return pd.DataFrame(rows).sort_values([
        'session_id',
        'sampling_budget',
        'repetition_id',
    ]).reset_index(drop=True)


def build_target_retrieval_table(
    *,
    records: list[dict[str, Any]],
    selected_temporal_weight: float,
) -> pd.DataFrame:
    """
    Flatten target-specific sampled-memory outcomes by repetition.
    """
    rows = []

    for record in records:
        session = record['session']
        metadata = pd.DataFrame(session['session_metadata']).set_index('session_position')
        targets = get_session_target_map(session)

        for grid_cell in record['choice_grid']:
            config = grid_cell['config']
            if not np.isclose(float(config['temporal_weight']), selected_temporal_weight):
                continue

            for repetition in grid_cell['repetitions']:
                for target_name, retrieval in repetition['target_retrievals'].items():
                    target = targets[target_name]
                    sampled_indices = retrieval['sampled_episode_indices']
                    sampled_target_fraction = float(np.mean([
                        metadata.loc[sampled_index, target['feature']] == target['value']
                        for sampled_index in sampled_indices
                    ]))
                    rows.append({
                        'session_id': int(session['session_id']),
                        'sampling_budget': int(config['sampling_budget']),
                        'repetition_id': int(repetition['repetition_id']),
                        'target_name': target_name,
                        'target_feature': target['feature'],
                        'target_value': int(target['value']),
                        'retrieval_seed': int(retrieval['retrieval_seed']),
                        'sampled_episode_indices': sampled_indices,
                        'effective_sample_count': int(retrieval['effective_sample_count']),
                        'sampled_target_fraction': sampled_target_fraction,
                    })

    return pd.DataFrame(rows).sort_values([
        'session_id',
        'sampling_budget',
        'repetition_id',
        'target_name',
    ]).reset_index(drop=True)


def build_session_budget_summary(
    *,
    choice_repetitions: pd.DataFrame,
    target_retrievals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize paired reconstruction and retrieval outcomes per session and budget.
    """
    rows = []

    for identifiers, group in choice_repetitions.groupby(['session_id', 'sampling_budget'], observed=True):
        session_id, sampling_budget = identifiers
        retrieval_group = target_retrievals.loc[
            (target_retrievals['session_id'] == session_id)
            & (target_retrievals['sampling_budget'] == sampling_budget)
        ]
        hit_rates = retrieval_group.groupby('target_name', observed=True)['sampled_target_fraction'].mean()
        rows.append({
            'session_id': int(session_id),
            'sampling_budget': int(sampling_budget),
            'required_oracle_choice': group['required_oracle_choice'].iloc[0],
            'oracle_choice': group['oracle_choice'].iloc[0],
            'true_reward_gap': float(group['true_reward_gap'].iloc[0]),
            'n_repetitions': int(len(group)),
            'mean_estimated_reward_gap': float(group['estimated_reward_gap'].mean()),
            'estimated_gap_standard_deviation': float(group['estimated_reward_gap'].std(ddof=1)),
            'choice_accuracy': float(group['is_correct'].mean()),
            'mean_gap_error': float(group['gap_error'].mean()),
            'gap_mae': float(group['absolute_gap_error'].mean()),
            'gap_rmse': float(np.sqrt(np.mean(np.square(group['gap_error'])))),
            'mean_estimated_gap_confidence': float(group['estimated_gap_confidence'].mean()),
            'cube_sampled_target_fraction': float(hit_rates['cube']),
            'sphere_sampled_target_fraction': float(hit_rates['sphere']),
        })

    return pd.DataFrame(rows).sort_values(['session_id', 'sampling_budget']).reset_index(drop=True)


def build_retrieval_distribution_summary(
    *,
    records: list[dict[str, Any]],
    selected_temporal_weight: float,
) -> pd.DataFrame:
    """
    Calculate target probability mass from stored retrieval distributions.
    """
    rows = []

    for record in records:
        session = record['session']
        metadata = pd.DataFrame(session['session_metadata']).sort_values('session_position')
        targets = get_session_target_map(session)

        for grid_cell in record['choice_grid']:
            config = grid_cell['config']
            if not np.isclose(float(config['temporal_weight']), selected_temporal_weight):
                continue

            for target_name, distribution in grid_cell['retrieval_distributions'].items():
                target = targets[target_name]
                probabilities = np.asarray(distribution['sampling_probabilities'], dtype=float)
                target_mask = metadata[target['feature']].to_numpy() == target['value']
                rows.append({
                    'session_id': int(session['session_id']),
                    'sampling_budget': int(config['sampling_budget']),
                    'target_name': target_name,
                    'target_probability_mass': float(probabilities[target_mask].sum()),
                })

    return pd.DataFrame(rows).sort_values(['session_id', 'sampling_budget', 'target_name']).reset_index(drop=True)


def bootstrap_standard_error(*, values: pd.Series, n_boot: int, random_state: int) -> float:
    """
    Estimate the bootstrap standard error of one session-level mean.
    """
    value_array = values.to_numpy(dtype=float)
    random_generator = np.random.default_rng(random_state)
    sample_indices = random_generator.integers(0, len(value_array), size=(n_boot, len(value_array)))

    return float(value_array[sample_indices].mean(axis=1).std(ddof=1))


def build_budget_summary(
    *,
    session_budget_summary: pd.DataFrame,
    n_boot: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Aggregate session-level performance metrics by sampling budget.
    """
    metric_names = (
        'choice_accuracy',
        'mean_gap_error',
        'gap_mae',
        'gap_rmse',
        'estimated_gap_standard_deviation',
        'mean_estimated_gap_confidence',
        'cube_sampled_target_fraction',
        'sphere_sampled_target_fraction',
    )
    rows = []

    for budget_index, (sampling_budget, group) in enumerate(
        session_budget_summary.groupby('sampling_budget', observed=True),
    ):
        row = {
            'sampling_budget': int(sampling_budget),
            'n_sessions': int(group['session_id'].nunique()),
        }
        for metric_index, metric_name in enumerate(metric_names):
            row[metric_name] = float(group[metric_name].mean())
            row[f'{metric_name}_bootstrap_se'] = bootstrap_standard_error(
                values=group[metric_name],
                n_boot=n_boot,
                random_state=random_state + budget_index * len(metric_names) + metric_index,
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values('sampling_budget').reset_index(drop=True)


def bootstrap_correlation_standard_error(
    *,
    x: pd.Series,
    y: pd.Series,
    n_boot: int,
    random_state: int,
) -> float:
    """
    Estimate the bootstrap standard error of a Pearson correlation.
    """
    x_array = x.to_numpy(dtype=float)
    y_array = y.to_numpy(dtype=float)
    random_generator = np.random.default_rng(random_state)
    correlations = []

    for sample_indices in random_generator.integers(0, len(x_array), size=(n_boot, len(x_array))):
        sampled_x = x_array[sample_indices]
        sampled_y = y_array[sample_indices]
        if np.std(sampled_x) > 0.0 and np.std(sampled_y) > 0.0:
            correlations.append(np.corrcoef(sampled_x, sampled_y)[0, 1])

    return float(np.std(correlations, ddof=1)) if len(correlations) > 1 else np.nan


def build_correlation_summary(
    *,
    session_budget_summary: pd.DataFrame,
    n_boot: int,
    random_state: int,
    absolute_gaps: bool = False,
) -> pd.DataFrame:
    """
    Relate true session gaps to mean reconstructed gaps by budget.
    """
    rows = []

    for budget_index, (sampling_budget, group) in enumerate(
        session_budget_summary.groupby('sampling_budget', observed=True),
    ):
        true_gaps = group['true_reward_gap']
        estimated_gaps = group['mean_estimated_reward_gap']
        if absolute_gaps:
            true_gaps = true_gaps.abs()
            estimated_gaps = estimated_gaps.abs()
        session_gap_errors = estimated_gaps - true_gaps
        if len(group) > 1 and true_gaps.nunique() > 1 and estimated_gaps.nunique() > 1:
            pearson_r = float(np.corrcoef(true_gaps, estimated_gaps)[0, 1])
            spearman_r = float(true_gaps.rank().corr(estimated_gaps.rank()))
            bootstrap_se = bootstrap_correlation_standard_error(
                x=true_gaps,
                y=estimated_gaps,
                n_boot=n_boot,
                random_state=random_state + budget_index,
            )
        else:
            pearson_r = np.nan
            spearman_r = np.nan
            bootstrap_se = np.nan
        rows.append({
            'sampling_budget': int(sampling_budget),
            'n_sessions': int(len(group)),
            'pearson_r': pearson_r,
            'spearman_r': spearman_r,
            'r_squared': pearson_r ** 2 if np.isfinite(pearson_r) else np.nan,
            'session_gap_mae': float(session_gap_errors.abs().mean()),
            'session_gap_rmse': float(np.sqrt(np.mean(np.square(session_gap_errors)))),
            'pearson_r_bootstrap_se': bootstrap_se,
        })

    return pd.DataFrame(rows).sort_values('sampling_budget').reset_index(drop=True)


def build_continuous_relationships(*, session_budget_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Export session-level margin and confidence relationships.
    """
    result = session_budget_summary.loc[:, [
        'session_id',
        'sampling_budget',
        'true_reward_gap',
        'choice_accuracy',
        'mean_estimated_gap_confidence',
    ]].copy()
    result['true_gap_margin'] = result['true_reward_gap'].abs()

    return result


def get_sampling_budget_colors(
    *,
    sampling_budgets: list[int],
    configured_colors: dict[str, str],
) -> dict[int, str]:
    """
    Resolve stable colors for configured sampling budgets.
    """
    return {
        sampling_budget: configured_colors.get(
            str(sampling_budget),
            DEFAULT_BUDGET_COLORS[index % len(DEFAULT_BUDGET_COLORS)],
        )
        for index, sampling_budget in enumerate(sampling_budgets)
    }


def style_axis(axis: plt.Axes) -> None:
    """
    Apply shared minimal axis styling.
    """
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)


def save_figure(*, figure: plt.Figure, savepath: Path) -> None:
    """
    Save and close one analysis figure.
    """
    figure.savefig(savepath, bbox_inches='tight')
    plt.close(figure)


def plot_true_reward_gap_distribution(*, session_budget_summary: pd.DataFrame, savepath: Path) -> None:
    """
    Plot the signed distribution of true cube-minus-sphere reward gaps.
    """
    true_gaps = session_budget_summary.drop_duplicates('session_id')['true_reward_gap']
    figure, axis = plt.subplots(figsize=(7, 5), layout='constrained')
    axis.hist(true_gaps, bins='auto', color='#4B5563', edgecolor='white')
    axis.axvline(0.0, color='#111827', linestyle='--', linewidth=1.2)
    axis.set_xlabel('True reward gap: cube minus sphere')
    axis.set_ylabel('Sessions')
    axis.set_title('Session-level feature reward dominance')
    style_axis(axis)
    save_figure(figure=figure, savepath=savepath)


def plot_gap_reconstruction(
    *,
    session_budget_summary: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    absolute_correlation_summary: pd.DataFrame,
    budget_colors: dict[int, str],
    savepath: Path,
) -> None:
    """
    Plot signed and absolute true versus reconstructed reward gaps by budget.
    """
    budgets = sorted(session_budget_summary['sampling_budget'].unique())
    figure, axes = plt.subplots(nrows=2, ncols=len(budgets), figsize=(5 * len(budgets), 9), layout='constrained')
    axes = np.atleast_2d(axes)
    gap_views = (
        (
            session_budget_summary['true_reward_gap'],
            session_budget_summary['mean_estimated_reward_gap'],
            correlation_summary,
            'True cube-minus-sphere gap',
            'Mean reconstructed gap',
            False,
        ),
        (
            session_budget_summary['true_reward_gap'].abs(),
            session_budget_summary['mean_estimated_reward_gap'].abs(),
            absolute_correlation_summary,
            'Absolute true cube-minus-sphere gap',
            'Mean absolute reconstructed gap',
            True,
        ),
    )

    for row_axes, (true_gaps, estimated_gaps, statistics, x_label, y_label, is_absolute) in zip(axes, gap_views):
        all_gaps = pd.concat([true_gaps, estimated_gaps])
        limits = (0.0, float(all_gaps.max()) + 0.1) if is_absolute else (
            float(all_gaps.min()) - 0.1,
            float(all_gaps.max()) + 0.1,
        )
        for axis, sampling_budget in zip(row_axes, budgets):
            group = session_budget_summary.loc[session_budget_summary['sampling_budget'] == sampling_budget]
            fit_statistics = statistics.loc[statistics['sampling_budget'] == sampling_budget].iloc[0]
            axis.scatter(
                group['true_reward_gap'].abs() if is_absolute else group['true_reward_gap'],
                group['mean_estimated_reward_gap'].abs()
                if is_absolute
                else group['mean_estimated_reward_gap'],
                color=budget_colors[sampling_budget],
                alpha=0.8,
            )
            axis.plot(limits, limits, color='#111827', linestyle='--', linewidth=1.2)
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.set_title(f'k={sampling_budget}')
            axis.text(
                0.03,
                0.97,
                '\n'.join([
                    f"Pearson r = {fit_statistics['pearson_r']:.3f}",
                    f"Spearman ρ = {fit_statistics['spearman_r']:.3f}",
                    f"R² = {fit_statistics['r_squared']:.3f}",
                    f"MAE = {fit_statistics['session_gap_mae']:.3f}",
                    f"RMSE = {fit_statistics['session_gap_rmse']:.3f}",
                ]),
                transform=axis.transAxes,
                va='top',
                fontsize=8,
                bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'},
            )
            style_axis(axis)

    save_figure(figure=figure, savepath=savepath)


def plot_choice_accuracy_and_error(
    *,
    budget_summary: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    budget_colors: dict[int, str],
    savepath: Path,
) -> None:
    """
    Plot budget-level choice accuracy, reconstruction error, and correlation.
    """
    figure, axes = plt.subplots(ncols=3, figsize=(15, 4.5), layout='constrained')
    budgets = budget_summary['sampling_budget']

    axes[0].errorbar(
        budgets,
        budget_summary['choice_accuracy'],
        yerr=budget_summary['choice_accuracy_bootstrap_se'],
        color='#4B5563',
        marker='o',
    )
    axes[0].axhline(0.5, color='#111827', linestyle='--', linewidth=1.2, label='Chance')
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xlabel('Sampling budget k')
    axes[0].set_ylabel('Choice accuracy')
    axes[0].set_title('Choice accuracy')
    axes[0].legend(frameon=False)

    for metric_name, label in (
        ('mean_gap_error', 'Signed bias'),
        ('gap_mae', 'MAE'),
        ('gap_rmse', 'RMSE'),
    ):
        axes[1].errorbar(
            budgets,
            budget_summary[metric_name],
            yerr=budget_summary[f'{metric_name}_bootstrap_se'],
            marker='o',
            label=label,
        )
    axes[1].axhline(0.0, color='#111827', linestyle='--', linewidth=1.2)
    axes[1].set_xlabel('Sampling budget k')
    axes[1].set_ylabel('Reward-gap error')
    axes[1].set_title('Reconstruction error')
    axes[1].legend(frameon=False)

    axes[2].errorbar(
        correlation_summary['sampling_budget'],
        correlation_summary['pearson_r'],
        yerr=correlation_summary['pearson_r_bootstrap_se'],
        color='#4B5563',
        marker='o',
    )
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].set_xlabel('Sampling budget k')
    axes[2].set_ylabel('Pearson r')
    axes[2].set_title('Gap reconstruction correlation')

    for axis in axes:
        axis.set_xticks(budgets)
        style_axis(axis)

    save_figure(figure=figure, savepath=savepath)


def plot_polynomial_accuracy_by_true_margin(
    *,
    axis: plt.Axes,
    relationships: pd.DataFrame,
    budget_colors: dict[int, str],
) -> None:
    """
    Plot session accuracy against true margin with quadratic confidence bands.
    """
    for sampling_budget, group in relationships.groupby('sampling_budget', observed=True):
        margins = group['true_gap_margin'].to_numpy(dtype=float)
        accuracy = group['choice_accuracy'].to_numpy(dtype=float)
        coefficients, covariance = np.polyfit(margins, accuracy, deg=2, cov=True)
        fitted_margins = np.linspace(margins.min(), margins.max(), num=200)
        fitted_accuracy = np.polyval(coefficients, fitted_margins)
        design_matrix = np.vander(fitted_margins, N=3)
        fitted_standard_error = np.sqrt(np.einsum(
            'ij,jk,ik->i',
            design_matrix,
            covariance,
            design_matrix,
        ))
        color = budget_colors[int(sampling_budget)]

        axis.scatter(
            margins,
            accuracy,
            color=color,
            alpha=0.8,
        )
        axis.plot(
            fitted_margins,
            fitted_accuracy,
            color=color,
            label=f'k={sampling_budget}',
        )
        axis.fill_between(
            fitted_margins,
            fitted_accuracy - 1.96 * fitted_standard_error,
            fitted_accuracy + 1.96 * fitted_standard_error,
            color=color,
            alpha=0.2,
        )

    axis.set_xlabel('Absolute true reward gap')
    axis.set_ylabel('Session choice accuracy')
    axis.set_title('Accuracy by true margin (quadratic fit)')
    axis.legend(frameon=False, fontsize=7)
    style_axis(axis)


def plot_margin_confidence_variability(
    *,
    relationships: pd.DataFrame,
    budget_summary: pd.DataFrame,
    budget_colors: dict[int, str],
    savepath: Path,
) -> None:
    """
    Plot continuous margin and confidence relationships with sampling variability.
    """
    figure, axes = plt.subplots(ncols=4, figsize=(20, 4.5), layout='constrained')

    for sampling_budget, group in relationships.groupby('sampling_budget', observed=True):
        axes[0].scatter(
            group['true_gap_margin'],
            group['choice_accuracy'],
            color=budget_colors[int(sampling_budget)],
            label=f'k={sampling_budget}',
            alpha=0.8,
        )
        axes[2].scatter(
            group['mean_estimated_gap_confidence'],
            group['choice_accuracy'],
            color=budget_colors[int(sampling_budget)],
            label=f'k={sampling_budget}',
            alpha=0.8,
        )
    axes[0].set_xlabel('Absolute true reward gap')
    axes[0].set_ylabel('Session choice accuracy')
    axes[0].set_title('Accuracy by true margin')
    plot_polynomial_accuracy_by_true_margin(
        axis=axes[1],
        relationships=relationships,
        budget_colors=budget_colors,
    )
    axes[2].set_xlabel('Mean absolute reconstructed gap')
    axes[2].set_ylabel('Session choice accuracy')
    axes[2].set_title('Confidence calibration')
    axes[2].legend(frameon=False)
    axes[3].errorbar(
        budget_summary['sampling_budget'],
        budget_summary['estimated_gap_standard_deviation'],
        yerr=budget_summary['estimated_gap_standard_deviation_bootstrap_se'],
        color='#4B5563',
        marker='o',
    )
    axes[3].set_xlabel('Sampling budget k')
    axes[3].set_ylabel('Within-session gap standard deviation')
    axes[3].set_title('Sampling variability')

    for axis in axes:
        style_axis(axis)

    save_figure(figure=figure, savepath=savepath)


def plot_semantic_retrieval_quality(
    *,
    budget_summary: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    savepath: Path,
) -> None:
    """
    Plot target probability mass and sampled target-hit fractions.
    """
    figure, axes = plt.subplots(ncols=2, figsize=(10, 4.5), layout='constrained')
    budgets = budget_summary['sampling_budget']
    axes[0].plot(budgets, budget_summary['cube_sampled_target_fraction'], marker='o', label='Cube query')
    axes[0].plot(budgets, budget_summary['sphere_sampled_target_fraction'], marker='o', label='Sphere query')
    axes[0].set_xlabel('Sampling budget k')
    axes[0].set_ylabel('Sampled target fraction')
    axes[0].set_title('Sampled retrieval precision')
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(frameon=False)

    for target_name, group in distribution_summary.groupby('target_name', observed=True):
        mass_by_budget = group.groupby('sampling_budget', observed=True)['target_probability_mass'].mean()
        axes[1].plot(mass_by_budget.index, mass_by_budget, marker='o', label=f'{target_name.title()} query')
    axes[1].set_xlabel('Sampling budget k')
    axes[1].set_ylabel('Target probability mass')
    axes[1].set_title('Semantic retrieval concentration')
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.set_xticks(budgets)
        style_axis(axis)

    save_figure(figure=figure, savepath=savepath)


def run_analysis(args: argparse.Namespace) -> None:
    """
    Export Nicholas-Mattar statistics and summary figures.
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
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / 'visuals'
    visual_dir.mkdir(parents=True, exist_ok=True)
    records = load_simulation_records(input_dir)
    choice_repetitions = build_choice_repetition_table(
        records=records,
        selected_temporal_weight=selected_temporal_weight,
    )
    target_retrievals = build_target_retrieval_table(
        records=records,
        selected_temporal_weight=selected_temporal_weight,
    )
    session_budget_summary = build_session_budget_summary(
        choice_repetitions=choice_repetitions,
        target_retrievals=target_retrievals,
    )
    budget_summary = build_budget_summary(
        session_budget_summary=session_budget_summary,
        n_boot=n_boot,
        random_state=random_state,
    )
    correlation_summary = build_correlation_summary(
        session_budget_summary=session_budget_summary,
        n_boot=n_boot,
        random_state=random_state,
    )
    absolute_correlation_summary = build_correlation_summary(
        session_budget_summary=session_budget_summary,
        n_boot=n_boot,
        random_state=random_state,
        absolute_gaps=True,
    )
    continuous_relationships = build_continuous_relationships(
        session_budget_summary=session_budget_summary,
    )
    distribution_summary = build_retrieval_distribution_summary(
        records=records,
        selected_temporal_weight=selected_temporal_weight,
    )
    budget_colors = get_sampling_budget_colors(
        sampling_budgets=budget_summary['sampling_budget'].tolist(),
        configured_colors=analysis_config.get('sampling_budget_colors', {}),
    )

    choice_repetitions.to_csv(output_dir / 'choice_repetitions.csv', index=False)
    target_retrievals.to_csv(output_dir / 'target_retrievals.csv', index=False)
    session_budget_summary.to_csv(output_dir / 'session_budget_summary.csv', index=False)
    budget_summary.to_csv(output_dir / 'budget_summary.csv', index=False)
    correlation_summary.to_csv(output_dir / 'correlation_summary.csv', index=False)
    continuous_relationships.to_csv(output_dir / 'continuous_relationships.csv', index=False)
    distribution_summary.to_csv(output_dir / 'retrieval_distribution_summary.csv', index=False)
    plot_true_reward_gap_distribution(
        session_budget_summary=session_budget_summary,
        savepath=visual_dir / 'true_reward_gap_distribution.pdf',
    )
    plot_gap_reconstruction(
        session_budget_summary=session_budget_summary,
        correlation_summary=correlation_summary,
        absolute_correlation_summary=absolute_correlation_summary,
        budget_colors=budget_colors,
        savepath=visual_dir / 'gap_reconstruction.pdf',
    )
    plot_choice_accuracy_and_error(
        budget_summary=budget_summary,
        correlation_summary=correlation_summary,
        budget_colors=budget_colors,
        savepath=visual_dir / 'choice_accuracy_and_error.pdf',
    )
    plot_margin_confidence_variability(
        relationships=continuous_relationships,
        budget_summary=budget_summary,
        budget_colors=budget_colors,
        savepath=visual_dir / 'margin_confidence_variability.pdf',
    )
    plot_semantic_retrieval_quality(
        budget_summary=budget_summary,
        distribution_summary=distribution_summary,
        savepath=visual_dir / 'semantic_retrieval_quality.pdf',
    )


def main(argv: list[str] | None = None) -> None:
    """
    Run the Nicholas-Mattar analysis.
    """
    run_analysis(parse_args(argv))


if __name__ == '__main__':
    main()
