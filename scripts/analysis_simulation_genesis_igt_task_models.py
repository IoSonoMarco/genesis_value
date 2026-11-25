"""
Analyze the unified six-model GENESIS Iowa Gambling Task experiment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path.cwd()))

from models import load_linear_decoder
from scripts.analysis_simulation_genesis_igt_task_temporal_valence import (
    add_igt_blocks,
    bootstrap_mean_ci,
    bootstrap_mean_se,
    build_agent_tables,
    build_block_behavior_summary,
    build_trial_probability_summary,
    centered_rolling_mean,
    plot_figure_1,
    plot_figure_2,
    plot_figure_3,
    resolve_contrast_colors,
    resolve_deck_colors,
    resolve_temporal_weight_colors,
    save_figure,
    style_axis,
)
from scripts.simulation_genesis_igt_task_models import IGT_REWARD_MODEL_PATH, VALENCE_EMBEDDINGS, load_igt_schedule
from scripts.simulation_pvl_delta_igt_task import MODEL_NAME as PVL_MODEL_NAME
from utils import filepaths
from utils.simulation_config import load_simulation_config


DEFAULT_CONFIG_PATH = filepaths.ROOT / 'configs' / 'simulation_genesis_igt_task_models.yaml'
DECK_NAMES = ('A', 'B', 'C', 'D')
PROBABILITY_COLUMNS = {'A': 'p_a', 'B': 'p_b', 'C': 'p_c', 'D': 'p_d'}
CHOICE_COLUMNS = {'A': 'i_a', 'B': 'i_b', 'C': 'i_c', 'D': 'i_d'}
VALENCE_NAMES = ('positive', 'negative', 'zero')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse unified IGT analysis arguments.

    Args:
        argv (list[str] | None): Optional CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Analyze all six GENESIS IGT model conditions.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-save-figures', action='store_true')
    return parser.parse_args(argv)


def reward_valence(reward: float) -> str:
    """
    Map one scaled payoff to its categorical valence.

    Args:
        reward (float): Scaled IGT payoff.

    Returns:
        str: Positive, negative, or zero label.
    """
    if reward > 0.0:
        return 'positive'
    if reward < 0.0:
        return 'negative'
    return 'zero'


def load_model_cells(*, input_dir: Path, model_config: dict[str, str]) -> list[dict[str, Any]]:
    """
    Load and validate all temporal-weight cells for one model.

    Args:
        input_dir (Path): Model-specific raw-record directory.
        model_config (dict[str, str]): Expected model definition.

    Returns:
        list[dict[str, Any]]: Cells ordered by temporal weight.
    """
    paths = list(input_dir.glob('*_temporal_weight-*.npy'))
    if not paths:
        raise ValueError(f'No unified IGT simulation files found in {input_dir}.')
    cells = [np.load(path, allow_pickle=True).item() for path in paths]
    for cell in cells:
        if cell['model'] != model_config:
            raise ValueError(f'Simulation model does not match {model_config["name"]}.')
        if cell['config']['episodic_sampling'] != {
            'sampling_budget': 20,
            'sample_replacement': True,
            'query_key_normalization': 'zscore',
        }:
            raise ValueError('Unified IGT retrieval settings do not match the experiment contract.')
        if cell['retrieval_temperature'] != 0.25:
            raise ValueError('Unified IGT retrieval temperature must be 0.25.')
    return sorted(cells, key=lambda cell: cell['temporal_weight'])


def build_tables(cells: Sequence[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flatten trial histories and deck-specific realized retrievals.

    Args:
        cells (Sequence[dict[str, Any]]): Validated model simulation cells.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Trial and deck-retrieval tables.
    """
    trial_rows = []
    retrieval_rows = []
    for cell in cells:
        model = cell['model']
        for agent_run in cell['agent_runs']:
            history = agent_run['simulation_log']
            for trial in history:
                all_sampled_indices = []
                for deck in DECK_NAMES:
                    sampled_indices = trial['retrieved_episode_indices'][deck]
                    sampled_valences = [reward_valence(history[index]['reward']) for index in sampled_indices]
                    counts = {
                        valence: sampled_valences.count(valence)
                        for valence in VALENCE_NAMES
                    }
                    sample_count = len(sampled_indices)
                    retrieval_rows.append({
                        'model_name': model['name'],
                        'model_family': model['family'],
                        'attention_condition': model['attention'],
                        'temporal_weight': float(trial['temporal_weight']),
                        'agent_id': agent_run['agent_id'],
                        'trial_id': trial['trial_id'],
                        'trial_number': trial['trial_id'] + 1,
                        'deck': deck,
                        'available_memory_count': trial['retrieval_metadata'][deck]['available_memory_count'],
                        'sampled_episode_count': sample_count,
                        'sampled_positive_count': counts['positive'],
                        'sampled_negative_count': counts['negative'],
                        'sampled_zero_count': counts['zero'],
                        'sampled_positive_proportion': counts['positive'] / sample_count if sample_count else np.nan,
                        'sampled_negative_proportion': counts['negative'] / sample_count if sample_count else np.nan,
                        'sampled_zero_proportion': counts['zero'] / sample_count if sample_count else np.nan,
                    })
                    all_sampled_indices.extend(sampled_indices)
                trial_row = {
                    'query_valence': model['name'],
                    'model_name': model['name'],
                    'model_family': model['family'],
                    'attention_condition': model['attention'],
                    'temporal_weight': float(trial['temporal_weight']),
                    'agent_id': agent_run['agent_id'],
                    'trial_id': trial['trial_id'],
                    'trial_number': trial['trial_id'] + 1,
                    'chosen_deck': trial['chosen_deck'],
                    'reward': trial['reward'],
                    'cumulative_reward': trial['cumulative_reward'],
                    'choice_temperature': trial['choice_temperature'],
                    'mean_sampled_episodic_lag': (
                        float(np.mean([trial['trial_id'] - index for index in all_sampled_indices]))
                        if all_sampled_indices else np.nan
                    ),
                    'block_id': trial['trial_id'] // 20 + 1,
                }
                trial_row.update({f'q_{deck.lower()}': trial['action_values'][deck] for deck in DECK_NAMES})
                trial_row.update({column: trial[column] for column in PROBABILITY_COLUMNS.values()})
                trial_row.update({
                    CHOICE_COLUMNS[deck]: int(trial['chosen_deck'] == deck)
                    for deck in DECK_NAMES
                })
                trial_rows.append(trial_row)
    trial_table = pd.DataFrame(trial_rows).sort_values(
        ['temporal_weight', 'agent_id', 'trial_id'],
    ).reset_index(drop=True)
    retrieval_table = pd.DataFrame(retrieval_rows).sort_values(
        ['temporal_weight', 'agent_id', 'trial_id', 'deck'],
    ).reset_index(drop=True)
    return trial_table, retrieval_table


def build_retrieval_summaries(
    *,
    retrieval_table: pd.DataFrame,
    n_boot: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate retrieval composition at agent and model-cell levels.

    Args:
        retrieval_table (pd.DataFrame): One row per deck-specific retrieval.
        n_boot (int): Number of agent-level bootstrap resamples.
        random_seed (int): Base bootstrap seed.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Agent and temporal-weight/deck summaries.
    """
    group_columns = [
        'model_name',
        'model_family',
        'attention_condition',
        'temporal_weight',
        'agent_id',
        'deck',
    ]
    count_columns = [
        'sampled_episode_count',
        'sampled_positive_count',
        'sampled_negative_count',
        'sampled_zero_count',
    ]
    agent_deck_table = retrieval_table.groupby(group_columns, sort=True)[count_columns].sum().reset_index()
    for valence in VALENCE_NAMES:
        agent_deck_table[f'sampled_{valence}_proportion'] = (
            agent_deck_table[f'sampled_{valence}_count'] / agent_deck_table['sampled_episode_count']
        )

    summary_rows = []
    summary_groups = [
        'model_name',
        'model_family',
        'attention_condition',
        'temporal_weight',
        'deck',
    ]
    for group_index, (group_key, group) in enumerate(agent_deck_table.groupby(summary_groups, sort=True)):
        row = dict(zip(summary_groups, group_key))
        row['sampled_episode_count'] = int(group['sampled_episode_count'].sum())
        for valence_index, valence in enumerate(VALENCE_NAMES):
            count_column = f'sampled_{valence}_count'
            proportion_column = f'sampled_{valence}_proportion'
            row[count_column] = int(group[count_column].sum())
            row[f'pooled_{proportion_column}'] = row[count_column] / row['sampled_episode_count']
            seed = random_seed + group_index * len(VALENCE_NAMES) + valence_index
            mean, ci_low, ci_high = bootstrap_mean_ci(
                values=group[proportion_column],
                n_boot=n_boot,
                random_state=seed,
            )
            row[f'{proportion_column}_mean'] = mean
            row[f'{proportion_column}_ci_low'] = ci_low
            row[f'{proportion_column}_ci_high'] = ci_high
            row[f'{proportion_column}_bootstrap_se'] = bootstrap_mean_se(
                values=group[proportion_column],
                n_boot=n_boot,
                random_state=seed,
            )
        summary_rows.append(row)
    return agent_deck_table, pd.DataFrame(summary_rows)


def merge_agent_composition(*, agent_table: pd.DataFrame, agent_deck_table: pd.DataFrame) -> pd.DataFrame:
    """
    Add whole-task sampled-valence proportions to the behavior table.

    Args:
        agent_table (pd.DataFrame): Whole-task behavioral summaries.
        agent_deck_table (pd.DataFrame): Agent/deck retrieval counts.

    Returns:
        pd.DataFrame: Agent behavior and retrieval-composition summaries.
    """
    count_columns = [
        'sampled_episode_count',
        'sampled_positive_count',
        'sampled_negative_count',
        'sampled_zero_count',
    ]
    composition = agent_deck_table.groupby(
        ['model_name', 'model_family', 'attention_condition', 'temporal_weight', 'agent_id'],
        sort=True,
    )[count_columns].sum().reset_index()
    for valence in VALENCE_NAMES:
        composition[f'sampled_{valence}_proportion'] = (
            composition[f'sampled_{valence}_count'] / composition['sampled_episode_count']
        )
    result = agent_table.rename(columns={'query_valence': 'model_name'}).merge(
        composition,
        on=['model_name', 'temporal_weight', 'agent_id'],
    )
    return result


@torch.no_grad()
def build_query_diagnostics(
    *,
    model_config: dict[str, str],
    schedule: np.ndarray,
    reward_model: torch.nn.Module,
    device: torch.device,
) -> pd.DataFrame:
    """
    Inspect categorical or learned-reward cosine similarity ordering.

    Args:
        model_config (dict[str, str]): Model family and attention condition.
        schedule (np.ndarray): Scaled payoff schedule with shape ``[N, 4]``.
        reward_model (torch.nn.Module): Frozen learned reward encoder/decoder.
        device (torch.device): Inference device.

    Returns:
        pd.DataFrame: Candidate/query cosine similarities and ordering diagnostic.
    """
    attention = model_config['attention']
    query_names = []
    if attention in {'loss', 'balanced'}:
        query_names.append('negative')
    if attention in {'win', 'balanced'}:
        query_names.append('positive')

    rows = []
    if model_config['family'] == 'valence':
        candidates = [
            ('negative', VALENCE_EMBEDDINGS['negative']),
            ('zero', VALENCE_EMBEDDINGS['zero']),
            ('positive', VALENCE_EMBEDDINGS['positive']),
        ]
        for query_name in query_names:
            query_embedding = VALENCE_EMBEDDINGS[query_name]
            for candidate_index, (candidate_label, candidate_embedding) in enumerate(candidates):
                similarity = torch.nn.functional.cosine_similarity(
                    query_embedding,
                    candidate_embedding,
                    dim=0,
                )
                rows.append({
                    'model_name': model_config['name'],
                    'model_family': model_config['family'],
                    'attention_condition': attention,
                    'query_component': query_name,
                    'query_reward': np.nan,
                    'candidate_label': candidate_label,
                    'candidate_reward': np.nan,
                    'candidate_order': candidate_index,
                    'cosine_similarity': float(similarity),
                    'monotonic_order': True,
                })
        return pd.DataFrame(rows)

    rewards = np.sort(np.unique(schedule))
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    embeddings = reward_model.encode(reward_tensor).cpu()
    query_rewards = {'negative': float(schedule.min()), 'positive': float(schedule.max())}
    for query_name in query_names:
        query_reward = query_rewards[query_name]
        query_index = int(np.flatnonzero(np.isclose(rewards, query_reward))[0])
        query_embedding = embeddings[query_index]
        similarities = torch.nn.functional.cosine_similarity(
            embeddings,
            query_embedding.expand_as(embeddings),
            dim=1,
        ).numpy()
        differences = np.diff(similarities)
        monotonic_order = bool(np.all(differences <= 0.0) if query_name == 'negative' else np.all(differences >= 0.0))
        for candidate_index, (reward, similarity) in enumerate(zip(rewards, similarities)):
            rows.append({
                'model_name': model_config['name'],
                'model_family': model_config['family'],
                'attention_condition': attention,
                'query_component': query_name,
                'query_reward': query_reward,
                'candidate_label': reward_valence(float(reward)),
                'candidate_reward': float(reward),
                'candidate_order': candidate_index,
                'cosine_similarity': float(similarity),
                'monotonic_order': monotonic_order,
            })
    return pd.DataFrame(rows)


def plot_retrieval_composition(*, summary: pd.DataFrame, visual_dir: Path) -> None:
    """
    Plot sampled positive, negative, and zero proportions by deck and weight.

    Args:
        summary (pd.DataFrame): Temporal-weight/deck retrieval summary.
        visual_dir (Path): Model-specific figure directory.
    """
    weights = sorted(summary['temporal_weight'].unique())
    colors = {'positive': '#2166AC', 'negative': '#B2182B', 'zero': '#999999'}
    figure, axes = plt.subplots(1, len(DECK_NAMES), figsize=(13, 3.4), sharey=True)
    for axis, deck in zip(axes, DECK_NAMES):
        deck_table = summary[summary['deck'] == deck].sort_values('temporal_weight')
        bottom = np.zeros(len(deck_table))
        for valence in VALENCE_NAMES:
            values = deck_table[f'pooled_sampled_{valence}_proportion'].to_numpy()
            axis.bar(
                np.arange(len(weights)),
                values,
                bottom=bottom,
                color=colors[valence],
                label=valence.capitalize(),
            )
            bottom += values
        axis.set_title(f'Deck {deck}')
        axis.set_xticks(np.arange(len(weights)), [f'{weight:g}' for weight in weights])
        axis.set_xlabel('Temporal weight')
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Sampled episode proportion')
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1.0))
    figure.tight_layout()
    save_figure(figure=figure, savepath=visual_dir / 'figure_4_igt_retrieval_composition')


def plot_query_diagnostics(*, diagnostics: pd.DataFrame, visual_dir: Path) -> None:
    """
    Plot cosine similarity to each active attention query.

    Args:
        diagnostics (pd.DataFrame): Query similarity diagnostic table.
        visual_dir (Path): Model-specific figure directory.
    """
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    colors = {'negative': '#B2182B', 'positive': '#2166AC'}
    for query_name, group in diagnostics.groupby('query_component', sort=True):
        x_values = (
            group['candidate_reward'].to_numpy()
            if group['candidate_reward'].notna().all()
            else group['candidate_order'].to_numpy()
        )
        axis.plot(
            x_values,
            group['cosine_similarity'],
            color=colors[query_name],
            marker='o',
            label=f'{query_name.capitalize()} query',
        )
    if diagnostics['candidate_reward'].isna().any():
        labels = diagnostics.drop_duplicates('candidate_order').sort_values('candidate_order')['candidate_label']
        axis.set_xticks(np.arange(len(labels)), labels)
        axis.set_xlabel('Stored categorical valence')
    else:
        axis.set_xlabel('Stored scaled reward')
    axis.set_ylabel('Cosine similarity')
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    save_figure(figure=figure, savepath=visual_dir / 'figure_5_igt_query_similarity_diagnostics')


def summarize_cross_model_agents(*, agent_table: pd.DataFrame, n_boot: int, random_seed: int) -> pd.DataFrame:
    """
    Build comparable behavior and retrieval summaries for all six models.

    Args:
        agent_table (pd.DataFrame): Concatenated whole-task agent records.
        n_boot (int): Number of bootstrap resamples.
        random_seed (int): Base bootstrap seed.

    Returns:
        pd.DataFrame: Model/weight means, intervals, and bootstrap errors.
    """
    metrics = [
        'total_net_score',
        'total_frequency_score',
        'final_cumulative_reward',
        'mean_sampled_episodic_lag',
        'sampled_positive_proportion',
        'sampled_negative_proportion',
        'sampled_zero_proportion',
    ]
    group_columns = ['model_name', 'model_family', 'attention_condition', 'temporal_weight']
    rows = []
    for group_index, (group_key, group) in enumerate(agent_table.groupby(group_columns, sort=True)):
        row = dict(zip(group_columns, group_key))
        for metric_index, metric in enumerate(metrics):
            seed = random_seed + group_index * len(metrics) + metric_index
            mean, ci_low, ci_high = bootstrap_mean_ci(
                values=group[metric],
                n_boot=n_boot,
                random_state=seed,
            )
            row[f'{metric}_mean'] = mean
            row[f'{metric}_ci_low'] = ci_low
            row[f'{metric}_ci_high'] = ci_high
            row[f'{metric}_bootstrap_se'] = bootstrap_mean_se(
                values=group[metric],
                n_boot=n_boot,
                random_state=seed,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_cross_model_behavior(
    *,
    summary: pd.DataFrame,
    model_colors: dict[str, str],
    output_dir: Path,
) -> None:
    """
    Plot direct whole-task behavior comparisons across models.

    Args:
        summary (pd.DataFrame): Cross-model agent summary.
        model_colors (dict[str, str]): Model display colors.
        output_dir (Path): Common analysis root.
    """
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    temporal_weights = sorted(summary['temporal_weight'].dropna().unique())
    for model_name, group in summary.groupby('model_name', sort=False):
        group = group.sort_values('temporal_weight')
        for axis, metric, ylabel in (
            (axes[0], 'total_net_score', 'Total Net Score'),
            (axes[1], 'total_frequency_score', 'Total Frequency Score'),
        ):
            if group['temporal_weight'].isna().all():
                mean = float(group[f'{metric}_mean'].iloc[0])
                standard_error = float(group[f'{metric}_bootstrap_se'].iloc[0])
                axis.axhline(
                    mean,
                    color=model_colors[model_name],
                    linestyle='--',
                    label='PVL-Delta (single fixed reference)',
                )
                axis.fill_between(
                    temporal_weights,
                    mean - standard_error,
                    mean + standard_error,
                    color=model_colors[model_name],
                    alpha=0.1,
                )
            else:
                axis.errorbar(
                    group['temporal_weight'],
                    group[f'{metric}_mean'],
                    yerr=group[f'{metric}_bootstrap_se'],
                    color=model_colors[model_name],
                    marker='o',
                    label=model_name,
                )
            axis.axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.5)
            axis.set_xlabel('Temporal weight')
            axis.set_ylabel(ylabel)
            axis.spines[['top', 'right']].set_visible(False)
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1.0), fontsize=8)
    figure.tight_layout()
    save_figure(figure=figure, savepath=output_dir / 'figure_1_igt_model_behavior_comparison')


def plot_cross_model_composition(
    *,
    summary: pd.DataFrame,
    model_colors: dict[str, str],
    output_dir: Path,
) -> None:
    """
    Plot positive and negative sampled proportions across all models.

    Args:
        summary (pd.DataFrame): Cross-model agent summary.
        model_colors (dict[str, str]): Model display colors.
        output_dir (Path): Common analysis root.
    """
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for model_name, group in summary.groupby('model_name', sort=False):
        group = group.sort_values('temporal_weight')
        for axis, valence in zip(axes, ('positive', 'negative')):
            metric = f'sampled_{valence}_proportion'
            axis.errorbar(
                group['temporal_weight'],
                group[f'{metric}_mean'],
                yerr=group[f'{metric}_bootstrap_se'],
                color=model_colors[model_name],
                marker='o',
                label=model_name,
            )
            axis.set_xlabel('Temporal weight')
            axis.set_ylabel(f'Sampled {valence} proportion')
            axis.set_ylim(0, 1)
            axis.spines[['top', 'right']].set_visible(False)
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1.0), fontsize=8)
    figure.tight_layout()
    save_figure(figure=figure, savepath=output_dir / 'figure_2_igt_model_retrieval_comparison')


def load_pvl_cell(input_dir: Path) -> dict[str, Any]:
    """
    Load the single fixed-parameter PVL-Delta simulation cell.

    Args:
        input_dir (Path): PVL raw-record directory.

    Returns:
        dict[str, Any]: Fixed-parameter PVL simulation payload.
    """
    paths = list(input_dir.glob('*_fixed_parameters.npy'))
    if len(paths) != 1:
        raise ValueError(f'Expected one PVL-Delta simulation file in {input_dir}.')
    cell = np.load(paths[0], allow_pickle=True).item()
    if cell['model']['name'] != PVL_MODEL_NAME:
        raise ValueError('PVL-Delta payload has an unexpected model name.')
    return cell


def build_pvl_trial_table(cell: dict[str, Any]) -> pd.DataFrame:
    """
    Flatten PVL histories into the common behavioral trial schema.

    Args:
        cell (dict[str, Any]): Fixed-parameter PVL simulation payload.

    Returns:
        pd.DataFrame: One behavioral row per agent and trial.
    """
    rows = []
    model = cell['model']
    for agent_run in cell['agent_runs']:
        for trial in agent_run['simulation_log']:
            row = {
                'model_name': model['name'],
                'model_family': model['family'],
                'attention_condition': model['attention'],
                'agent_id': agent_run['agent_id'],
                'trial_id': trial['trial_id'],
                'trial_number': trial['trial_id'] + 1,
                'chosen_deck': trial['chosen_deck'],
                'reward': trial['reward'],
                'cumulative_reward': trial['cumulative_reward'],
                'choice_temperature': trial['choice_temperature'],
                'response_sensitivity': trial['response_sensitivity'],
                'subjective_utility': trial['subjective_utility'],
                'prediction_error': trial['prediction_error'],
                'block_id': trial['trial_id'] // 20 + 1,
            }
            row.update({f'q_{deck.lower()}': trial['action_values'][deck] for deck in DECK_NAMES})
            row.update({column: trial[column] for column in PROBABILITY_COLUMNS.values()})
            row.update({
                CHOICE_COLUMNS[deck]: int(trial['chosen_deck'] == deck)
                for deck in DECK_NAMES
            })
            rows.append(row)
    return pd.DataFrame(rows).sort_values(['agent_id', 'trial_id']).reset_index(drop=True)


def build_pvl_behavior_tables(
    *,
    trial_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build PVL block and whole-task behavioral summaries.

    Args:
        trial_table (pd.DataFrame): PVL behavioral trial records.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Agent-block and whole-task agent tables.
    """
    model_columns = ['model_name', 'model_family', 'attention_condition']
    count_columns = list(CHOICE_COLUMNS.values())
    block_table = trial_table.groupby(
        model_columns + ['agent_id', 'block_id'],
        sort=True,
    )[count_columns].sum().reset_index()
    block_table['net_score'] = block_table['i_c'] + block_table['i_d'] - block_table['i_a'] - block_table['i_b']
    block_table['frequency_score'] = block_table['i_b'] + block_table['i_d'] - block_table['i_a'] - block_table['i_c']
    aggregations = {column: 'sum' for column in count_columns}
    aggregations.update({'reward': 'sum', 'cumulative_reward': 'last'})
    agent_table = trial_table.groupby(
        model_columns + ['agent_id'],
        sort=True,
    ).agg(aggregations).reset_index()
    agent_table = agent_table.rename(columns={
        'i_a': 'total_a_choices',
        'i_b': 'total_b_choices',
        'i_c': 'total_c_choices',
        'i_d': 'total_d_choices',
        'reward': 'total_reward',
        'cumulative_reward': 'final_cumulative_reward',
    })
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


def build_pvl_trial_summary(
    *,
    trial_table: pd.DataFrame,
    n_boot: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Summarize PVL deck-choice probabilities across agents by trial.

    Args:
        trial_table (pd.DataFrame): PVL behavioral trial records.
        n_boot (int): Number of agent-level bootstrap resamples.
        random_seed (int): Base bootstrap seed.

    Returns:
        pd.DataFrame: Trial/deck probability summary.
    """
    rows = []
    group_index = 0
    for trial_number, group in trial_table.groupby('trial_number', sort=True):
        for deck in DECK_NAMES:
            values = group[PROBABILITY_COLUMNS[deck]]
            mean, ci_low, ci_high = bootstrap_mean_ci(
                values=values,
                n_boot=n_boot,
                random_state=random_seed + group_index,
            )
            rows.append({
                'model_name': PVL_MODEL_NAME,
                'trial_number': trial_number,
                'deck': deck,
                'mean_choice_probability': mean,
                'choice_probability_ci_low': ci_low,
                'choice_probability_ci_high': ci_high,
                'choice_probability_bootstrap_se': bootstrap_mean_se(
                    values=values,
                    n_boot=n_boot,
                    random_state=random_seed + group_index,
                ),
                'observed_choice_proportion': group[CHOICE_COLUMNS[deck]].mean(),
            })
            group_index += 1
    return pd.DataFrame(rows)


def build_pvl_block_summary(
    *,
    block_table: pd.DataFrame,
    n_boot: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Summarize PVL block-level behavior across agents.

    Args:
        block_table (pd.DataFrame): PVL agent-block behavioral records.
        n_boot (int): Number of agent-level bootstrap resamples.
        random_seed (int): Base bootstrap seed.

    Returns:
        pd.DataFrame: Bootstrap behavior summary by block.
    """
    metrics = ['net_score', 'frequency_score', 'i_a', 'i_b', 'i_c', 'i_d']
    rows = []
    for group_index, (block_id, group) in enumerate(block_table.groupby('block_id', sort=True)):
        row = {'model_name': PVL_MODEL_NAME, 'block_id': block_id}
        for metric_index, metric in enumerate(metrics):
            seed = random_seed + group_index * len(metrics) + metric_index
            mean, ci_low, ci_high = bootstrap_mean_ci(
                values=group[metric],
                n_boot=n_boot,
                random_state=seed,
            )
            row[f'{metric}_mean'] = mean
            row[f'{metric}_ci_low'] = ci_low
            row[f'{metric}_ci_high'] = ci_high
            row[f'{metric}_bootstrap_se'] = bootstrap_mean_se(
                values=group[metric],
                n_boot=n_boot,
                random_state=seed,
            )
        rows.append(row)
    return pd.DataFrame(rows).rename(columns={
        'i_a_mean': 'a_choices_mean',
        'i_b_mean': 'b_choices_mean',
        'i_c_mean': 'c_choices_mean',
        'i_d_mean': 'd_choices_mean',
    })


def plot_pvl_block_metric(*, axis: plt.Axes, summary: pd.DataFrame, metric: str, ylabel: str) -> None:
    """
    Plot one PVL block-level behavioral metric.

    Args:
        axis (plt.Axes): Destination axis.
        summary (pd.DataFrame): PVL block summary.
        metric (str): Summary metric base name.
        ylabel (str): Vertical-axis label.
    """
    estimate = summary[f'{metric}_mean'].to_numpy()
    standard_error = summary[f'{metric}_bootstrap_se'].to_numpy()
    axis.plot(summary['block_id'], estimate, color='black', marker='o')
    axis.fill_between(
        summary['block_id'],
        estimate - standard_error,
        estimate + standard_error,
        color='black',
        alpha=0.16,
    )
    axis.axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.5)
    axis.set_xlabel('IGT block')
    axis.set_ylabel(ylabel)
    style_axis(axis)


def plot_pvl_figure_1(*, trial_summary: pd.DataFrame, deck_colors: dict[str, str], visual_dir: Path) -> None:
    """
    Plot PVL deck probabilities in the aligned Figure 1 layout.

    Args:
        trial_summary (pd.DataFrame): PVL trial/deck probability summary.
        deck_colors (dict[str, str]): Configured deck colors.
        visual_dir (Path): PVL figure directory.
    """
    figure, axes = plt.subplots(2, 4, figsize=(12.4, 6.2), squeeze=False)
    for index, deck in enumerate(DECK_NAMES):
        data = trial_summary[trial_summary['deck'] == deck]
        mean = centered_rolling_mean(values=data['mean_choice_probability'], window=5)
        standard_error = centered_rolling_mean(values=data['choice_probability_bootstrap_se'], window=5)
        axis = axes[0, index]
        axis.plot(data['trial_number'], mean, color=deck_colors[deck])
        axis.fill_between(
            data['trial_number'],
            mean - standard_error,
            mean + standard_error,
            color=deck_colors[deck],
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
    comparison_axis = axes[1, 0]
    for deck in DECK_NAMES:
        data = trial_summary[trial_summary['deck'] == deck]
        mean = centered_rolling_mean(values=data['mean_choice_probability'], window=5)
        standard_error = centered_rolling_mean(values=data['choice_probability_bootstrap_se'], window=5)
        comparison_axis.plot(
            data['trial_number'],
            mean,
            color=deck_colors[deck],
            label=deck,
        )
        comparison_axis.fill_between(
            data['trial_number'],
            mean - standard_error,
            mean + standard_error,
            color=deck_colors[deck],
            alpha=0.16,
        )
    comparison_axis.axhline(0.25, color='black', linewidth=1, linestyle=':', alpha=0.5)
    comparison_axis.axvline(10.5, color='black', linewidth=1, linestyle='--', alpha=0.35)
    comparison_axis.set_title('E. Deck comparison (fixed PVL-Delta)')
    comparison_axis.set_xlim(1, 100)
    comparison_axis.set_ylim(0, 1)
    comparison_axis.set_xlabel('Trial')
    comparison_axis.set_ylabel('Mean choice probability')
    comparison_axis.legend(frameon=False)
    style_axis(comparison_axis)
    for axis in axes[1, 1:]:
        axis.set_visible(False)
    figure.tight_layout()
    save_figure(figure=figure, savepath=visual_dir / 'figure_1_igt_deck_choice_probability_trajectories')


def plot_pvl_figure_2(
    *,
    block_summary: pd.DataFrame,
    block_summary_10: pd.DataFrame,
    agent_table: pd.DataFrame,
    visual_dir: Path,
) -> None:
    """
    Plot PVL Net Score learning and agent-level performance.

    Args:
        block_summary (pd.DataFrame): Twenty-trial PVL summary.
        block_summary_10 (pd.DataFrame): Ten-trial PVL summary.
        agent_table (pd.DataFrame): Whole-task PVL agent records.
        visual_dir (Path): PVL figure directory.
    """
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    plot_pvl_block_metric(axis=axes[0, 0], summary=block_summary, metric='net_score', ylabel='Net Score')
    axes[0, 0].set_title('A. Advantageous-choice learning')
    values = agent_table['total_net_score']
    axes[0, 1].scatter(np.zeros(len(values)), values, color='black', alpha=0.3, s=14)
    axes[0, 1].scatter([0], [values.mean()], color='black', s=50)
    axes[0, 1].axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.5)
    axes[0, 1].set_xticks([0], ['PVL-Delta'])
    axes[0, 1].set_ylabel('Total Net Score')
    axes[0, 1].set_title('B. Overall advantageous choice')
    style_axis(axes[0, 1])
    plot_pvl_block_metric(
        axis=axes[1, 0],
        summary=block_summary_10,
        metric='net_score',
        ylabel='Net Score',
    )
    axes[1, 0].set_title('C. Advantageous-choice learning (10-trial blocks)')
    axes[1, 1].set_visible(False)
    figure.tight_layout()
    save_figure(figure=figure, savepath=visual_dir / 'figure_2_igt_net_score_learning')


def plot_pvl_figure_3(
    *,
    block_summary: pd.DataFrame,
    block_summary_10: pd.DataFrame,
    agent_table: pd.DataFrame,
    visual_dir: Path,
) -> None:
    """
    Plot PVL frequency behavior without episodic-retrieval panels.

    Args:
        block_summary (pd.DataFrame): Twenty-trial PVL summary.
        block_summary_10 (pd.DataFrame): Ten-trial PVL summary.
        agent_table (pd.DataFrame): Whole-task PVL agent records.
        visual_dir (Path): PVL figure directory.
    """
    figure, axes = plt.subplots(2, 3, figsize=(14, 7.2))
    plot_pvl_block_metric(
        axis=axes[0, 0],
        summary=block_summary,
        metric='frequency_score',
        ylabel='Frequency Score',
    )
    axes[0, 0].set_title('A. Rare vs frequent punishment')
    contrasts = [agent_table['b_minus_a'].mean(), agent_table['d_minus_c'].mean()]
    axes[0, 1].bar(['B - A', 'D - C'], contrasts, color=['#7F3C8D', '#11A579'])
    axes[0, 1].axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.5)
    axes[0, 1].set_ylabel('Mean choice-count contrast')
    axes[0, 1].set_title('B. Within-value frequency contrasts')
    style_axis(axes[0, 1])
    plot_pvl_block_metric(
        axis=axes[1, 0],
        summary=block_summary_10,
        metric='frequency_score',
        ylabel='Frequency Score',
    )
    axes[1, 0].set_title('C. Rare vs frequent punishment (10-trial blocks)')
    axes[0, 2].set_visible(False)
    axes[1, 1].set_visible(False)
    axes[1, 2].set_visible(False)
    figure.tight_layout()
    save_figure(figure=figure, savepath=visual_dir / 'figure_3_igt_frequency_behavior')


def analyze_pvl_model(
    *,
    input_dir: Path,
    output_dir: Path,
    analysis_config: dict[str, Any],
    deck_colors: dict[str, str],
    save_figures: bool,
) -> pd.DataFrame:
    """
    Export behavior-only summaries for the fixed PVL reference.

    Args:
        input_dir (Path): PVL raw-record directory.
        output_dir (Path): PVL analysis directory.
        analysis_config (dict[str, Any]): Shared bootstrap and display settings.
        deck_colors (dict[str, str]): Configured deck colors.
        save_figures (bool): Whether to write PDF and SVG figures.

    Returns:
        pd.DataFrame: Whole-task PVL agent behavior table.
    """
    cell = load_pvl_cell(input_dir)
    trial_table = build_pvl_trial_table(cell)
    block_table, agent_table = build_pvl_behavior_tables(trial_table=trial_table)
    trial_summary = build_pvl_trial_summary(
        trial_table=trial_table,
        n_boot=analysis_config['n_boot'],
        random_seed=analysis_config['random_seed'],
    )
    block_summary = build_pvl_block_summary(
        block_table=block_table,
        n_boot=analysis_config['n_boot'],
        random_seed=analysis_config['random_seed'],
    )
    trial_table_10 = trial_table.copy()
    trial_table_10['block_id'] = trial_table_10['trial_id'] // 10 + 1
    block_table_10, _ = build_pvl_behavior_tables(trial_table=trial_table_10)
    block_summary_10 = build_pvl_block_summary(
        block_table=block_table_10,
        n_boot=analysis_config['n_boot'],
        random_seed=analysis_config['random_seed'],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_summary.to_csv(output_dir / 'igt_trial_probability_summary.csv', index=False)
    block_summary.to_csv(output_dir / 'igt_block_behavior_summary.csv', index=False)
    agent_table.to_csv(output_dir / 'igt_agent_behavior_summary.csv', index=False)
    if not save_figures:
        return agent_table
    visual_dir = output_dir / 'visuals'
    visual_dir.mkdir(parents=True, exist_ok=True)
    plot_pvl_figure_1(trial_summary=trial_summary, deck_colors=deck_colors, visual_dir=visual_dir)
    plot_pvl_figure_2(
        block_summary=block_summary,
        block_summary_10=block_summary_10,
        agent_table=agent_table,
        visual_dir=visual_dir,
    )
    plot_pvl_figure_3(
        block_summary=block_summary,
        block_summary_10=block_summary_10,
        agent_table=agent_table,
        visual_dir=visual_dir,
    )
    return agent_table


def summarize_cross_model_behavior(
    *,
    agent_table: pd.DataFrame,
    n_boot: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Summarize common behavior across GENESIS and optional PVL agents.

    Args:
        agent_table (pd.DataFrame): Common whole-task agent records.
        n_boot (int): Number of bootstrap resamples.
        random_seed (int): Base bootstrap seed.

    Returns:
        pd.DataFrame: Model-level common behavioral comparison.
    """
    metrics = ['total_net_score', 'total_frequency_score', 'final_cumulative_reward']
    group_columns = ['model_name', 'model_family', 'attention_condition', 'temporal_weight']
    rows = []
    for group_index, (group_key, group) in enumerate(
        agent_table.groupby(group_columns, sort=True, dropna=False),
    ):
        row = dict(zip(group_columns, group_key))
        for metric_index, metric in enumerate(metrics):
            seed = random_seed + group_index * len(metrics) + metric_index
            mean, ci_low, ci_high = bootstrap_mean_ci(
                values=group[metric],
                n_boot=n_boot,
                random_state=seed,
            )
            row[f'{metric}_mean'] = mean
            row[f'{metric}_ci_low'] = ci_low
            row[f'{metric}_ci_high'] = ci_high
            row[f'{metric}_bootstrap_se'] = bootstrap_mean_se(
                values=group[metric],
                n_boot=n_boot,
                random_state=seed,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_model(
    *,
    config: dict[str, Any],
    model_config: dict[str, str],
    input_root: Path,
    output_root: Path,
    schedule: np.ndarray,
    reward_model: torch.nn.Module,
    device: torch.device,
    save_figures: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Export complete behavior and retrieval analysis for one model.

    Args:
        config (dict[str, Any]): Unified simulation and analysis configuration.
        model_config (dict[str, str]): Active model condition.
        input_root (Path): Common raw-record root.
        output_root (Path): Common analysis root.
        schedule (np.ndarray): Scaled payoff schedule with shape ``[N, 4]``.
        reward_model (torch.nn.Module): Frozen learned reward encoder/decoder.
        device (torch.device): Inference device.
        save_figures (bool): Whether to write PDF and SVG figures.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Agent and query-diagnostic tables.
    """
    analysis_config = config['analysis']
    cells = load_model_cells(
        input_dir=input_root / model_config['name'],
        model_config=model_config,
    )
    trial_table, retrieval_table = build_tables(cells)
    n_anneal_trials = config['choice_policy']['n_anneal_trials']
    block_table, agent_table = build_agent_tables(
        trial_table=trial_table,
        n_anneal_trials=n_anneal_trials,
    )
    agent_deck_table, retrieval_summary = build_retrieval_summaries(
        retrieval_table=retrieval_table,
        n_boot=analysis_config['n_boot'],
        random_seed=analysis_config['random_seed'],
    )
    agent_table = merge_agent_composition(
        agent_table=agent_table,
        agent_deck_table=agent_deck_table,
    )
    trial_summary = build_trial_probability_summary(
        trial_table=trial_table,
        n_boot=analysis_config['n_boot'],
        random_seed=analysis_config['random_seed'],
    )
    block_summary = build_block_behavior_summary(
        block_table=block_table,
        n_boot=analysis_config['n_boot'],
        random_seed=analysis_config['random_seed'],
    )
    trial_table_10 = add_igt_blocks(table=trial_table, block_size=10)
    block_table_10, _ = build_agent_tables(
        trial_table=trial_table_10,
        n_anneal_trials=n_anneal_trials,
    )
    block_summary_10 = build_block_behavior_summary(
        block_table=block_table_10,
        n_boot=analysis_config['n_boot'],
        random_seed=analysis_config['random_seed'],
    )
    diagnostics = build_query_diagnostics(
        model_config=model_config,
        schedule=schedule,
        reward_model=reward_model,
        device=device,
    )
    output_dir = output_root / model_config['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_summary.to_csv(output_dir / 'igt_trial_probability_summary.csv', index=False)
    block_summary.to_csv(output_dir / 'igt_block_behavior_summary.csv', index=False)
    agent_table.to_csv(output_dir / 'igt_agent_behavior_summary.csv', index=False)
    retrieval_table.to_csv(output_dir / 'igt_retrieval_composition_by_deck.csv', index=False)
    retrieval_summary.to_csv(output_dir / 'igt_retrieval_composition_summary.csv', index=False)
    diagnostics.to_csv(output_dir / 'igt_query_similarity_diagnostics.csv', index=False)
    if not save_figures:
        return agent_table, diagnostics

    visual_dir = output_dir / 'visuals'
    visual_dir.mkdir(parents=True, exist_ok=True)
    temporal_weights = sorted(trial_table['temporal_weight'].unique())
    deck_colors = resolve_deck_colors(configured_colors=analysis_config['deck_colors'])
    temporal_weight_colors = resolve_temporal_weight_colors(
        temporal_weights=temporal_weights,
        configured_colors=analysis_config['temporal_weight_colors'],
    )
    contrast_colors = resolve_contrast_colors(configured_colors=analysis_config['contrast_colors'])
    plot_figure_1(
        trial_summary=trial_summary,
        deck_colors=deck_colors,
        temporal_weight_colors=temporal_weight_colors,
        visual_dir=visual_dir,
    )
    plot_figure_2(
        block_summary=block_summary,
        block_summary_10=block_summary_10,
        agent_table=agent_table,
        temporal_weights=temporal_weights,
        temporal_weight_colors=temporal_weight_colors,
        n_boot=analysis_config['n_boot'],
        random_seed=analysis_config['random_seed'],
        visual_dir=visual_dir,
    )
    plot_figure_3(
        block_summary=block_summary,
        block_summary_10=block_summary_10,
        agent_table=agent_table,
        temporal_weights=temporal_weights,
        temporal_weight_colors=temporal_weight_colors,
        contrast_colors=contrast_colors,
        n_boot=analysis_config['n_boot'],
        random_seed=analysis_config['random_seed'],
        visual_dir=visual_dir,
    )
    plot_retrieval_composition(summary=retrieval_summary, visual_dir=visual_dir)
    plot_query_diagnostics(diagnostics=diagnostics, visual_dir=visual_dir)
    return agent_table, diagnostics


def main(args: argparse.Namespace) -> None:
    """
    Analyze the six GENESIS models and the optional PVL reference.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    config = load_simulation_config(args.config)
    analysis_config = config['analysis']
    input_root = args.input_dir or filepaths.EXPERIMENTS_DIR / config['experiment_name']
    output_root = args.output_dir or filepaths.OUTPUT_DIR / analysis_config['output_dir_name']
    output_root.mkdir(parents=True, exist_ok=True)
    schedule = load_igt_schedule(
        schedule_path=filepaths.ROOT / config['task']['schedule_path'],
        reward_scaling_factor=config['task']['reward_scaling_factor'],
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reward_model = load_linear_decoder(checkpoint_path=IGT_REWARD_MODEL_PATH, device=device)
    save_figures = analysis_config['save_figures'] and not args.no_save_figures
    agent_tables = []
    diagnostic_tables = []
    for model_config in config['models']:
        agent_table, diagnostics = analyze_model(
            config=config,
            model_config=model_config,
            input_root=input_root,
            output_root=output_root,
            schedule=schedule,
            reward_model=reward_model,
            device=device,
            save_figures=save_figures,
        )
        agent_tables.append(agent_table)
        diagnostic_tables.append(diagnostics)

    genesis_agents = pd.concat(agent_tables, ignore_index=True)
    all_diagnostics = pd.concat(diagnostic_tables, ignore_index=True)
    retrieval_comparison = summarize_cross_model_agents(
        agent_table=genesis_agents,
        n_boot=analysis_config['n_boot'],
        random_seed=analysis_config['random_seed'],
    )
    retrieval_comparison.to_csv(
        output_root / 'igt_model_behavior_and_retrieval_comparison.csv',
        index=False,
    )
    all_diagnostics.to_csv(output_root / 'igt_query_similarity_comparison.csv', index=False)

    behavior_agent_tables = [genesis_agents]
    pvl_input_dir = input_root / PVL_MODEL_NAME
    if pvl_input_dir.exists():
        pvl_agents = analyze_pvl_model(
            input_dir=pvl_input_dir,
            output_dir=output_root / PVL_MODEL_NAME,
            analysis_config=analysis_config,
            deck_colors=resolve_deck_colors(configured_colors=analysis_config['deck_colors']),
            save_figures=save_figures,
        )
        behavior_agent_tables.append(pvl_agents.assign(temporal_weight=np.nan))
    behavior_agents = pd.concat(behavior_agent_tables, ignore_index=True)
    behavior_comparison = summarize_cross_model_behavior(
        agent_table=behavior_agents,
        n_boot=analysis_config['n_boot'],
        random_seed=analysis_config['random_seed'],
    )
    behavior_comparison.to_csv(output_root / 'igt_model_behavior_comparison.csv', index=False)
    if save_figures:
        plot_cross_model_behavior(
            summary=behavior_comparison,
            model_colors=analysis_config['model_colors'],
            output_dir=output_root,
        )
        plot_cross_model_composition(
            summary=retrieval_comparison,
            model_colors=analysis_config['model_colors'],
            output_dir=output_root,
        )


if __name__ == '__main__':
    main(parse_args())
