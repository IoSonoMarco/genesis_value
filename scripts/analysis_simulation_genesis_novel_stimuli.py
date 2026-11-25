"""
Analyze GENESIS value reconstruction for unseen or seen target stimuli.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.cwd()))

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import filepaths
from utils.simulation_config import get_replacement_output_dir, load_simulation_config


DEFAULT_CONFIG_PATH = filepaths.ROOT / 'configs' / 'simulation_genesis_novel_stimuli.yaml'
COLOR_FACTORS = ('object_color', 'floor_color', 'wall_color')
ACTIVE_COMPONENTS = ('z', 'object_color', 'floor_color', 'wall_color')
CONDITIONS = ('unseen', 'seen')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse novel-stimulus analysis arguments.

    Args:
        argv (list[str] | None): Optional command-line arguments.

    Returns:
        argparse.Namespace: Parsed analysis arguments.
    """
    parser = argparse.ArgumentParser(description='Analyze GENESIS unseen- or seen-stimulus retrieval.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--condition', choices=CONDITIONS, default='unseen')
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--n-boot', type=int)
    parser.add_argument('--random-state', type=int)

    return parser.parse_args(argv)


def load_records(input_dir: Path) -> list[dict[str, Any]]:
    """
    Load all novel-stimulus session records.

    Args:
        input_dir (Path): Simulation record directory.

    Returns:
        list[dict[str, Any]]: Ordered session records.
    """
    records = [
        np.load(record_path, allow_pickle=True).item()
        for record_path in sorted(input_dir.glob('001_session-*.npy'))
    ]
    if not records:
        raise FileNotFoundError(f'no novel-stimulus records found in {input_dir}')

    return records


def record_dataset(record: dict[str, Any]) -> pd.DataFrame:
    """
    Normalize legacy and current dataset membership fields.

    Args:
        record (dict[str, Any]): Simulation record.

    Returns:
        pd.DataFrame: Dataset with explicit target and memory membership.
    """
    dataset = pd.DataFrame(record['dataset']).copy()
    if 'is_target' not in dataset:
        dataset['is_target'] = dataset['role'] == 'target'
    if 'in_memory' not in dataset:
        dataset['in_memory'] = dataset['role'] == 'memory'

    return dataset


def memory_table(record: dict[str, Any]) -> pd.DataFrame:
    """
    Return memory metadata in retrieval-candidate order.

    Args:
        record (dict[str, Any]): Simulation record.

    Returns:
        pd.DataFrame: Ordered episodic-memory rows.
    """
    return record_dataset(record).loc[lambda table: table['in_memory']].sort_values('memory_position')


def validate_records(
    *,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    condition: str = 'unseen',
) -> None:
    """
    Validate the complete record collection before analysis.

    Args:
        records (list[dict[str, Any]]): Loaded session records.
        config (dict[str, Any]): Resolved experiment configuration.
        condition (str): Requested experimental condition.
    """
    if len(records) != config['n_sessions']:
        raise ValueError('record count does not match configured n_sessions')

    session_ids = [int(record['session']['session_id']) for record in records]
    if len(set(session_ids)) != len(session_ids):
        raise ValueError('session IDs must be unique')
    if any(record['schema_version'] not in (1, 2) for record in records):
        raise ValueError('analysis requires novel-stimulus schema version 1 or 2')
    if condition == 'seen' and any(record['schema_version'] == 1 for record in records):
        raise ValueError('legacy schema-v1 records are valid only for the unseen condition')

    expected_budgets = config['grid']['sampling_budget']
    expected_memory_count = 124 if condition == 'unseen' else 125
    for record in records:
        record_condition = record.get('condition', record.get('session', {}).get('condition', 'unseen'))
        if record_condition != condition:
            raise ValueError('record condition does not match the requested condition')
        dataset = record_dataset(record)
        memory = memory_table(record)
        target = dataset.loc[dataset['is_target']]
        if len(dataset) != 125 or dataset['conjunction_id'].nunique() != 125:
            raise ValueError('every record must contain 125 distinct conjunctions')
        if dataset['h5_idx'].nunique() != 125:
            raise ValueError('every record must contain 125 distinct HDF5 images')
        if len(memory) != expected_memory_count or len(target) != 1:
            raise ValueError(f'every record must contain {expected_memory_count} memories and one target')
        target_occurrences = int((memory['conjunction_id'] == int(target['conjunction_id'].iloc[0])).sum())
        if target_occurrences != int(condition == 'seen'):
            raise ValueError('target memory membership does not match the condition')
        if memory['memory_position'].tolist() != list(range(expected_memory_count)):
            raise ValueError('memory positions are not aligned with candidate order')
        if any(
            np.asarray(record['memory_embeddings'][name]).shape[0] != expected_memory_count
            for name in ACTIVE_COMPONENTS
        ):
            raise ValueError('memory embedding arrays have an unexpected row count')
        probabilities = np.asarray(record['target']['sampling_probabilities'])
        scores = np.asarray(record['target']['query_key_scores'])
        if probabilities.shape != (expected_memory_count,) or scores.shape != (expected_memory_count,):
            raise ValueError('retrieval arrays have an unexpected candidate count')
        if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
            raise ValueError('sampling probabilities must be finite and nonnegative')
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError('sampling probabilities must sum to one')
        if condition == 'seen':
            exact_match = record['target'].get('exact_match')
            if not exact_match or not exact_match['target_in_memory']:
                raise ValueError('seen records require exact-target retrieval diagnostics')
            target_position = int(exact_match['memory_position'])
            if int(memory.iloc[target_position]['h5_idx']) != int(target['h5_idx'].iloc[0]):
                raise ValueError('exact-target memory position does not identify the target image')
            ranks = probability_ranks(probabilities)
            if not np.isclose(exact_match['sampling_probability'], probabilities[target_position]):
                raise ValueError('exact-target sampling probability is inconsistent')
            if exact_match['sampling_probability_rank'] != int(ranks[target_position]):
                raise ValueError('exact-target sampling probability rank is inconsistent')
            for sampling_budget in expected_budgets:
                expected_probability = 1.0 - (1.0 - probabilities[target_position]) ** sampling_budget
                stored_probability = exact_match['retrieval_probability_at_least_once'][sampling_budget]
                if not np.isclose(stored_probability, expected_probability):
                    raise ValueError('exact-target budget accessibility is inconsistent')

        retrieval_grid = record['retrieval_grid']
        if [cell['sampling_budget'] for cell in retrieval_grid] != expected_budgets:
            raise ValueError('record does not contain the configured sampling-budget grid')
        for cell in retrieval_grid:
            decoded = np.asarray(cell['decoded_reconstructed_rewards'])
            if len(decoded) != config['n_retrieval_repetitions']:
                raise ValueError('retrieval cell has an unexpected repetition count')
            if not np.isfinite(decoded).all():
                raise ValueError('retrieval cell contains non-finite reconstructions')

    geometry_session_id = config['analysis']['geometry']['session_id']
    if geometry_session_id not in session_ids:
        raise ValueError('configured Plot A session ID does not exist')
    if len(expected_budgets) != 5:
        raise ValueError('Plot B requires exactly five configured sampling budgets')


def build_dataset_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten the generated session datasets.

    Args:
        records (list[dict[str, Any]]): Validated session records.

    Returns:
        pd.DataFrame: Conjunction-level audit table.
    """
    return pd.concat(
        [record_dataset(record) for record in records],
        ignore_index=True,
    ).sort_values(['session_id', 'conjunction_id']).reset_index(drop=True)


def build_retrieval_repetitions(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Build one row per session, budget, and retrieval repetition.

    Args:
        records (list[dict[str, Any]]): Validated session records.

    Returns:
        pd.DataFrame: Repetition-level reconstruction table.
    """
    rows = []
    for record in records:
        target = record['target']
        true_value = float(target['oracle_reward'])
        for cell in record['retrieval_grid']:
            for repetition_id, reconstructed_value in enumerate(cell['decoded_reconstructed_rewards']):
                signed_error = float(reconstructed_value) - true_value
                rows.append({
                    'session_id': int(record['session']['session_id']),
                    'target_conjunction_id': int(target['conjunction_id']),
                    'target_h5_idx': int(target['h5_idx']),
                    'target_object_color': int(target['object_color']),
                    'target_floor_color': int(target['floor_color']),
                    'target_wall_color': int(target['wall_color']),
                    'true_stimulus_value': true_value,
                    'sampling_budget': int(cell['sampling_budget']),
                    'repetition_id': repetition_id,
                    'decoded_reconstructed_reward': float(reconstructed_value),
                    'signed_error': signed_error,
                    'absolute_error': abs(signed_error),
                    'squared_error': signed_error ** 2,
                })

    return pd.DataFrame(rows).sort_values(
        ['session_id', 'sampling_budget', 'repetition_id'],
    ).reset_index(drop=True)


def build_session_budget_summary(retrieval_repetitions: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize reconstruction repetitions within each session and budget.

    Args:
        retrieval_repetitions (pd.DataFrame): Repetition-level reconstructions.

    Returns:
        pd.DataFrame: Session-budget reconstruction summary.
    """
    rows = []
    group_columns = ['session_id', 'sampling_budget']
    for (session_id, sampling_budget), group in retrieval_repetitions.groupby(group_columns, sort=True):
        reconstructed = group['decoded_reconstructed_reward'].to_numpy()
        true_value = float(group['true_stimulus_value'].iloc[0])
        mean_reconstructed = float(reconstructed.mean())
        rows.append({
            'session_id': int(session_id),
            'target_conjunction_id': int(group['target_conjunction_id'].iloc[0]),
            'target_h5_idx': int(group['target_h5_idx'].iloc[0]),
            'target_object_color': int(group['target_object_color'].iloc[0]),
            'target_floor_color': int(group['target_floor_color'].iloc[0]),
            'target_wall_color': int(group['target_wall_color'].iloc[0]),
            'true_stimulus_value': true_value,
            'sampling_budget': int(sampling_budget),
            'n_repetitions': len(group),
            'mean_reconstructed_value': mean_reconstructed,
            'median_reconstructed_value': float(np.median(reconstructed)),
            'reconstruction_sd': float(np.std(reconstructed, ddof=1)),
            'reconstruction_2_5_percentile': float(np.percentile(reconstructed, 2.5)),
            'reconstruction_97_5_percentile': float(np.percentile(reconstructed, 97.5)),
            'signed_bias_of_mean': mean_reconstructed - true_value,
            'absolute_error_of_mean': abs(mean_reconstructed - true_value),
            'repetition_mae': float(group['absolute_error'].mean()),
            'repetition_rmse': float(np.sqrt(group['squared_error'].mean())),
        })

    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def pearson_correlation(true_values: np.ndarray, reconstructed_values: np.ndarray) -> float:
    """
    Calculate Pearson correlation between true and reconstructed values.

    Args:
        true_values (np.ndarray): Oracle session values.
        reconstructed_values (np.ndarray): Session-mean reconstructed values.

    Returns:
        float: Pearson correlation, or NaN for a constant input.
    """
    if np.std(true_values) == 0.0 or np.std(reconstructed_values) == 0.0:
        return np.nan

    return float(np.corrcoef(true_values, reconstructed_values)[0, 1])


def calculate_budget_metrics(session_summary: pd.DataFrame) -> dict[str, float]:
    """
    Calculate across-session reconstruction metrics for one budget.

    Args:
        session_summary (pd.DataFrame): Session rows for one sampling budget.

    Returns:
        dict[str, float]: Session-mean and repetition-level reconstruction metrics.
    """
    true_values = session_summary['true_stimulus_value'].to_numpy()
    reconstructed = session_summary['mean_reconstructed_value'].to_numpy()
    errors = reconstructed - true_values
    regression_slope = np.nan
    if np.std(true_values) > 0.0:
        regression_slope = float(np.polyfit(true_values, reconstructed, deg=1)[0])

    return {
        'mae': float(np.mean(np.abs(errors))),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mean_signed_bias': float(np.mean(errors)),
        'pearson_correlation': pearson_correlation(true_values, reconstructed),
        'regression_slope': regression_slope,
        'repetition_mae': float(session_summary['repetition_mae'].mean()),
        'repetition_rmse': float(session_summary['repetition_rmse'].mean()),
        'reconstruction_sd': float(session_summary['reconstruction_sd'].mean()),
    }


def build_sampling_budget_summary(
    *,
    session_budget_summary: pd.DataFrame,
    n_boot: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Aggregate budget metrics with a session bootstrap.

    Args:
        session_budget_summary (pd.DataFrame): Session-budget summary table.
        n_boot (int): Number of session bootstrap samples.
        random_state (int): Bootstrap seed.

    Returns:
        pd.DataFrame: Across-session metrics and bootstrap uncertainty.
    """
    rows = []
    random_generator = np.random.default_rng(random_state)
    for sampling_budget, group in session_budget_summary.groupby('sampling_budget', sort=True):
        group = group.reset_index(drop=True)
        observed = calculate_budget_metrics(group)
        bootstrap_metrics = {metric_name: [] for metric_name in observed}
        for _ in range(n_boot):
            sampled_positions = random_generator.integers(0, len(group), size=len(group))
            sampled_metrics = calculate_budget_metrics(group.iloc[sampled_positions])
            for metric_name, metric_value in sampled_metrics.items():
                bootstrap_metrics[metric_name].append(metric_value)

        row: dict[str, Any] = {
            'sampling_budget': int(sampling_budget),
            'n_sessions': len(group),
        }
        for metric_name, metric_value in observed.items():
            bootstrap_values = np.asarray(bootstrap_metrics[metric_name])
            finite_values = bootstrap_values[np.isfinite(bootstrap_values)]
            row[metric_name] = metric_value
            row[f'{metric_name}_bootstrap_se'] = float(np.std(finite_values, ddof=1))
            row[f'{metric_name}_ci_lower'] = float(np.percentile(finite_values, 2.5))
            row[f'{metric_name}_ci_upper'] = float(np.percentile(finite_values, 97.5))
        rows.append(row)

    return pd.DataFrame(rows).sort_values('sampling_budget').reset_index(drop=True)


def probability_ranks(probabilities: np.ndarray) -> np.ndarray:
    """
    Rank probabilities descending with memory position as the tie breaker.

    Args:
        probabilities (np.ndarray): Candidate probabilities with shape ``[N]``.

    Returns:
        np.ndarray: One-based probability ranks.
    """
    order = np.lexsort((np.arange(len(probabilities)), -probabilities))
    ranks = np.empty(len(probabilities), dtype=int)
    ranks[order] = np.arange(1, len(probabilities) + 1)

    return ranks


def target_overlap_counts(memory: pd.DataFrame, target: dict[str, Any]) -> np.ndarray:
    """
    Count target-matching colors for every memory episode.

    Args:
        memory (pd.DataFrame): Ordered memory metadata.
        target (dict[str, Any]): Query-target metadata.

    Returns:
        np.ndarray: Matching-color counts with shape ``[N]``.
    """
    return sum(
        (memory[factor_name].to_numpy() == int(target[factor_name])).astype(int)
        for factor_name in COLOR_FACTORS
    )


def build_target_retrieval_summary(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Summarize each target's fixed retrieval distribution.

    Args:
        records (list[dict[str, Any]]): Validated session records.

    Returns:
        pd.DataFrame: Target-level retrieval diagnostics.
    """
    rows = []
    for record in records:
        target = record['target']
        memory = memory_table(record)
        probabilities = np.asarray(target['sampling_probabilities'])
        scores = np.asarray(target['query_key_scores'])
        overlaps = target_overlap_counts(memory, target)
        ranks = probability_ranks(probabilities)
        order = np.lexsort((np.arange(len(probabilities)), -probabilities))
        top_index = int(order[0])
        safe_probabilities = probabilities[probabilities > 0.0]
        exact_match = target.get('exact_match', {})
        target_in_memory = bool(exact_match.get('target_in_memory', False))
        target_position = exact_match.get('memory_position')
        row = {
            'session_id': int(record['session']['session_id']),
            'target_conjunction_id': int(target['conjunction_id']),
            'target_h5_idx': int(target['h5_idx']),
            'true_target_value': float(target['oracle_reward']),
            'target_in_memory': target_in_memory,
            'maximum_candidate_probability': float(probabilities[top_index]),
            'strongest_candidate_probability_rank': 1,
            'top_5_probability_mass': float(probabilities[order[:5]].sum()),
            'top_10_probability_mass': float(probabilities[order[:10]].sum()),
            'retrieval_entropy': float(-(safe_probabilities * np.log(safe_probabilities)).sum()),
            'expected_matching_color_features': float(np.dot(probabilities, overlaps)),
            'highest_probability_memory_position': top_index,
            'highest_probability_conjunction_id': int(memory.iloc[top_index]['conjunction_id']),
            'highest_probability_episode_reward': float(memory.iloc[top_index]['oracle_reward']),
            'highest_probability_query_key_score': float(scores[top_index]),
            'highest_probability_matching_color_features': int(overlaps[top_index]),
            'exact_target_memory_position': target_position,
            'exact_target_query_key_score': exact_match.get('query_key_score'),
            'exact_target_sampling_probability': exact_match.get('sampling_probability'),
            'exact_target_sampling_probability_rank': exact_match.get('sampling_probability_rank'),
            'exact_target_is_highest_probability': exact_match.get('is_highest_probability', False),
        }
        for sampling_budget, probability in exact_match.get(
            'retrieval_probability_at_least_once',
            {},
        ).items():
            row[f'exact_target_retrieval_probability_at_least_once_k_{sampling_budget}'] = probability
        if target_in_memory:
            target_position = int(target_position)
            if ranks[target_position] != int(row['exact_target_sampling_probability_rank']):
                raise ValueError('stored exact-target probability rank is inconsistent')
        rows.append(row)

    return pd.DataFrame(rows).sort_values('session_id').reset_index(drop=True)


def l2_normalize(values: np.ndarray) -> np.ndarray:
    """
    Normalize each embedding row to unit length.

    Args:
        values (np.ndarray): Embeddings with shape ``[N, D]``.

    Returns:
        np.ndarray: Unit-normalized embeddings with shape ``[N, D]``.
    """
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def classical_mds(distances: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Project a distance matrix with Torgerson-Guttman classical MDS.

    Args:
        distances (np.ndarray): Pairwise distances with shape ``[N, N]``.
        n_components (int): Number of projection dimensions.

    Returns:
        np.ndarray: Point coordinates with shape ``[N, n_components]``.
    """
    n_points = distances.shape[0]
    centering = np.eye(n_points) - np.ones((n_points, n_points)) / n_points
    gram_matrix = -0.5 * centering @ (distances ** 2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)
    order = np.argsort(eigenvalues)[::-1]
    positive_eigenvalues = np.maximum(eigenvalues[order[:n_components]], 0.0)

    return eigenvectors[:, order[:n_components]] * np.sqrt(positive_eigenvalues)


def build_geometry_table(
    *,
    record: dict[str, Any],
    top_n_neighbors: int,
) -> pd.DataFrame:
    """
    Project memory and target representations for Plot A.

    Args:
        record (dict[str, Any]): Selected session record.
        top_n_neighbors (int): Number of probability-ranked memories to highlight.

    Returns:
        pd.DataFrame: Memory and target projection points.
    """
    target = record['target']
    memory = memory_table(record)
    target_in_memory = bool(target.get('exact_match', {}).get('target_in_memory', False))
    target_position = target.get('exact_match', {}).get('memory_position')
    memory_blocks = []
    target_blocks = []
    for component_name in ACTIVE_COMPONENTS:
        weight_scale = np.sqrt(float(target['component_weights'][component_name]))
        memory_blocks.append(
            weight_scale * l2_normalize(np.asarray(record['memory_embeddings'][component_name])),
        )
        target_embedding = np.asarray(target['active_embeddings'][component_name]).reshape(1, -1)
        target_blocks.append(weight_scale * l2_normalize(target_embedding))

    memory_representations = np.hstack(memory_blocks)
    if target_in_memory:
        representations = memory_representations
    else:
        representations = np.vstack([memory_representations, np.hstack(target_blocks)])
    coordinates = classical_mds(pairwise_distances(representations, metric='euclidean'))
    probabilities = np.asarray(target['sampling_probabilities'])
    scores = np.asarray(target['query_key_scores'])
    ranks = probability_ranks(probabilities)
    overlaps = target_overlap_counts(memory, target)
    top_mask = ranks <= top_n_neighbors
    rows = []
    for memory_index, (_, memory_row) in enumerate(memory.iterrows()):
        rows.append({
            'session_id': int(record['session']['session_id']),
            'point_role': 'memory',
            'conjunction_id': int(memory_row['conjunction_id']),
            'memory_position': int(memory_row['memory_position']),
            'object_color': int(memory_row['object_color']),
            'floor_color': int(memory_row['floor_color']),
            'wall_color': int(memory_row['wall_color']),
            'oracle_reward': float(memory_row['oracle_reward']),
            'mds_dimension_1': float(coordinates[memory_index, 0]),
            'mds_dimension_2': float(coordinates[memory_index, 1]),
            'sampling_probability': float(probabilities[memory_index]),
            'query_key_score': float(scores[memory_index]),
            'sampling_probability_rank': int(ranks[memory_index]),
            'matching_color_features': int(overlaps[memory_index]),
            'is_top_neighbor': bool(top_mask[memory_index]),
            'is_exact_target': bool(target_in_memory and memory_index == int(target_position)),
        })

    if not target_in_memory:
        rows.append({
            'session_id': int(record['session']['session_id']),
            'point_role': 'target',
            'conjunction_id': int(target['conjunction_id']),
            'memory_position': np.nan,
            'object_color': int(target['object_color']),
            'floor_color': int(target['floor_color']),
            'wall_color': int(target['wall_color']),
            'oracle_reward': float(target['oracle_reward']),
            'mds_dimension_1': float(coordinates[-1, 0]),
            'mds_dimension_2': float(coordinates[-1, 1]),
            'sampling_probability': np.nan,
            'query_key_score': np.nan,
            'sampling_probability_rank': np.nan,
            'matching_color_features': np.nan,
            'is_top_neighbor': False,
            'is_exact_target': True,
        })

    return pd.DataFrame(rows)


def build_retrieval_value_table(record: dict[str, Any]) -> pd.DataFrame:
    """
    Aggregate query-specific retrieval probability by stored reward value.

    Args:
        record (dict[str, Any]): Selected session record.

    Returns:
        pd.DataFrame: Reward-level probability masses and value contributions.
    """
    target = record['target']
    memory = memory_table(record).copy()
    probabilities = np.asarray(target['sampling_probabilities'], dtype=float)
    if len(memory) != len(probabilities):
        raise ValueError('Plot A2 requires aligned memory rewards and probabilities')
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError('Plot A2 probabilities must be finite and nonnegative')
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError('Plot A2 probabilities must sum to one')

    memory['sampling_probability'] = probabilities
    memory['weighted_contribution'] = memory['oracle_reward'] * memory['sampling_probability']
    retrieval_values = memory.groupby('oracle_reward', sort=True).agg(
        memory_count=('conjunction_id', 'size'),
        probability_mass=('sampling_probability', 'sum'),
        weighted_contribution=('weighted_contribution', 'sum'),
    ).reset_index().rename(columns={'oracle_reward': 'reward_value'})
    probability_weighted_estimate = float(retrieval_values['weighted_contribution'].sum())
    decoded_reconstructions = np.concatenate([
        np.asarray(cell['decoded_reconstructed_rewards'], dtype=float)
        for cell in record['retrieval_grid']
    ])
    if not len(decoded_reconstructions) or not np.isfinite(decoded_reconstructions).all():
        raise ValueError('Plot A2 requires finite decoded model reconstructions')

    retrieval_values.insert(0, 'session_id', int(record['session']['session_id']))
    retrieval_values['true_target_value'] = float(target['oracle_reward'])
    retrieval_values['probability_weighted_estimate'] = probability_weighted_estimate
    retrieval_values['model_reconstructed_value'] = float(decoded_reconstructions.mean())
    retrieval_values['n_model_reconstructions'] = len(decoded_reconstructions)

    if int(retrieval_values['memory_count'].sum()) != len(memory):
        raise ValueError('Plot A2 grouped memory counts must match episodic memory')
    if not np.isclose(retrieval_values['probability_mass'].sum(), 1.0):
        raise ValueError('Plot A2 grouped probability mass must sum to one')
    grouped_estimate = float(
        (retrieval_values['reward_value'] * retrieval_values['probability_mass']).sum()
    )
    if not np.isclose(grouped_estimate, probability_weighted_estimate):
        raise ValueError('Plot A2 weighted contributions do not match the grouped reward expectation')

    return retrieval_values


def plot_novel_stimulus_retrieval_explanation(
    *,
    geometry: pd.DataFrame,
    retrieval_values: pd.DataFrame,
    output_path: Path,
    condition: str = 'unseen',
) -> None:
    """
    Plot query geometry and its explanatory scalar retrieval-value view.

    Args:
        geometry (pd.DataFrame): Plot A1 projection points.
        retrieval_values (pd.DataFrame): Plot A2 reward probability masses.
        output_path (Path): Destination PDF path.
        condition (str): Experimental condition.
    """
    memory = geometry.query("point_role == 'memory'")
    target = geometry.loc[geometry['is_exact_target']].iloc[0]
    geometry_session_id = int(target['session_id'])
    retrieval_session_ids = retrieval_values['session_id'].unique()
    if retrieval_session_ids.tolist() != [geometry_session_id]:
        raise ValueError('Plot A1 and A2 must use the same session')

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)
    geometry_axis, value_axis = axes
    top_neighbors = memory.loc[memory['is_top_neighbor']]
    for _, neighbor in top_neighbors.iterrows():
        geometry_axis.plot(
            [target['mds_dimension_1'], neighbor['mds_dimension_1']],
            [target['mds_dimension_2'], neighbor['mds_dimension_2']],
            color='black',
            alpha=0.18,
            linewidth=0.8,
            zorder=1,
        )
    plotted_memory = memory.loc[~memory['is_exact_target']]
    sizes = 25.0 + 900.0 * plotted_memory['sampling_probability'].to_numpy()
    scatter = geometry_axis.scatter(
        plotted_memory['mds_dimension_1'],
        plotted_memory['mds_dimension_2'],
        c=plotted_memory['oracle_reward'],
        s=sizes,
        cmap='coolwarm',
        alpha=0.75,
        linewidths=0.0,
        zorder=2,
    )
    geometry_axis.scatter(
        target['mds_dimension_1'],
        target['mds_dimension_2'],
        marker='*',
        s=(
            280
            if condition == 'unseen'
            else 25.0 + 900.0 * float(target['sampling_probability'])
        ),
        color='#F2C14E',
        edgecolor='black',
        linewidth=1.2,
        label='Unseen query' if condition == 'unseen' else 'Exact stored target',
        zorder=3,
    )
    if condition == 'seen':
        geometry_axis.annotate(
            f"p = {target['sampling_probability']:.3f}, rank = {int(target['sampling_probability_rank'])}",
            (target['mds_dimension_1'], target['mds_dimension_2']),
            xytext=(7, 7),
            textcoords='offset points',
            fontsize=8.5,
        )
    colorbar = figure.colorbar(scatter, ax=geometry_axis)
    colorbar.set_label('Stored stimulus value')
    geometry_axis.set_title('A1  Query geometry')
    geometry_axis.set_xlabel('MDS dimension 1')
    geometry_axis.set_ylabel('MDS dimension 2')
    geometry_axis.legend(frameon=False)
    geometry_axis.spines['top'].set_visible(False)
    geometry_axis.spines['right'].set_visible(False)

    bar_width = 0.75 * np.min(np.diff(retrieval_values['reward_value']))
    value_axis.bar(
        retrieval_values['reward_value'],
        retrieval_values['probability_mass'],
        width=bar_width,
        color='#6A71C5',
        alpha=0.8,
    )
    true_value = float(retrieval_values['true_target_value'].iloc[0])
    probability_weighted_estimate = float(retrieval_values['probability_weighted_estimate'].iloc[0])
    model_reconstructed_value = float(retrieval_values['model_reconstructed_value'].iloc[0])
    value_axis.axvline(
        true_value,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label=rf'$V^\star = {true_value:.3f}$',
    )
    value_axis.axvline(
        probability_weighted_estimate,
        color='#EA2264',
        linestyle='-',
        linewidth=1.7,
        label=rf'$\tilde V(q) = {probability_weighted_estimate:.3f}$',
    )
    value_axis.axvline(
        model_reconstructed_value,
        color='#408175',
        linestyle='-.',
        linewidth=1.7,
        label=rf'$\hat V = {model_reconstructed_value:.3f}$',
    )
    value_axis.set_title('A2  Query-weighted reward evidence')
    value_axis.set_xlabel('Stored reward value')
    value_axis.set_ylabel('Query probability mass')
    value_axis.text(
        0.02,
        0.98,
        'Scalar explanatory view; GENESIS samples,\naverages, and decodes reward embeddings.',
        transform=value_axis.transAxes,
        ha='left',
        va='top',
        fontsize=8.5,
        color='#444444',
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.88, 'pad': 2.0},
        zorder=4,
    )
    value_axis.legend(loc='upper right', frameon=False)
    value_axis.spines['top'].set_visible(False)
    value_axis.spines['right'].set_visible(False)
    condition_label = 'Unseen-stimulus' if condition == 'unseen' else 'Seen-stimulus'
    figure.suptitle(f'{condition_label} retrieval geometry and reward evidence')
    figure.savefig(output_path)
    plt.close(figure)


def build_plot_b_table(
    *,
    session_budget_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build session-level reconstruction points for every Plot B budget.

    Args:
        session_budget_summary (pd.DataFrame): Session-budget summary table.

    Returns:
        pd.DataFrame: One reconstruction point per session and budget.
    """
    points = session_budget_summary.copy().rename(columns={
        'target_object_color': 'object_color',
        'target_floor_color': 'floor_color',
        'target_wall_color': 'wall_color',
        'absolute_error_of_mean': 'absolute_error',
        'signed_bias_of_mean': 'signed_bias',
    })

    return points[[
        'session_id',
        'target_conjunction_id',
        'target_h5_idx',
        'object_color',
        'floor_color',
        'wall_color',
        'true_stimulus_value',
        'sampling_budget',
        'mean_reconstructed_value',
        'absolute_error',
        'signed_bias',
        'reconstruction_sd',
        'reconstruction_2_5_percentile',
        'reconstruction_97_5_percentile',
        'repetition_mae',
        'repetition_rmse',
    ]].sort_values(['sampling_budget', 'session_id']).reset_index(drop=True)


def plot_reconstruction(
    *,
    points: pd.DataFrame,
    budget_summary: pd.DataFrame,
    sampling_budgets: list[int],
    output_path: Path,
    condition: str = 'unseen',
    limits: tuple[float, float] | None = None,
) -> None:
    """
    Plot budget-specific calibration and repetition-level reconstruction error.

    Args:
        points (pd.DataFrame): Plot B session-budget points.
        budget_summary (pd.DataFrame): Across-session budget metrics.
        sampling_budgets (list[int]): Ordered sampling budgets.
        output_path (Path): Destination PDF path.
        condition (str): Experimental condition.
        limits (tuple[float, float] | None): Optional shared plot limits.
    """
    if len(sampling_budgets) != 5:
        raise ValueError('Plot B requires exactly five sampling budgets')

    plotted_values = np.concatenate([
        points['true_stimulus_value'].to_numpy(),
        points['mean_reconstructed_value'].to_numpy(),
        points['reconstruction_2_5_percentile'].to_numpy(),
        points['reconstruction_97_5_percentile'].to_numpy(),
    ])
    if limits is None:
        padding = max(0.1, 0.05 * (plotted_values.max() - plotted_values.min()))
        limits = (float(plotted_values.min() - padding), float(plotted_values.max() + padding))
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.5))
    calibration_axes = axes.flatten()[:5]

    for axis, sampling_budget in zip(calibration_axes, sampling_budgets):
        budget_points = points.loc[points['sampling_budget'] == sampling_budget]
        if budget_points.empty:
            raise ValueError(f'Plot B has no points for sampling budget {sampling_budget}')

        axis.vlines(
            budget_points['true_stimulus_value'],
            budget_points['reconstruction_2_5_percentile'],
            budget_points['reconstruction_97_5_percentile'],
            color='#2E338E',
            alpha=0.18,
            linewidth=0.8,
            zorder=1,
        )
        axis.scatter(
            budget_points['true_stimulus_value'],
            budget_points['mean_reconstructed_value'],
            color='#2E338E',
            alpha=0.62,
            s=25,
            linewidths=0.0,
            zorder=2,
        )
        axis.plot(limits, limits, linestyle='--', color='black', linewidth=0.9)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect('equal', adjustable='box')
        axis.set_title(f'k = {sampling_budget}')
        axis.set_xlabel('True stimulus value')
        axis.set_ylabel('Reconstructed stimulus value')
        if sampling_budget == 20:
            metrics = budget_summary.loc[budget_summary['sampling_budget'] == sampling_budget].iloc[0]
            axis.text(
                0.04,
                0.96,
                f"r = {metrics['pearson_correlation']:.3f}\n"
                f"slope = {metrics['regression_slope']:.3f}\n"
                f"MAE = {metrics['mae']:.3f}\n"
                f"RMSE = {metrics['rmse']:.3f}",
                transform=axis.transAxes,
                va='top',
                fontsize=8.0,
            )
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)

    error_axis = axes.flatten()[5]
    ordered_summary = budget_summary.set_index('sampling_budget').loc[sampling_budgets].reset_index()
    lower_errors = ordered_summary['repetition_mae'] - ordered_summary['repetition_mae_ci_lower']
    upper_errors = ordered_summary['repetition_mae_ci_upper'] - ordered_summary['repetition_mae']
    error_axis.errorbar(
        ordered_summary['sampling_budget'],
        ordered_summary['repetition_mae'],
        yerr=np.vstack([lower_errors, upper_errors]),
        color='#EA2264',
        marker='o',
        markersize=5,
        linewidth=1.4,
        capsize=3,
    )
    error_axis.set_xticks(sampling_budgets)
    error_axis.set_xlabel('Sampling budget (k)')
    error_axis.set_ylabel('Repetition-level MAE')
    error_axis.set_title('Sampling error')
    error_axis.spines['top'].set_visible(False)
    error_axis.spines['right'].set_visible(False)
    condition_label = 'Unseen-stimulus' if condition == 'unseen' else 'Seen-stimulus'
    figure.suptitle(f'{condition_label} reward reconstruction')
    figure.tight_layout(pad=1.4)
    figure.savefig(output_path)
    plt.close(figure)


def build_seen_unseen_comparison(
    *,
    seen_records: list[dict[str, Any]],
    unseen_records: list[dict[str, Any]],
    sampling_budget: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build paired session rows and condition metrics for seen versus unseen targets.

    Args:
        seen_records (list[dict[str, Any]]): Validated seen records.
        unseen_records (list[dict[str, Any]]): Validated unseen records.
        sampling_budget (int): Sampling budget used for comparison.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Paired sessions and condition metrics.
    """
    condition_summaries = {}
    for condition, records in (('seen', seen_records), ('unseen', unseen_records)):
        repetitions = build_retrieval_repetitions(records)
        summary = build_session_budget_summary(repetitions)
        condition_summaries[condition] = summary.loc[summary['sampling_budget'] == sampling_budget].copy()

    key_columns = ['session_id', 'target_conjunction_id', 'target_h5_idx']
    seen = condition_summaries['seen'].rename(columns={
        'true_stimulus_value': 'seen_true_stimulus_reward',
        'mean_reconstructed_value': 'seen_reconstructed_reward',
        'absolute_error_of_mean': 'seen_absolute_reconstruction_error',
        'reconstruction_2_5_percentile': 'seen_reconstruction_2_5_percentile',
        'reconstruction_97_5_percentile': 'seen_reconstruction_97_5_percentile',
    })
    unseen = condition_summaries['unseen'].rename(columns={
        'true_stimulus_value': 'unseen_true_stimulus_reward',
        'mean_reconstructed_value': 'unseen_reconstructed_reward',
        'absolute_error_of_mean': 'unseen_absolute_reconstruction_error',
        'reconstruction_2_5_percentile': 'unseen_reconstruction_2_5_percentile',
        'reconstruction_97_5_percentile': 'unseen_reconstruction_97_5_percentile',
    })
    paired = seen.merge(
        unseen,
        on=key_columns,
        how='outer',
        validate='one_to_one',
        indicator=True,
        suffixes=('_seen', '_unseen'),
    )
    if not (paired['_merge'] == 'both').all():
        raise ValueError('seen and unseen records do not contain identical target session keys')
    if not np.allclose(
        paired['seen_true_stimulus_reward'],
        paired['unseen_true_stimulus_reward'],
    ):
        raise ValueError('paired seen and unseen targets have different oracle rewards')

    seen_retrieval = build_target_retrieval_summary(seen_records)[[
        'session_id',
        'target_conjunction_id',
        'target_h5_idx',
        'exact_target_sampling_probability',
    ]]
    paired = paired.merge(seen_retrieval, on=key_columns, validate='one_to_one')
    paired['true_stimulus_reward'] = paired['seen_true_stimulus_reward']
    paired['difference_in_absolute_reconstruction_error'] = (
        paired['seen_absolute_reconstruction_error']
        - paired['unseen_absolute_reconstruction_error']
    )
    paired = paired[[
        *key_columns,
        'true_stimulus_reward',
        'seen_reconstructed_reward',
        'unseen_reconstructed_reward',
        'seen_absolute_reconstruction_error',
        'unseen_absolute_reconstruction_error',
        'difference_in_absolute_reconstruction_error',
        'exact_target_sampling_probability',
        'seen_reconstruction_2_5_percentile',
        'seen_reconstruction_97_5_percentile',
        'unseen_reconstruction_2_5_percentile',
        'unseen_reconstruction_97_5_percentile',
    ]].sort_values('session_id').reset_index(drop=True)

    metrics = []
    for condition, summary in condition_summaries.items():
        condition_metrics = calculate_budget_metrics(summary)
        metrics.append({
            'condition': condition,
            'sampling_budget': sampling_budget,
            'n_sessions': len(summary),
            'pearson_correlation': condition_metrics['pearson_correlation'],
            'regression_slope': condition_metrics['regression_slope'],
            'mae': condition_metrics['mae'],
            'rmse': condition_metrics['rmse'],
        })

    return paired, pd.DataFrame(metrics)


def plot_seen_unseen_comparison(
    *,
    comparison: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot matched unseen and seen reconstruction at k=20.

    Args:
        comparison (pd.DataFrame): Paired session reconstruction table.
        metrics (pd.DataFrame): Condition-level comparison metrics.
        output_path (Path): Destination PDF path.
    """
    plotted_values = np.concatenate([
        comparison[column].to_numpy()
        for column in (
            'true_stimulus_reward',
            'seen_reconstructed_reward',
            'unseen_reconstructed_reward',
            'seen_reconstruction_2_5_percentile',
            'seen_reconstruction_97_5_percentile',
            'unseen_reconstruction_2_5_percentile',
            'unseen_reconstruction_97_5_percentile',
        )
    ])
    padding = max(0.1, 0.05 * (plotted_values.max() - plotted_values.min()))
    limits = (float(plotted_values.min() - padding), float(plotted_values.max() + padding))
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), constrained_layout=True)
    colors = {'unseen': '#2E338E', 'seen': '#408175'}
    for axis, condition in zip(axes, ('unseen', 'seen')):
        reconstructed_column = f'{condition}_reconstructed_reward'
        lower_column = f'{condition}_reconstruction_2_5_percentile'
        upper_column = f'{condition}_reconstruction_97_5_percentile'
        axis.vlines(
            comparison['true_stimulus_reward'],
            comparison[lower_column],
            comparison[upper_column],
            color=colors[condition],
            alpha=0.18,
            linewidth=0.8,
        )
        axis.scatter(
            comparison['true_stimulus_reward'],
            comparison[reconstructed_column],
            color=colors[condition],
            alpha=0.62,
            s=25,
            linewidths=0.0,
        )
        axis.plot(limits, limits, linestyle='--', color='black', linewidth=0.9)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect('equal', adjustable='box')
        condition_metrics = metrics.loc[metrics['condition'] == condition].iloc[0]
        axis.set_title(f"{condition.capitalize()} stimulus at k = 20")
        axis.set_xlabel('True stimulus reward')
        axis.set_ylabel('Mean reconstructed reward')
        axis.text(
            0.04,
            0.96,
            f"r = {condition_metrics['pearson_correlation']:.3f}\n"
            f"slope = {condition_metrics['regression_slope']:.3f}\n"
            f"MAE = {condition_metrics['mae']:.3f}\n"
            f"RMSE = {condition_metrics['rmse']:.3f}",
            transform=axis.transAxes,
            va='top',
            fontsize=8.5,
        )
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)

    figure.savefig(output_path)
    plt.close(figure)


def condition_output_dir(
    *,
    root_dir: Path,
    output_name: str,
    sample_replacement: bool,
    condition: str,
) -> Path:
    """
    Resolve a legacy-compatible condition output directory.

    Args:
        root_dir (Path): Experiment or analysis root.
        output_name (str): Base output directory name.
        sample_replacement (bool): Whether retrieval samples with replacement.
        condition (str): Experimental condition.

    Returns:
        Path: Condition-specific replacement directory.
    """
    if condition == 'unseen':
        condition_root = root_dir
        condition_name = output_name
    else:
        condition_root = root_dir / output_name
        condition_name = condition

    return get_replacement_output_dir(
        root_dir=condition_root,
        output_name=condition_name,
        sample_replacement=sample_replacement,
    )


def resolve_directories(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[Path, Path]:
    """
    Resolve simulation input and analysis output directories.

    Args:
        args (argparse.Namespace): Parsed analysis arguments.
        config (dict[str, Any]): Resolved experiment configuration.

    Returns:
        tuple[Path, Path]: Input and output directories.
    """
    input_dir = args.input_dir or get_replacement_output_dir(
        root_dir=(
            filepaths.EXPERIMENTS_DIR
            if args.condition == 'unseen'
            else filepaths.EXPERIMENTS_DIR / config['experiment_name']
        ),
        output_name=config['experiment_name'] if args.condition == 'unseen' else args.condition,
        sample_replacement=config['sample_replacement'],
    )
    output_dir = args.output_dir or condition_output_dir(
        root_dir=filepaths.OUTPUT_DIR,
        output_name=config['analysis']['output_dir_name'],
        sample_replacement=config['sample_replacement'],
        condition=args.condition,
    )

    return input_dir, output_dir


def generate_comparison_if_available(*, config: dict[str, Any]) -> None:
    """
    Export the paired comparison when both default record sets are valid.

    Args:
        config (dict[str, Any]): Resolved experiment configuration.
    """
    record_directories = {
        condition: condition_output_dir(
            root_dir=filepaths.EXPERIMENTS_DIR,
            output_name=config['experiment_name'],
            sample_replacement=config['sample_replacement'],
            condition=condition,
        )
        for condition in CONDITIONS
    }
    try:
        records = {
            condition: load_records(record_directory)
            for condition, record_directory in record_directories.items()
        }
        for condition, condition_records in records.items():
            validate_records(records=condition_records, config=config, condition=condition)
        comparison, metrics = build_seen_unseen_comparison(
            seen_records=records['seen'],
            unseen_records=records['unseen'],
        )
    except (FileNotFoundError, ValueError) as error:
        print(f'Skipped seen-versus-unseen comparison: {error}')
        return

    comparison_dir = condition_output_dir(
        root_dir=filepaths.OUTPUT_DIR,
        output_name=config['analysis']['output_dir_name'],
        sample_replacement=config['sample_replacement'],
        condition='comparison',
    )
    visuals_dir = comparison_dir / 'visuals'
    visuals_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_dir / 'seen_unseen_comparison.csv', index=False)
    metrics.to_csv(comparison_dir / 'seen_unseen_condition_metrics.csv', index=False)
    plot_seen_unseen_comparison(
        comparison=comparison,
        metrics=metrics,
        output_path=visuals_dir / 'plot_seen_unseen_value_reconstruction.pdf',
    )
    print(f'Saved seen-versus-unseen comparison to {comparison_dir}')


def run_analysis(args: argparse.Namespace) -> None:
    """
    Validate records, export tables, and generate both coded panels.

    Args:
        args (argparse.Namespace): Parsed analysis arguments.
    """
    config = load_simulation_config(args.config)
    analysis_config = config['analysis']
    n_boot = args.n_boot if args.n_boot is not None else analysis_config['n_boot']
    random_state = args.random_state if args.random_state is not None else analysis_config['random_state']
    input_dir, output_dir = resolve_directories(args=args, config=config)
    records = load_records(input_dir)
    validate_records(records=records, config=config, condition=args.condition)

    output_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = output_dir / 'visuals'
    visuals_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset_table(records)
    repetitions = build_retrieval_repetitions(records)
    session_summary = build_session_budget_summary(repetitions)
    budget_summary = build_sampling_budget_summary(
        session_budget_summary=session_summary,
        n_boot=n_boot,
        random_state=random_state,
    )
    retrieval_summary = build_target_retrieval_summary(records)
    geometry_record = next(
        record
        for record in records
        if record['session']['session_id'] == analysis_config['geometry']['session_id']
    )
    geometry = build_geometry_table(
        record=geometry_record,
        top_n_neighbors=analysis_config['geometry']['top_n_neighbors'],
    )
    retrieval_values = build_retrieval_value_table(geometry_record)
    plot_b_points = build_plot_b_table(
        session_budget_summary=session_summary,
    )

    tables: dict[str, pd.DataFrame] = {
        'novel_stimuli_dataset.csv': dataset,
        'retrieval_repetitions.csv': repetitions,
        'session_budget_summary.csv': session_summary,
        'sampling_budget_summary.csv': budget_summary,
        'target_retrieval_summary.csv': retrieval_summary,
        'plot_a_geometry_points.csv': geometry,
        'plot_a2_retrieval_value_points.csv': retrieval_values,
        'plot_b_reconstruction_points.csv': plot_b_points,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)

    plot_novel_stimulus_retrieval_explanation(
        geometry=geometry,
        retrieval_values=retrieval_values,
        output_path=visuals_dir / 'plot_a_novel_stimulus_geometry.pdf',
        condition=args.condition,
    )
    plot_reconstruction(
        points=plot_b_points,
        budget_summary=budget_summary,
        sampling_budgets=config['grid']['sampling_budget'],
        output_path=visuals_dir / 'plot_b_novel_stimulus_value_reconstruction.pdf',
        condition=args.condition,
    )
    print(f'Saved {args.condition}-stimulus analysis to {output_dir}')
    generate_comparison_if_available(config=config)


def main() -> None:
    run_analysis(parse_args())


if __name__ == '__main__':
    main()
