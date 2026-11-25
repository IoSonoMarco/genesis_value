"""
Replay saved IGT histories under counterfactual retrieval and policy settings.
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path.cwd()))

from models import load_linear_decoder
from models.agent import IowaGamblingTaskPolicy
from models.memory import EpisodicSamplingModuleIowaGamblingTask, IGTQuery
from models.memory.utils import generate_temporal_embeddings
from scripts.simulation_genesis_igt_task_temporal_valence import (
    IGT_REWARD_MODEL_PATH,
    get_query_valence_embedding,
    get_valence_embedding,
)
from utils import filepaths
from utils.simulation_config import load_simulation_config


DEFAULT_CONFIG_PATH = filepaths.ROOT / 'configs' / 'diagnose_igt_retrieval_replay.yaml'
DECK_NAMES = ('A', 'B', 'C', 'D')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse fixed-history IGT retrieval diagnostic arguments.

    Args:
        argv (list[str] | None): Optional CLI arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Replay IGT retrieval from a saved simulation cell.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--source-cell', type=Path, required=True)
    parser.add_argument('--agent-id', type=int, default=0)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-save-figures', action='store_true')
    return parser.parse_args(argv)


def build_temporal_embeddings(config: dict[str, Any]) -> torch.Tensor:
    """
    Recreate the temporal trajectory used by a saved simulation cell.

    Args:
        config (dict[str, Any]): Resolved simulation configuration.

    Returns:
        torch.Tensor: Temporal embeddings with shape ``[N, D]``.
    """
    temporal_config = config['temporal']
    random_state = np.random.RandomState(temporal_config['random_seed'])
    embeddings = generate_temporal_embeddings(
        [temporal_config['embedding_angle']] * config['task']['n_trials'],
        dim=temporal_config['embedding_dim'],
        random_state=random_state,
    )
    return torch.tensor(embeddings).float()


@torch.no_grad()
def build_reward_embedding_cache(
    *,
    history: list[dict[str, Any]],
    reward_model: torch.nn.Module,
    device: torch.device,
) -> dict[float, torch.Tensor]:
    """
    Encode all observed source-history rewards once.

    Args:
        history (list[dict[str, Any]]): Source trial history.
        reward_model (torch.nn.Module): Frozen IGT reward encoder/decoder.
        device (torch.device): Inference device.

    Returns:
        dict[float, torch.Tensor]: Reward embeddings keyed by scalar reward.
    """
    rewards = sorted({float(trial['reward']) for trial in history})
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    embeddings = reward_model.encode(reward_tensor).cpu()
    return {
        reward: embedding
        for reward, embedding in zip(rewards, embeddings)
    }


def get_agent_run(*, cell: dict[str, Any], agent_id: int) -> dict[str, Any]:
    """
    Resolve one saved agent run by its stable identifier.

    Args:
        cell (dict[str, Any]): Loaded simulation cell.
        agent_id (int): Saved agent identifier.

    Returns:
        dict[str, Any]: Complete saved agent run.
    """
    return next(agent_run for agent_run in cell['agent_runs'] if agent_run['agent_id'] == agent_id)


def replay_agent_history(
    *,
    cell: dict[str, Any],
    agent_id: int,
    temporal_embeddings: torch.Tensor,
    reward_embeddings: dict[float, torch.Tensor],
    reward_decoder: torch.nn.Module,
    query_key_normalizations: list[str],
    temporal_weights: list[float],
    retrieval_temperatures: list[float],
    policy_min_temperatures: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all counterfactual settings against one immutable source history.

    Args:
        cell (dict[str, Any]): Loaded source simulation cell.
        agent_id (int): Source agent identifier.
        temporal_embeddings (torch.Tensor): Shared temporal trajectory with shape ``[N, D]``.
        reward_embeddings (dict[float, torch.Tensor]): Reward embeddings keyed by observed reward.
        reward_decoder (torch.nn.Module): Frozen reward decoder.
        query_key_normalizations (list[str]): Retrieval normalization conditions.
        temporal_weights (list[float]): Temporal retrieval weights.
        retrieval_temperatures (list[float]): Retrieval softmax temperatures.
        policy_min_temperatures (list[float]): Counterfactual policy minimum temperatures.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Candidate, deck, and choice diagnostic tables.
    """
    source_config = cell['config']
    source_run = get_agent_run(cell=cell, agent_id=agent_id)
    history = source_run['simulation_log']
    retrieval_conditions = list(product(
        query_key_normalizations,
        temporal_weights,
        retrieval_temperatures,
    ))
    samplers = {
        condition: EpisodicSamplingModuleIowaGamblingTask(
            reward_decoder=reward_decoder,
            k_samples=source_config['episodic_sampling']['sampling_budget'],
            sampling_temperature=condition[2],
            query_key_normalization=condition[0],
            device=next(reward_decoder.parameters(), torch.empty(0)).device,
        )
        for condition in retrieval_conditions
    }
    component_weights = {
        condition: {
            'time_embedding': condition[1],
            'valence_embedding': 1.0 - condition[1],
        }
        for condition in retrieval_conditions
    }
    candidate_rows = []
    deck_rows = []
    choice_rows = []
    query_valence = source_config['query_valence']
    query_valence_embedding = get_query_valence_embedding(query_valence=query_valence)
    choice_policy_config = source_config['choice_policy']

    for trial in history:
        trial_id = trial['trial_id']
        query = IGTQuery(
            time_embedding=temporal_embeddings[trial_id],
            valence_embedding=query_valence_embedding,
        )

        for condition, sampler in samplers.items():
            normalization, temporal_weight, retrieval_temperature = condition
            action_values, metadata = sampler.estimate_expected_action_values(
                query=query,
                weights=component_weights[condition],
                return_metadata=True,
            )
            condition_columns = {
                'agent_id': agent_id,
                'trial_id': trial_id,
                'query_valence': query_valence,
                'query_key_normalization': normalization,
                'temporal_weight': temporal_weight,
                'valence_weight': 1.0 - temporal_weight,
                'retrieval_temperature': retrieval_temperature,
            }

            for deck in DECK_NAMES:
                deck_metadata = metadata[deck]
                probabilities = deck_metadata['sampling_probabilities']
                candidate_indices = deck_metadata['candidate_indices']
                expected_lag = np.nan
                effective_sample_size = np.nan
                if len(candidate_indices):
                    lags = np.asarray([
                        trial_id - sampler.memory_log[candidate_index]['trial_metadata']['trial_id']
                        for candidate_index in candidate_indices
                    ])
                    expected_lag = float(np.dot(probabilities, lags))
                    effective_sample_size = float(1.0 / np.square(probabilities).sum())

                deck_rows.append({
                    **condition_columns,
                    'deck': deck,
                    'available_memory_count': deck_metadata['available_memory_count'],
                    'expected_action_value': action_values[deck],
                    'expected_episodic_lag': expected_lag,
                    'effective_sample_size': effective_sample_size,
                })

                for candidate_position, candidate_index in enumerate(candidate_indices):
                    memory_log = sampler.memory_log[candidate_index]
                    raw_similarities = deck_metadata['raw_component_similarities']
                    normalized_similarities = deck_metadata['normalized_component_similarities']
                    missing_component = np.full(len(candidate_indices), np.nan)
                    candidate_rows.append({
                        **condition_columns,
                        'deck': deck,
                        'candidate_index': candidate_index,
                        'candidate_trial_id': memory_log['trial_metadata']['trial_id'],
                        'candidate_lag': trial_id - memory_log['trial_metadata']['trial_id'],
                        'candidate_reward': memory_log['reward'],
                        'raw_time_similarity': raw_similarities.get(
                            'time_embedding',
                            missing_component,
                        )[candidate_position],
                        'raw_valence_similarity': raw_similarities.get(
                            'valence_embedding',
                            missing_component,
                        )[candidate_position],
                        'normalized_time_similarity': (
                            normalized_similarities.get('time_embedding', missing_component)[candidate_position]
                        ),
                        'normalized_valence_similarity': (
                            normalized_similarities.get('valence_embedding', missing_component)[candidate_position]
                        ),
                        'query_key_score': deck_metadata['query_key_scores'][candidate_position],
                        'sampling_probability': probabilities[candidate_position],
                    })

            for policy_min_temperature in policy_min_temperatures:
                policy = IowaGamblingTaskPolicy(
                    max_choice_temperature=choice_policy_config['max_choice_temperature'],
                    min_choice_temperature=policy_min_temperature,
                    n_anneal_trials=choice_policy_config['n_anneal_trials'],
                )
                policy_output = policy.choose_deck(
                    action_values=action_values,
                    trial_id=trial_id,
                    random_state=np.random.RandomState(0),
                )
                probabilities = np.asarray([
                    policy_output['p_a'],
                    policy_output['p_b'],
                    policy_output['p_c'],
                    policy_output['p_d'],
                ])
                choice_rows.append({
                    **condition_columns,
                    'policy_min_temperature': policy_min_temperature,
                    'source_chosen_deck': trial['chosen_deck'],
                    'counterfactual_choice_temperature': policy_output['choice_temperature'],
                    'expected_value_a': action_values['A'],
                    'expected_value_b': action_values['B'],
                    'expected_value_c': action_values['C'],
                    'expected_value_d': action_values['D'],
                    'p_a': policy_output['p_a'],
                    'p_b': policy_output['p_b'],
                    'p_c': policy_output['p_c'],
                    'p_d': policy_output['p_d'],
                    'max_choice_probability': float(probabilities.max()),
                    'choice_entropy': float(-np.sum(probabilities * np.log(probabilities))),
                    'action_value_span': float(max(action_values.values()) - min(action_values.values())),
                })

        reward = float(trial['reward'])
        for sampler in samplers.values():
            sampler.store_episode(
                chosen_deck=trial['chosen_deck'],
                time_embedding=temporal_embeddings[trial_id],
                valence_embedding=get_valence_embedding(reward=reward),
                reward_embedding=reward_embeddings[reward],
                reward=reward,
                trial_metadata={'trial_id': trial_id},
            )

    return pd.DataFrame(candidate_rows), pd.DataFrame(deck_rows), pd.DataFrame(choice_rows)


def plot_diagnostics(
    *,
    deck_table: pd.DataFrame,
    choice_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot retrieval concentration and downstream counterfactual policy entropy.

    Args:
        deck_table (pd.DataFrame): Per-deck expected retrieval table.
        choice_table (pd.DataFrame): Counterfactual policy table.
        output_dir (Path): Diagnostic artifact directory.
    """
    grouped_deck_table = deck_table.groupby(
        ['trial_id', 'query_key_normalization', 'temporal_weight'],
        as_index=False,
    ).agg({
        'expected_episodic_lag': 'mean',
        'effective_sample_size': 'mean',
    })
    normalizations = sorted(grouped_deck_table['query_key_normalization'].unique())
    temporal_weights = sorted(grouped_deck_table['temporal_weight'].unique())
    colors = {
        temporal_weight: plt.get_cmap('viridis', len(temporal_weights))(index)
        for index, temporal_weight in enumerate(temporal_weights)
    }
    figure, axes = plt.subplots(
        len(normalizations),
        2,
        figsize=(10, 2.6 * len(normalizations)),
        sharex=True,
    )
    figure.subplots_adjust(left=0.09, right=0.78, hspace=0.12, wspace=0.1)
    for row_index, normalization in enumerate(normalizations):
        group = grouped_deck_table[grouped_deck_table['query_key_normalization'] == normalization]
        for temporal_weight, condition in group.groupby('temporal_weight', sort=True):
            label = f'$w_T={temporal_weight:g}$'
            axes[row_index, 0].plot(
                condition['trial_id'],
                condition['expected_episodic_lag'],
                color=colors[temporal_weight],
                label=label,
            )
            axes[row_index, 1].plot(
                condition['trial_id'],
                condition['effective_sample_size'],
                color=colors[temporal_weight],
                label=label,
            )
        axes[row_index, 0].set_ylabel(f'{normalization}\nEpisodes')
        for axis in axes[row_index]:
            axis.spines[['top', 'right']].set_visible(False)
    axes[0, 0].set_title('Expected retrieval lag')
    axes[0, 1].set_title('Effective retrieval pool')
    axes[-1, 0].set_xlabel('Trial')
    axes[-1, 1].set_xlabel('Trial')
    handles, labels = axes[0, 1].get_legend_handles_labels()
    figure.legend(handles, labels, loc='center right', frameon=False, fontsize=7)
    figure.savefig(output_dir / 'figure_1_igt_retrieval_replay.pdf', bbox_inches='tight')
    plt.close(figure)

    policy_min_temperatures = sorted(choice_table['policy_min_temperature'].unique())
    figure, axes = plt.subplots(
        len(normalizations),
        len(policy_min_temperatures),
        figsize=(3.2 * len(policy_min_temperatures), 2.4 * len(normalizations)),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(left=0.1, right=0.78, hspace=0.12, wspace=0.12)
    for row_index, normalization in enumerate(normalizations):
        for column_index, policy_min_temperature in enumerate(policy_min_temperatures):
            axis = axes[row_index, column_index]
            condition_table = choice_table[
                (choice_table['query_key_normalization'] == normalization)
                & (choice_table['policy_min_temperature'] == policy_min_temperature)
            ]
            for temporal_weight, condition in condition_table.groupby('temporal_weight', sort=True):
                axis.plot(
                    condition['trial_id'],
                    condition['choice_entropy'],
                    color=colors[temporal_weight],
                    label=f'$w_T={temporal_weight:g}$',
                )
            axis.spines[['top', 'right']].set_visible(False)
            if row_index == 0:
                axis.set_title(f'$T_C={policy_min_temperature:g}$')
            if column_index == 0:
                axis.set_ylabel(f'{normalization}\nNats')
            if row_index == len(normalizations) - 1:
                axis.set_xlabel('Trial')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='center right', frameon=False, fontsize=7)
    figure.savefig(output_dir / 'figure_2_igt_counterfactual_policy_entropy.pdf', bbox_inches='tight')
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    """
    Load one saved IGT cell and export counterfactual replay diagnostics.

    Args:
        args (argparse.Namespace): Parsed diagnostic arguments.
    """
    config = load_simulation_config(args.config)
    cell = np.load(args.source_cell, allow_pickle=True).item()
    source_config = cell['config']
    source_run = get_agent_run(cell=cell, agent_id=args.agent_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reward_model = load_linear_decoder(checkpoint_path=IGT_REWARD_MODEL_PATH, device=device)
    temporal_embeddings = build_temporal_embeddings(source_config)
    reward_embeddings = build_reward_embedding_cache(
        history=source_run['simulation_log'],
        reward_model=reward_model,
        device=device,
    )
    candidate_table, deck_table, choice_table = replay_agent_history(
        cell=cell,
        agent_id=args.agent_id,
        temporal_embeddings=temporal_embeddings,
        reward_embeddings=reward_embeddings,
        reward_decoder=reward_model.decoder,
        query_key_normalizations=config['grid']['query_key_normalization'],
        temporal_weights=config['grid']['temporal_weight'],
        retrieval_temperatures=config['grid']['retrieval_temperature'],
        policy_min_temperatures=config['policy']['min_choice_temperature'],
    )
    output_dir = args.output_dir or filepaths.OUTPUT_DIR / config['output_dir_name'] / (
        f'{args.source_cell.stem}_agent-{args.agent_id}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_table.to_csv(output_dir / 'igt_retrieval_candidates.csv', index=False)
    deck_table.to_csv(output_dir / 'igt_retrieval_decks.csv', index=False)
    choice_table.to_csv(output_dir / 'igt_counterfactual_choices.csv', index=False)
    if config['save_figures'] and not args.no_save_figures:
        plot_diagnostics(
            deck_table=deck_table,
            choice_table=choice_table,
            output_dir=output_dir,
        )


if __name__ == '__main__':
    main(parse_args())
