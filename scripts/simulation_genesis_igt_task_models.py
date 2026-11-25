"""
Simulate the unified six-model GENESIS Iowa Gambling Task experiment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Literal

from joblib import Parallel, delayed
import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))

from models import load_linear_decoder
from models.agent import IowaGamblingTaskPolicy
from models.memory import EpisodicSamplingModuleIowaGamblingTask, IGTQuery, IGTQueryComponent
from scripts.simulation_genesis_igt_task_temporal_valence import (
    IGT_REWARD_MODEL_PATH,
    VALENCE_EMBEDDINGS,
    build_reward_embedding_cache,
    build_temporal_embeddings,
    config_to_filename,
    get_valence_embedding,
    load_igt_schedule,
)
from utils import filepaths
from utils.simulation_config import load_simulation_config


DEFAULT_SIMULATION_CONFIG_PATH = filepaths.ROOT / 'configs' / 'simulation_genesis_igt_task_models.yaml'
DECK_NAMES = ('A', 'B', 'C', 'D')
ModelFamily = Literal['valence', 'magnitude']
AttentionCondition = Literal['loss', 'win', 'balanced']


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse unified IGT simulation arguments.

    Args:
        argv (list[str] | None): Optional CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run all six GENESIS IGT model conditions.')
    parser.add_argument('--config', type=Path, default=DEFAULT_SIMULATION_CONFIG_PATH)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--n-jobs', type=int, default=-1)
    return parser.parse_args(argv)


def build_query(
    *,
    model_config: dict[str, str],
    time_embedding: torch.Tensor,
    reward_embeddings: dict[float, torch.Tensor],
    reward_extrema: dict[str, float],
) -> IGTQuery:
    """
    Build one temporal-plus-attention query for a configured IGT model.

    Args:
        model_config (dict[str, str]): Model family and attention condition.
        time_embedding (torch.Tensor): Current temporal representation with shape ``[D]``.
        reward_embeddings (dict[float, torch.Tensor]): Learned embeddings keyed by scaled payoff.
        reward_extrema (dict[str, float]): Scaled negative and positive schedule extrema.

    Returns:
        IGTQuery: Named query components and their stored episode keys.
    """
    family = model_config['family']
    attention = model_config['attention']
    episode_key = 'valence_embedding' if family == 'valence' else 'reward_embedding'
    negative_embedding = (
        VALENCE_EMBEDDINGS['negative']
        if family == 'valence'
        else reward_embeddings[reward_extrema['negative']]
    )
    positive_embedding = (
        VALENCE_EMBEDDINGS['positive']
        if family == 'valence'
        else reward_embeddings[reward_extrema['positive']]
    )
    components = {
        'time_embedding': IGTQueryComponent(
            embedding=time_embedding,
            episode_key='time_embedding',
        ),
    }
    if attention in {'loss', 'balanced'}:
        components['negative_attention'] = IGTQueryComponent(
            embedding=negative_embedding,
            episode_key=episode_key,
        )
    if attention in {'win', 'balanced'}:
        components['positive_attention'] = IGTQueryComponent(
            embedding=positive_embedding,
            episode_key=episode_key,
        )
    return IGTQuery(components=components)


def build_component_weights(*, attention: AttentionCondition, temporal_weight: float) -> dict[str, float]:
    """
    Resolve temporal and attention weights for one model condition.

    Args:
        attention (AttentionCondition): Loss, win, or balanced attention condition.
        temporal_weight (float): Temporal similarity contribution.

    Returns:
        dict[str, float]: Named component weights summing to one.
    """
    attention_weight = 1.0 - temporal_weight
    weights = {'time_embedding': temporal_weight}
    if attention == 'loss':
        weights['negative_attention'] = attention_weight
    elif attention == 'win':
        weights['positive_attention'] = attention_weight
    else:
        weights['negative_attention'] = attention_weight / 2.0
        weights['positive_attention'] = attention_weight / 2.0
    return weights


def run_agent(
    *,
    config: dict[str, Any],
    model_config: dict[str, str],
    schedule: np.ndarray,
    temporal_embeddings: torch.Tensor,
    reward_embeddings: dict[float, torch.Tensor],
    reward_extrema: dict[str, float],
    reward_decoder: torch.nn.Module,
    agent_id: int,
) -> dict[str, Any]:
    """
    Run one seeded agent for one unified IGT model condition.

    Args:
        config (dict[str, Any]): Resolved temporal-weight cell configuration.
        model_config (dict[str, str]): Model name, family, and attention condition.
        schedule (np.ndarray): Scaled payoff schedule with shape ``[N, 4]``.
        temporal_embeddings (torch.Tensor): Shared temporal trajectory with shape ``[N, D]``.
        reward_embeddings (dict[float, torch.Tensor]): Learned embeddings keyed by payoff.
        reward_extrema (dict[str, float]): Scaled negative and positive schedule extrema.
        reward_decoder (torch.nn.Module): Frozen decoder from reward embeddings to scalars.
        agent_id (int): Stable agent identifier shared across conditions.

    Returns:
        dict[str, Any]: Agent seeds and complete trial history.
    """
    base_seed = config['base_random_seed'] + agent_id * 2
    retrieval_random_state = np.random.RandomState(base_seed)
    choice_random_state = np.random.RandomState(base_seed + 1)
    sampling_config = config['episodic_sampling']
    sampler = EpisodicSamplingModuleIowaGamblingTask(
        reward_decoder=reward_decoder,
        k_samples=sampling_config['sampling_budget'],
        sampling_temperature=config['retrieval_temperature'],
        query_key_normalization=sampling_config['query_key_normalization'],
        device=next(reward_decoder.parameters(), torch.empty(0)).device,
    )
    policy = IowaGamblingTaskPolicy(**config['choice_policy'])
    component_weights = build_component_weights(
        attention=model_config['attention'],
        temporal_weight=config['temporal_weight'],
    )
    deck_draw_indices = dict.fromkeys(DECK_NAMES, 0)
    cumulative_reward = 0.0
    history = []

    for trial_id in range(config['task']['n_trials']):
        query = build_query(
            model_config=model_config,
            time_embedding=temporal_embeddings[trial_id],
            reward_embeddings=reward_embeddings,
            reward_extrema=reward_extrema,
        )
        action_values, retrieval_metadata = sampler.estimate_action_values(
            query=query,
            weights=component_weights,
            random_state=retrieval_random_state,
            return_metadata=True,
        )
        policy_output = policy.choose_deck(
            action_values=action_values,
            trial_id=trial_id,
            random_state=choice_random_state,
        )
        chosen_deck = policy_output['chosen_deck']
        deck_index = DECK_NAMES.index(chosen_deck)
        draw_index = deck_draw_indices[chosen_deck]
        reward = float(schedule[draw_index, deck_index])
        deck_draw_indices[chosen_deck] += 1
        cumulative_reward += reward
        sampler.store_episode(
            chosen_deck=chosen_deck,
            time_embedding=temporal_embeddings[trial_id],
            valence_embedding=get_valence_embedding(reward=reward),
            reward_embedding=reward_embeddings[reward],
            reward=reward,
            trial_metadata={'trial_id': trial_id, 'draw_index': draw_index},
        )
        history.append({
            'trial_id': trial_id,
            'chosen_deck': chosen_deck,
            'deck_draw_index': draw_index,
            'reward': reward,
            'cumulative_reward': cumulative_reward,
            'action_values': dict(action_values),
            'p_a': policy_output['p_a'],
            'p_b': policy_output['p_b'],
            'p_c': policy_output['p_c'],
            'p_d': policy_output['p_d'],
            'choice_temperature': policy_output['choice_temperature'],
            'model_name': model_config['name'],
            'model_family': model_config['family'],
            'attention_condition': model_config['attention'],
            'temporal_weight': config['temporal_weight'],
            'attention_weight': 1.0 - config['temporal_weight'],
            'component_weights': dict(component_weights),
            'query_key_normalization': sampling_config['query_key_normalization'],
            'retrieval_temperature': config['retrieval_temperature'],
            'previous_experience_counts': {
                deck: retrieval_metadata[deck]['available_memory_count']
                for deck in DECK_NAMES
            },
            'retrieved_episode_indices': {
                deck: retrieval_metadata[deck]['sampled_episode_indices']
                for deck in DECK_NAMES
            },
            'retrieval_metadata': retrieval_metadata,
        })

    return {
        'agent_id': agent_id,
        'retrieval_seed': base_seed,
        'choice_seed': base_seed + 1,
        'simulation_log': history,
    }


def run_temporal_weight_cell(
    *,
    config_index: int,
    temporal_weight: float,
    config: dict[str, Any],
    model_config: dict[str, str],
    schedule: np.ndarray,
    temporal_embeddings: torch.Tensor,
    reward_embeddings: dict[float, torch.Tensor],
    reward_extrema: dict[str, float],
    reward_decoder: torch.nn.Module,
    output_dir: Path,
) -> None:
    """
    Run one temporal-weight cell with serial agent execution.

    Args:
        config_index (int): One-based temporal-weight cell index.
        temporal_weight (float): Temporal query-key contribution.
        config (dict[str, Any]): Complete shared simulation configuration.
        model_config (dict[str, str]): Active model condition.
        schedule (np.ndarray): Scaled payoff schedule with shape ``[N, 4]``.
        temporal_embeddings (torch.Tensor): Shared temporal trajectory with shape ``[N, D]``.
        reward_embeddings (dict[float, torch.Tensor]): Learned embeddings keyed by payoff.
        reward_extrema (dict[str, float]): Scaled negative and positive schedule extrema.
        reward_decoder (torch.nn.Module): Frozen reward decoder.
        output_dir (Path): Model-specific raw-record directory.
    """
    output_path = output_dir / config_to_filename(
        temporal_weight=temporal_weight,
        config_index=config_index,
    )
    if output_path.exists():
        return

    resolved_config = dict(config, temporal_weight=temporal_weight)
    agent_runs = [
        run_agent(
            config=resolved_config,
            model_config=model_config,
            schedule=schedule,
            temporal_embeddings=temporal_embeddings,
            reward_embeddings=reward_embeddings,
            reward_extrema=reward_extrema,
            reward_decoder=reward_decoder,
            agent_id=agent_id,
        )
        for agent_id in range(config['n_agents'])
    ]
    np.save(output_path, {
        'config': resolved_config,
        'model': dict(model_config),
        'temporal_weight': temporal_weight,
        'attention_weight': 1.0 - temporal_weight,
        'reward_extrema': dict(reward_extrema),
        'retrieval_temperature': resolved_config['retrieval_temperature'],
        'source_schedule_path': str(filepaths.ROOT / config['task']['schedule_path']),
        'reward_scaling_factor': config['task']['reward_scaling_factor'],
        'temporal_seed': config['temporal']['random_seed'],
        'agent_runs': agent_runs,
    }, allow_pickle=True)


def run_model(
    *,
    config: dict[str, Any],
    model_config: dict[str, str],
    schedule: np.ndarray,
    temporal_embeddings: torch.Tensor,
    reward_embeddings: dict[float, torch.Tensor],
    reward_extrema: dict[str, float],
    reward_decoder: torch.nn.Module,
    parallel: bool,
    n_jobs: int,
) -> None:
    """
    Complete all temporal-weight cells for one model condition.

    Args:
        config (dict[str, Any]): Complete shared simulation configuration.
        model_config (dict[str, str]): Active model condition.
        schedule (np.ndarray): Scaled payoff schedule with shape ``[N, 4]``.
        temporal_embeddings (torch.Tensor): Shared temporal trajectory with shape ``[N, D]``.
        reward_embeddings (dict[float, torch.Tensor]): Learned embeddings keyed by payoff.
        reward_extrema (dict[str, float]): Scaled negative and positive schedule extrema.
        reward_decoder (torch.nn.Module): Frozen reward decoder.
        parallel (bool): Whether to parallelize temporal-weight cells.
        n_jobs (int): Number of cell workers.
    """
    output_dir = filepaths.EXPERIMENTS_DIR / config['experiment_name'] / model_config['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_args = [
        dict(
            config_index=config_index,
            temporal_weight=temporal_weight,
            config=config,
            model_config=model_config,
            schedule=schedule,
            temporal_embeddings=temporal_embeddings,
            reward_embeddings=reward_embeddings,
            reward_extrema=reward_extrema,
            reward_decoder=reward_decoder,
            output_dir=output_dir,
        )
        for config_index, temporal_weight in enumerate(config['grid']['temporal_weight'], start=1)
    ]
    if parallel:
        Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(run_temporal_weight_cell)(**cell_kwargs)
            for cell_kwargs in cell_args
        )
        return
    for cell_kwargs in cell_args:
        run_temporal_weight_cell(**cell_kwargs)


def main(args: argparse.Namespace) -> None:
    """
    Run all configured IGT models sequentially.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    config = load_simulation_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    schedule = load_igt_schedule(
        schedule_path=filepaths.ROOT / config['task']['schedule_path'],
        reward_scaling_factor=config['task']['reward_scaling_factor'],
    )
    reward_model = load_linear_decoder(checkpoint_path=IGT_REWARD_MODEL_PATH, device=device)
    reward_embeddings = build_reward_embedding_cache(
        schedule=schedule,
        reward_model=reward_model,
        device=device,
    )
    reward_extrema = {
        'negative': float(schedule.min()),
        'positive': float(schedule.max()),
    }
    temporal_embeddings = build_temporal_embeddings(config)
    for model_config in config['models']:
        run_model(
            config=config,
            model_config=model_config,
            schedule=schedule,
            temporal_embeddings=temporal_embeddings,
            reward_embeddings=reward_embeddings,
            reward_extrema=reward_extrema,
            reward_decoder=reward_model.decoder,
            parallel=args.parallel,
            n_jobs=args.n_jobs,
        )


if __name__ == '__main__':
    main(parse_args())
