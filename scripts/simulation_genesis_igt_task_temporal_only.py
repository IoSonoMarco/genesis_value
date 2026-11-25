"""
Simulate temporal-only GENESIS episodic retrieval in the Iowa Gambling Task.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path.cwd()))

from models import load_linear_decoder
from models.agent import IowaGamblingTaskPolicy
from models.memory import EpisodicSamplingModuleIowaGamblingTask, IGTQuery
from models.memory.utils import generate_temporal_embeddings
from utils import filepaths
from utils.simulation_config import load_simulation_config


DEFAULT_SIMULATION_CONFIG_PATH = filepaths.ROOT / 'configs' / 'simulation_genesis_igt_task_temporal_only.yaml'
IGT_REWARD_MODEL_PATH = filepaths.OUTPUT_DIR / 'reward_linear_decoder' / 'igt' / 'weights.pt'
DECK_NAMES = ('A', 'B', 'C', 'D')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse temporal-only IGT simulation arguments.

    Args:
        argv (list[str] | None): Optional CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run GENESIS temporal-only IGT simulations.')
    parser.add_argument('--config', type=Path, default=DEFAULT_SIMULATION_CONFIG_PATH)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--n-jobs', type=int, default=-1)
    return parser.parse_args(argv)


def load_igt_schedule(*, schedule_path: Path, reward_scaling_factor: float) -> np.ndarray:
    """
    Load and scale the headerless A/B/C/D IGT payoff schedule.

    Args:
        schedule_path (Path): Source payoff CSV.
        reward_scaling_factor (float): Scalar applied to every payoff.

    Returns:
        np.ndarray: Scaled schedule with shape ``[100, 4]``.
    """
    return pd.read_csv(schedule_path, header=None).to_numpy(dtype=np.float32) * reward_scaling_factor


@torch.no_grad()
def build_reward_embedding_cache(
    *,
    schedule: np.ndarray,
    reward_model: torch.nn.Module,
    device: torch.device,
) -> dict[float, torch.Tensor]:
    """
    Encode each unique scaled IGT payoff once.

    Args:
        schedule (np.ndarray): Scaled IGT schedule with shape ``[N, 4]``.
        reward_model (torch.nn.Module): Frozen IGT reward encoder/decoder.
        device (torch.device): Inference device.

    Returns:
        dict[float, torch.Tensor]: Reward embeddings keyed by scalar payoff.
    """
    reward_embeddings = {}
    for reward in np.unique(schedule):
        embedding = reward_model.encode(torch.tensor([reward], dtype=torch.float32, device=device))[0].cpu()
        reward_embeddings[float(reward)] = embedding
    return reward_embeddings


def build_temporal_embeddings(config: dict[str, Any]) -> torch.Tensor:
    """
    Generate the shared temporal trajectory for all IGT agents and conditions.

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


def run_agent(
    *,
    config: dict[str, Any],
    schedule: np.ndarray,
    temporal_embeddings: torch.Tensor,
    reward_embeddings: dict[float, torch.Tensor],
    reward_decoder: torch.nn.Module,
    agent_id: int,
) -> dict[str, Any]:
    """
    Run one stochastic temporal-only IGT agent.

    Args:
        config (dict[str, Any]): Resolved grid-cell configuration.
        schedule (np.ndarray): Scaled IGT schedule with shape ``[N, 4]``.
        temporal_embeddings (torch.Tensor): Shared trajectory with shape ``[N, D]``.
        reward_embeddings (dict[float, torch.Tensor]): Cached payoff representations.
        reward_decoder (torch.nn.Module): Frozen decoder from reward embeddings to scalars.
        agent_id (int): Stable agent identifier shared across grid cells.

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
        device=next(reward_decoder.parameters()).device,
    )
    policy = IowaGamblingTaskPolicy(**config['choice_policy'])
    deck_draw_indices = dict.fromkeys(DECK_NAMES, 0)
    cumulative_reward = 0.0
    history = []

    for trial_id in range(config['task']['n_trials']):
        action_values, retrieval_metadata = sampler.estimate_action_values(
            query=IGTQuery(time_embedding=temporal_embeddings[trial_id]),
            weights={'time_embedding': 1.0},
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
            valence_embedding=None,
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
            'retrieval_temperature': config['retrieval_temperature'],
            'previous_experience_counts': {
                deck: retrieval_metadata[deck]['available_memory_count']
                for deck in DECK_NAMES
            },
            'retrieved_episode_indices': {
                deck: retrieval_metadata[deck]['sampled_episode_indices']
                for deck in DECK_NAMES
            },
        })

    return {
        'agent_id': agent_id,
        'retrieval_seed': base_seed,
        'choice_seed': base_seed + 1,
        'simulation_log': history,
    }


def config_to_filename(*, retrieval_temperature: float, config_index: int) -> str:
    """
    Build one retrieval-temperature-specific result filename.

    Args:
        retrieval_temperature (float): Grid-cell retrieval temperature.
        config_index (int): One-based grid-cell index.

    Returns:
        str: Grid-cell output filename.
    """
    temperature = str(retrieval_temperature).replace('.', 'p')
    return f'{config_index:03d}_retrieval_temperature-{temperature}.npy'


def run_retrieval_temperature_cell(
    *,
    config_index: int,
    retrieval_temperature: float,
    config: dict[str, Any],
    schedule: np.ndarray,
    temporal_embeddings: torch.Tensor,
    reward_embeddings: dict[float, torch.Tensor],
    reward_decoder: torch.nn.Module,
    output_dir: Path,
) -> None:
    """
    Run one retrieval-temperature cell with serial agent execution.

    Args:
        config_index (int): One-based retrieval-temperature cell index.
        retrieval_temperature (float): Retrieval sampling temperature for the cell.
        config (dict[str, Any]): Complete simulation configuration.
        schedule (np.ndarray): Scaled IGT schedule with shape ``[N, 4]``.
        temporal_embeddings (torch.Tensor): Shared trajectory with shape ``[N, D]``.
        reward_embeddings (dict[float, torch.Tensor]): Cached payoff representations.
        reward_decoder (torch.nn.Module): Frozen decoder from reward embeddings to scalars.
        output_dir (Path): Directory for grid-cell outputs.
    """
    output_path = output_dir / config_to_filename(
        retrieval_temperature=retrieval_temperature,
        config_index=config_index,
    )
    if output_path.exists():
        return
    resolved_config = dict(config, retrieval_temperature=retrieval_temperature)
    agent_runs = [
        run_agent(
            config=resolved_config,
            schedule=schedule,
            temporal_embeddings=temporal_embeddings,
            reward_embeddings=reward_embeddings,
            reward_decoder=reward_decoder,
            agent_id=agent_id,
        )
        for agent_id in range(config['n_agents'])
    ]
    np.save(output_path, {
        'config': resolved_config,
        'retrieval_temperature': retrieval_temperature,
        'source_schedule_path': str(filepaths.ROOT / config['task']['schedule_path']),
        'reward_scaling_factor': config['task']['reward_scaling_factor'],
        'temporal_seed': config['temporal']['random_seed'],
        'agent_runs': agent_runs,
    }, allow_pickle=True)


def main(args: argparse.Namespace) -> None:
    """
    Run the complete temporal-only IGT retrieval-temperature grid.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    config = load_simulation_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    schedule_path = filepaths.ROOT / config['task']['schedule_path']
    schedule = load_igt_schedule(
        schedule_path=schedule_path,
        reward_scaling_factor=config['task']['reward_scaling_factor'],
    )
    reward_model = load_linear_decoder(checkpoint_path=IGT_REWARD_MODEL_PATH, device=device)
    reward_embeddings = build_reward_embedding_cache(
        schedule=schedule,
        reward_model=reward_model,
        device=device,
    )
    temporal_embeddings = build_temporal_embeddings(config)
    output_dir = filepaths.EXPERIMENTS_DIR / config['experiment_name']
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_args = [
        dict(
            config_index=config_index,
            retrieval_temperature=retrieval_temperature,
            config=config,
            schedule=schedule,
            temporal_embeddings=temporal_embeddings,
            reward_embeddings=reward_embeddings,
            reward_decoder=reward_model.decoder,
            output_dir=output_dir,
        )
        for config_index, retrieval_temperature in enumerate(config['grid']['retrieval_temperature'], start=1)
    ]
    if args.parallel:
        Parallel(n_jobs=args.n_jobs, prefer='threads')(
            delayed(run_retrieval_temperature_cell)(**cell_kwargs)
            for cell_kwargs in cell_args
        )
        return

    for cell_kwargs in cell_args:
        run_retrieval_temperature_cell(**cell_kwargs)


if __name__ == '__main__':
    main(parse_args())
