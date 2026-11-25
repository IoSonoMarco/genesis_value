"""
Generate factorized GENESIS episode embeddings for figure construction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import torch

from configs import shape3d_feature_config_genesis
from data import get_Shapes3D_datasets
from models import AEModelConfig, CVAEFilmMultipleFeatures, LinearDecoder, load_linear_decoder
from models.memory.utils import generate_temporal_embeddings
from scripts.simulation_genesis_feature_recombination import (
    materialize_session_batch,
    sample_balanced_session,
)
from utils import constants, filepaths
from utils.simulation_config import load_simulation_config


FACTOR_COLUMNS = (
    'object_shape',
    'object_color',
    'floor_color',
    'wall_color',
)
STIMULUS_COMPONENT_NAMES = (
    'z',
    'object_shape',
    'object_color',
    'floor_color',
    'wall_color',
)
DEFAULT_CONFIG_PATH = filepaths.ROOT / 'configs' / 'generate_figure_episode_embeddings.yaml'
MAX_REWARD_ASSIGNMENT_RESTARTS = 100
MAX_REWARD_ASSIGNMENT_SWAPS = 10_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse episode-embedding generation arguments.

    Args:
        argv (list[str] | None): Optional command-line arguments.

    Returns:
        argparse.Namespace: Parsed generation arguments.
    """
    parser = argparse.ArgumentParser(description='Generate factorized GENESIS episode embeddings.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)

    return parser.parse_args(argv)


def load_stimulus_model(
    *,
    stimulus_model_dir: Path,
    feature_config: dict[str, list[int]],
    device: torch.device,
) -> CVAEFilmMultipleFeatures:
    """
    Load the frozen Shape3D stimulus CVAE.

    Args:
        stimulus_model_dir (Path): Trained CVAE output directory.
        feature_config (dict[str, list[int]]): Ordered Shape3D feature levels.
        device (torch.device): Inference device.

    Returns:
        CVAEFilmMultipleFeatures: Frozen stimulus encoder.
    """
    model_parameters = np.load(stimulus_model_dir / 'model_params.npy', allow_pickle=True).item()
    model_weights = torch.load(stimulus_model_dir / 'ckpt' / 'best.pt', map_location='cpu')
    model_config = AEModelConfig(
        latent_dim=model_parameters['latent_dim'],
        hidden_dim_base=model_parameters['hidden_dim_base'],
        m=3,
        kld_beta=model_parameters['kld_beta'],
    )
    model = CVAEFilmMultipleFeatures(
        config=model_config,
        feature_sizes={
            feature_name: len(feature_values)
            for feature_name, feature_values in feature_config.items()
        },
        n_cond_channels=model_parameters['n_cond_channels'],
        image_size=constants.SHAPE3D_IMAGE_SIZE,
    )
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


def encode_stimulus_components(
    *,
    stimulus_model: CVAEFilmMultipleFeatures,
    pixel_values: torch.Tensor,
    factors: dict[str, torch.Tensor],
    feature_config: dict[str, list[int]],
) -> dict[str, torch.Tensor]:
    """
    Encode posterior-mean and feature-specific stimulus components.

    Args:
        stimulus_model (CVAEFilmMultipleFeatures): Frozen stimulus encoder.
        pixel_values (torch.Tensor): Images with shape ``[N, C, H, W]``.
        factors (dict[str, torch.Tensor]): Feature IDs with shape ``[N]``.
        feature_config (dict[str, list[int]]): Ordered conditional-feature mapping.

    Returns:
        dict[str, torch.Tensor]: Factorized stimulus embeddings with shape ``[N, D]``.
    """
    mu, _ = stimulus_model.encode(pixel_values)
    conditional_embeddings = stimulus_model.embed_features(factors).flatten(1)
    components = {'z': mu.flatten(1)}

    for feature_index, feature_name in enumerate(feature_config):
        start = feature_index * stimulus_model.cond_dim
        stop = start + stimulus_model.cond_dim
        components[feature_name] = conditional_embeddings[:, start:stop]

    return components


def get_reward_balance_score(
    *,
    session: pd.DataFrame,
    rewards: np.ndarray,
    reward_values: np.ndarray,
) -> int:
    """
    Measure departures from per-feature-level reward balance.

    Args:
        session (pd.DataFrame): Ordered sampled Shape3D session.
        rewards (np.ndarray): Candidate reward allocation with shape ``[N]``.
        reward_values (np.ndarray): Discrete reward values with shape ``[R]``.

    Returns:
        int: Total count outside floor-or-ceiling level allocation bounds.
    """
    score = 0
    for factor_name in FACTOR_COLUMNS:
        for factor_value in session[factor_name].unique():
            level_rewards = rewards[session[factor_name].to_numpy() == factor_value]
            lower_count, remainder = divmod(len(level_rewards), len(reward_values))
            upper_count = lower_count + int(remainder > 0)

            for reward_value in reward_values:
                count = int(np.count_nonzero(level_rewards == reward_value))
                if count < lower_count:
                    score += lower_count - count
                if count > upper_count:
                    score += count - upper_count

    return score


def assign_balanced_rewards(
    *,
    session: pd.DataFrame,
    reward_values: np.ndarray,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """
    Assign equal reward counts without feature-level reward associations.

    Args:
        session (pd.DataFrame): Ordered sampled Shape3D session.
        reward_values (np.ndarray): Discrete scalar rewards with shape ``[R]``.
        random_state (np.random.RandomState): Seeded allocation random source.

    Returns:
        np.ndarray: Balanced scalar reward allocation with shape ``[N]``.
    """
    repeats, remainder = divmod(len(session), len(reward_values))
    assert remainder == 0, 'n_samples must allocate every reward equally'
    reward_template = np.repeat(reward_values, repeats)

    for _ in range(MAX_REWARD_ASSIGNMENT_RESTARTS):
        rewards = random_state.permutation(reward_template)
        score = get_reward_balance_score(
            session=session,
            rewards=rewards,
            reward_values=reward_values,
        )
        if score == 0:
            return rewards

        for _ in range(MAX_REWARD_ASSIGNMENT_SWAPS):
            first_index, second_index = random_state.choice(len(rewards), size=2, replace=False)
            if rewards[first_index] == rewards[second_index]:
                continue

            candidate_rewards = rewards.copy()
            candidate_rewards[first_index], candidate_rewards[second_index] = (
                candidate_rewards[second_index],
                candidate_rewards[first_index],
            )
            candidate_score = get_reward_balance_score(
                session=session,
                rewards=candidate_rewards,
                reward_values=reward_values,
            )
            if candidate_score <= score:
                rewards = candidate_rewards
                score = candidate_score
                if score == 0:
                    return rewards

    raise RuntimeError('failed to balance reward assignments across Shape3D factors')


@torch.no_grad()
def build_reward_embedding_cache(
    *,
    reward_values: np.ndarray,
    reward_model: LinearDecoder,
    device: torch.device,
) -> dict[float, torch.Tensor]:
    """
    Encode each unique scalar reward once.

    Args:
        reward_values (np.ndarray): Discrete scalar rewards with shape ``[R]``.
        reward_model (LinearDecoder): Frozen IGT reward encoder.
        device (torch.device): Inference device.

    Returns:
        dict[float, torch.Tensor]: Reward embeddings keyed by scalar reward.
    """
    reward_embeddings = {}
    for reward_value in np.unique(reward_values):
        reward_tensor = torch.tensor([reward_value], dtype=torch.float32, device=device)
        reward_embeddings[float(reward_value)] = reward_model.encode(reward_tensor)[0]

    return reward_embeddings


def build_temporal_embeddings(
    *,
    n_samples: int,
    embedding_dim: int,
    random_seed: int,
) -> np.ndarray:
    """
    Generate ordered temporal embeddings with a pi-over-N angular step.

    Args:
        n_samples (int): Number of temporal embeddings.
        embedding_dim (int): Temporal embedding width.
        random_seed (int): Seed for the temporal trajectory.

    Returns:
        np.ndarray: Temporal embeddings with shape ``[N, D]``.
    """
    return generate_temporal_embeddings(
        [np.pi / n_samples] * n_samples,
        dim=embedding_dim,
        random_state=np.random.RandomState(random_seed),
    )


def build_episode_records(
    *,
    session: pd.DataFrame,
    stimulus_embeddings: dict[str, torch.Tensor],
    session_rewards: np.ndarray,
    reward_embedding_cache: dict[float, torch.Tensor],
    temporal_embeddings: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Assemble ordered factorized episode records.

    Args:
        session (pd.DataFrame): Ordered sampled Shape3D feature indices with shape ``[N, 4]``.
        stimulus_embeddings (dict[str, torch.Tensor]): Stimulus components with shape ``[N, D]``.
        session_rewards (np.ndarray): Scalar reward sequence with shape ``[N]``.
        reward_embedding_cache (dict[float, torch.Tensor]): Cached reward embeddings.
        temporal_embeddings (np.ndarray): Temporal embeddings with shape ``[N, D]``.

    Returns:
        list[dict[str, Any]]: Ordered serializable episode records.
    """
    records = []
    for index, reward_value in enumerate(session_rewards):
        session_row = session.iloc[index]
        records.append({
            'stimulus': {
                component_name: stimulus_embeddings[component_name][index].cpu().numpy()
                for component_name in STIMULUS_COMPONENT_NAMES
            },
            'reward': {
                'reward_embedding': reward_embedding_cache[float(reward_value)].cpu().numpy(),
            },
            'time': np.asarray(temporal_embeddings[index]),
            'metadata': {
                'scalar_reward': float(reward_value),
                **{
                    feature_name: int(session_row[feature_name])
                    for feature_name in FACTOR_COLUMNS
                },
            },
        })

    return records


def run_generation(args: argparse.Namespace) -> Path:
    """
    Generate and save one ordered list of factorized episodes.

    Args:
        args (argparse.Namespace): Parsed generation arguments.

    Returns:
        Path: Saved NumPy artifact path.
    """
    config = load_simulation_config(args.config)
    feature_config = shape3d_feature_config_genesis.feature_config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_dataset, _ = get_Shapes3D_datasets(
        file_path=filepaths.SHAPES3D_FILE_PATH,
        **feature_config,
    )
    session = sample_balanced_session(
        data_table=test_dataset.data_table,
        session_size=config['n_samples'],
        session_seed=config['random_seed'],
    )
    reward_values = np.asarray(config['reward_values'], dtype=np.float32)
    session_rewards = assign_balanced_rewards(
        session=session,
        reward_values=reward_values,
        random_state=np.random.RandomState(config['random_seed']),
    )
    session = session.copy()
    session['scalar_reward'] = session_rewards
    batch = materialize_session_batch(session=session, dataset=test_dataset)
    stimulus_model = load_stimulus_model(
        stimulus_model_dir=filepaths.ROOT / config['stimulus_model_path'],
        feature_config=feature_config,
        device=device,
    )
    reward_model = load_linear_decoder(
        checkpoint_path=filepaths.ROOT / config['igt_reward_checkpoint_path'],
        device=device,
    )

    for parameter in reward_model.parameters():
        parameter.requires_grad = False

    with torch.no_grad():
        stimulus_embeddings = encode_stimulus_components(
            stimulus_model=stimulus_model,
            pixel_values=batch.pixel_values.to(device),
            factors={
                feature_name: getattr(batch.factors, feature_name).to(device)
                for feature_name in feature_config
            },
            feature_config=feature_config,
        )
        reward_embedding_cache = build_reward_embedding_cache(
            reward_values=reward_values,
            reward_model=reward_model,
            device=device,
        )

    temporal_embeddings = build_temporal_embeddings(
        n_samples=config['n_samples'],
        embedding_dim=config['temporal_embedding_dim'],
        random_seed=config['random_seed'],
    )
    records = build_episode_records(
        session=session,
        stimulus_embeddings=stimulus_embeddings,
        session_rewards=session_rewards,
        reward_embedding_cache=reward_embedding_cache,
        temporal_embeddings=temporal_embeddings,
    )
    output_dir = filepaths.OUTPUT_DIR / config['output_subdirectory']
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / config['output_filename']
    np.save(output_path, records, allow_pickle=True)
    print(f"Saved {config['experiment_name']} to {output_path}")

    return output_path


def main() -> None:
    """
    Run episode-embedding generation from command-line arguments.
    """
    run_generation(parse_args())


if __name__ == '__main__':
    main()
