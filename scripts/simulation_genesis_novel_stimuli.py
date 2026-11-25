"""
Simulate GENESIS value reconstruction for unseen or seen target stimuli.
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
from joblib import Parallel, delayed

from configs import shape3d_feature_config_genesis
from data import get_Shapes3D_datasets
from data.datasets.shapes3d import Shape
from models import AEModelConfig, CVAEFilmMultipleFeatures, LinearDecoder, load_linear_decoder
from models.memory import EpisodicSamplingModuleFeatureRecombination
from models.memory.rag import CompressedEpisodeFactorized, FeatureRecombinationHiddenState
from utils import constants, filepaths
from utils.simulation_config import get_replacement_output_dir, load_simulation_config


DEVICE = torch.device('cuda')
FEATURE_CONFIG = shape3d_feature_config_genesis.feature_config
DEFAULT_CONFIG_PATH = filepaths.ROOT / 'configs' / 'simulation_genesis_novel_stimuli.yaml'
STIMULUS_MODEL_DIR = filepaths.OUTPUT_DIR / 'cvae' / 'shape3d_genesis_2'
REWARD_MODEL_PATH = (
    filepaths.OUTPUT_DIR / 'reward_linear_decoder' / 'shape3d_feature_debugging' / 'weights.pt'
)
COLOR_FACTORS = ('object_color', 'floor_color', 'wall_color')
STIMULUS_COMPONENTS = ('z', 'object_shape', 'object_color', 'floor_color', 'wall_color')
ACTIVE_COMPONENTS = ('z', 'object_color', 'floor_color', 'wall_color')
CONDITIONS = ('unseen', 'seen')


class CortexEncoder:
    """
    Encode Shape3D stimuli and scalar rewards with frozen cortical models.
    """

    def __init__(
        self,
        *,
        stimulus_model_dir: Path,
        reward_model_path: Path,
        device: torch.device,
    ) -> None:
        """
        Load and freeze the stimulus and reward models.

        Args:
            stimulus_model_dir (Path): Trained Shape3D CVAE directory.
            reward_model_path (Path): Trained Shape3D reward checkpoint.
            device (torch.device): Model execution device.
        """
        self.device = device
        model_parameters = np.load(
            stimulus_model_dir / 'model_params.npy',
            allow_pickle=True,
        ).item()
        model_config = AEModelConfig(
            latent_dim=model_parameters['latent_dim'],
            hidden_dim_base=model_parameters['hidden_dim_base'],
            m=3,
            kld_beta=model_parameters['kld_beta'],
        )
        self.stimulus_model = CVAEFilmMultipleFeatures(
            config=model_config,
            feature_sizes={
                feature_name: len(feature_values)
                for feature_name, feature_values in FEATURE_CONFIG.items()
            },
            n_cond_channels=model_parameters['n_cond_channels'],
            image_size=constants.SHAPE3D_IMAGE_SIZE,
        )
        model_weights = torch.load(
            stimulus_model_dir / 'ckpt' / 'best.pt',
            map_location='cpu',
        )
        self.stimulus_model.load_state_dict(model_weights)
        self.reward_model: LinearDecoder = load_linear_decoder(
            checkpoint_path=reward_model_path,
            device='cpu',
        )
        self.stimulus_model.to(device).eval()
        self.reward_model.to(device).eval()

        for model in (self.stimulus_model, self.reward_model):
            for parameter in model.parameters():
                parameter.requires_grad = False

    @torch.no_grad()
    def encode_stimuli(
        self,
        *,
        pixel_values: torch.Tensor,
        factors: object,
    ) -> dict[str, torch.Tensor]:
        """
        Encode a stimulus batch into image and factor embeddings.

        Args:
            pixel_values (torch.Tensor): Image batch with shape ``[N, C, H, W]``.
            factors (object): Batched Shapes3D factor container.

        Returns:
            dict[str, torch.Tensor]: Component embeddings with shape ``[N, D]``.
        """
        features = {
            feature_name: getattr(factors, feature_name).to(self.device)
            for feature_name in FEATURE_CONFIG
        }
        mu, _ = self.stimulus_model.encode(pixel_values.to(self.device))
        feature_embeddings = self.stimulus_model.embed_features(features).flatten(1).cpu()
        stimulus_embeddings = {'z': mu.flatten(1).cpu()}

        for feature_index, feature_name in enumerate(FEATURE_CONFIG):
            start = feature_index * self.stimulus_model.cond_dim
            stop = start + self.stimulus_model.cond_dim
            stimulus_embeddings[feature_name] = feature_embeddings[:, start:stop]

        return stimulus_embeddings

    @torch.no_grad()
    def encode_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Encode stored scalar rewards.

        Args:
            rewards (torch.Tensor): Scalar rewards with shape ``[N]``.

        Returns:
            torch.Tensor: Reward embeddings with shape ``[N, D]``.
        """
        return self.reward_model.encode(rewards.to(self.device)).cpu()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse novel-stimulus simulation arguments.

    Args:
        argv (list[str] | None): Optional command-line arguments.

    Returns:
        argparse.Namespace: Parsed simulation arguments.
    """
    parser = argparse.ArgumentParser(description='Simulate GENESIS retrieval for unseen or seen stimuli.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--condition', choices=CONDITIONS, default='unseen')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--n-jobs', type=int, default=-1)

    return parser.parse_args(argv)


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate the novel-stimulus experiment definition.

    Args:
        config (dict[str, Any]): Parsed experiment configuration.
    """
    if config['dataset']['n_color_levels'] != 5:
        raise ValueError('novel-stimulus simulation requires exactly five color levels')
    if config['dataset']['memory_images_per_conjunction'] != 1:
        raise ValueError('novel-stimulus simulation requires one memory image per conjunction')
    if len(config['reward_assignment']['level_values']) != 5:
        raise ValueError('reward level_values must contain exactly five entries')
    if config['reward_assignment']['aggregation'] != 'mean':
        raise ValueError('only mean reward aggregation is supported')
    if config['dataset']['object_shape'] != 'cube':
        raise ValueError('novel-stimulus simulation requires cube stimuli')
    if not config['sample_replacement']:
        raise ValueError('novel-stimulus simulation requires replacement sampling')

    weights = config['query_weights']
    if any(weight < 0.0 for weight in weights.values()):
        raise ValueError('query weights must be nonnegative')
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError('query weights must sum to one')
    if tuple(name for name, weight in weights.items() if weight > 0.0) != ACTIVE_COMPONENTS:
        raise ValueError(f'active query components must be {ACTIVE_COMPONENTS}')

    budgets = config['grid']['sampling_budget']
    if any(not isinstance(budget, int) or budget <= 0 for budget in budgets):
        raise ValueError('sampling budgets must be positive integers')


def build_coordinate_seed(base_seed: int, *coordinates: int) -> int:
    """
    Derive an independent deterministic seed from experiment coordinates.

    Args:
        base_seed (int): Experiment seed.
        *coordinates (int): Session or budget coordinates.

    Returns:
        int: Derived unsigned seed.
    """
    return int(np.random.SeedSequence([base_seed, *coordinates]).generate_state(1)[0])


def conjunction_id(object_color: int, floor_color: int, wall_color: int) -> int:
    """
    Convert a local color tuple into its stable base-five identifier.

    Args:
        object_color (int): Local object-color ID.
        floor_color (int): Local floor-color ID.
        wall_color (int): Local wall-color ID.

    Returns:
        int: Identifier in the range 0 through 124.
    """
    return object_color * 25 + floor_color * 5 + wall_color


def build_session_dataset(
    *,
    data_table: pd.DataFrame,
    cube_local_id: int,
    level_values: np.ndarray,
    session_id: int,
    session_seed: int,
    condition: str = 'unseen',
) -> pd.DataFrame:
    """
    Select one image per cube conjunction and designate one query target.

    Args:
        data_table (pd.DataFrame): Held-out Shapes3D metadata.
        cube_local_id (int): Dataset-local cube identifier.
        level_values (np.ndarray): Reward value for each local color level.
        session_id (int): Independent simulation identifier.
        session_seed (int): Dataset-generation seed.
        condition (str): Episodic-memory inclusion condition.

    Returns:
        pd.DataFrame: Complete 125-stimulus session table.
    """
    cube_table = data_table.loc[data_table['object_shape'] == cube_local_id].copy()
    cube_table['dataset_index'] = cube_table.index
    conjunction_counts = cube_table.groupby(list(COLOR_FACTORS)).size()
    if len(conjunction_counts) != 125 or (conjunction_counts == 0).any():
        raise ValueError('held-out Shapes3D partition does not contain all 125 cube conjunctions')

    random_state = np.random.RandomState(session_seed)
    selected_rows = []
    for color_tuple in sorted(conjunction_counts.index.tolist()):
        candidates = cube_table.loc[
            (cube_table['object_color'] == color_tuple[0])
            & (cube_table['floor_color'] == color_tuple[1])
            & (cube_table['wall_color'] == color_tuple[2])
        ].sort_values('h5_idx')
        selected_index = int(random_state.randint(len(candidates)))
        selected_rows.append(candidates.iloc[selected_index].to_dict())

    session = pd.DataFrame(selected_rows)
    session['conjunction_id'] = [
        conjunction_id(int(row.object_color), int(row.floor_color), int(row.wall_color))
        for row in session.itertuples()
    ]
    target_conjunction_id = int(random_state.choice(session['conjunction_id'].to_numpy()))
    session['is_target'] = session['conjunction_id'] == target_conjunction_id
    session['role'] = np.where(session['is_target'], 'target', 'memory')
    session['in_memory'] = ~session['is_target'] if condition == 'unseen' else True
    session['memory_position'] = np.nan
    session.loc[session['in_memory'], 'memory_position'] = np.arange(session['in_memory'].sum(), dtype=float)
    session['session_id'] = session_id
    session['session_seed'] = session_seed

    for factor_name in COLOR_FACTORS:
        session[f'{factor_name}_value'] = level_values[session[factor_name].to_numpy(dtype=int)]
    session['oracle_reward'] = session[
        [f'{factor_name}_value' for factor_name in COLOR_FACTORS]
    ].mean(axis=1)
    session = session.sort_values('conjunction_id').reset_index(drop=True)
    session['dataset_index'] = session['dataset_index'].astype(int)
    session['h5_idx'] = session['h5_idx'].astype(int)

    validate_session_dataset(session=session, level_values=level_values, condition=condition)

    return session


def validate_session_dataset(
    *,
    session: pd.DataFrame,
    level_values: np.ndarray,
    condition: str = 'unseen',
) -> None:
    """
    Validate one complete novel-stimulus dataset.

    Args:
        session (pd.DataFrame): Complete session table.
        level_values (np.ndarray): Configured color-level rewards.
        condition (str): Episodic-memory inclusion condition.
    """
    memory = session.loc[session['in_memory']]
    target = session.loc[session['is_target']]
    triples = session[list(COLOR_FACTORS)].apply(tuple, axis=1)
    expected_memory_count = 124 if condition == 'unseen' else 125

    if condition not in CONDITIONS:
        raise ValueError(f'condition must be one of {CONDITIONS}')
    if len(session) != 125 or len(memory) != expected_memory_count or len(target) != 1:
        raise ValueError(f'session must contain {expected_memory_count} memories and one target')
    if session['conjunction_id'].nunique() != 125 or triples.nunique() != 125:
        raise ValueError('session must contain every color conjunction exactly once')
    if session['h5_idx'].nunique() != 125:
        raise ValueError('selected HDF5 indices must be unique within a session')
    target_occurrences = int((memory['conjunction_id'] == int(target['conjunction_id'].iloc[0])).sum())
    expected_target_occurrences = 0 if condition == 'unseen' else 1
    if target_occurrences != expected_target_occurrences:
        raise ValueError('target memory membership does not match the condition')
    if memory['memory_position'].tolist() != list(range(expected_memory_count)):
        raise ValueError('memory positions must be consecutive and aligned with memory order')
    if any(memory[factor_name].nunique() != 5 for factor_name in COLOR_FACTORS):
        raise ValueError('all color levels must remain represented in memory')

    contribution_columns = [f'{factor_name}_value' for factor_name in COLOR_FACTORS]
    if any(not session[column].isin(level_values).all() for column in contribution_columns):
        raise ValueError('reward contributions must belong to level_values')
    expected_rewards = session[contribution_columns].mean(axis=1)
    if not np.allclose(session['oracle_reward'], expected_rewards):
        raise ValueError('oracle rewards must equal the mean color contribution')
    if not np.isfinite(session['oracle_reward']).all():
        raise ValueError('oracle rewards must be finite')
    if not session['oracle_reward'].between(level_values.min(), level_values.max()).all():
        raise ValueError('oracle rewards lie outside the configured range')


def materialize_stimulus_batch(*, session: pd.DataFrame, dataset: Any) -> tuple[torch.Tensor, object]:
    """
    Load the selected session images and factors in conjunction order.

    Args:
        session (pd.DataFrame): Ordered session table.
        dataset (Any): Held-out Shapes3D dataset.

    Returns:
        tuple[torch.Tensor, object]: Image batch and factor container.
    """
    items = [dataset[int(dataset_index)] for dataset_index in session['dataset_index']]
    pixel_values, factor_values, _ = zip(*items)
    factor_type = type(factor_values[0])
    factors = factor_type(**{
        feature_name: torch.stack([
            getattr(factor_value, feature_name)
            for factor_value in factor_values
        ])
        for feature_name in FEATURE_CONFIG
    })

    return torch.stack(pixel_values), factors


def build_hidden_states(
    *,
    stimulus_embeddings: dict[str, torch.Tensor],
    reward_embeddings: torch.Tensor,
    memory_indices: np.ndarray,
) -> list[FeatureRecombinationHiddenState]:
    """
    Build ordered factorized hidden states for the memory stimuli.

    Args:
        stimulus_embeddings (dict[str, torch.Tensor]): Complete-session stimulus embeddings.
        reward_embeddings (torch.Tensor): Memory-only reward embeddings.
        memory_indices (np.ndarray): Complete-session indices assigned to memory.

    Returns:
        list[FeatureRecombinationHiddenState]: Ordered RAG storage inputs.
    """
    return [
        {
            'stimulus_embeddings': {
                component_name: stimulus_embeddings[component_name][session_index]
                for component_name in STIMULUS_COMPONENTS
            },
            'reward_embedding': reward_embeddings[memory_position],
        }
        for memory_position, session_index in enumerate(memory_indices)
    ]


def build_target_query(
    *,
    stimulus_embeddings: dict[str, torch.Tensor],
    target_index: int,
) -> CompressedEpisodeFactorized:
    """
    Build a reward-free and time-free factorized target query.

    Args:
        stimulus_embeddings (dict[str, torch.Tensor]): Complete-session stimulus embeddings.
        target_index (int): Complete-session target position.

    Returns:
        CompressedEpisodeFactorized: Reward-free target stimulus query.
    """
    return CompressedEpisodeFactorized(
        z_embedding=stimulus_embeddings['z'][target_index],
        object_shape_embedding=stimulus_embeddings['object_shape'][target_index],
        object_color_embedding=stimulus_embeddings['object_color'][target_index],
        floor_color_embedding=stimulus_embeddings['floor_color'][target_index],
        wall_color_embedding=stimulus_embeddings['wall_color'][target_index],
    )


@torch.no_grad()
def run_budget(
    *,
    sampling_budget: int,
    budget_index: int,
    session_id: int,
    config: dict[str, Any],
    sampling_probabilities: np.ndarray,
    episodic_sampler: EpisodicSamplingModuleFeatureRecombination,
) -> dict[str, Any]:
    """
    Sample, combine, and batch-decode one retrieval-budget cell.

    Args:
        sampling_budget (int): Number of episodes per reconstruction.
        budget_index (int): Budget coordinate used for seeding.
        session_id (int): Independent simulation identifier.
        config (dict[str, Any]): Resolved experiment configuration.
        sampling_probabilities (np.ndarray): Fixed candidate probabilities with shape ``[N]``.
        episodic_sampler (EpisodicSamplingModuleFeatureRecombination): Stored session memory.

    Returns:
        dict[str, Any]: Budget seed and decoded reconstructions.
    """
    retrieval_seed = build_coordinate_seed(config['random_seed'], session_id, budget_index)
    random_state = np.random.RandomState(retrieval_seed)
    sampled_indices = random_state.choice(
        len(episodic_sampler.memory),
        size=(config['n_retrieval_repetitions'], sampling_budget),
        replace=True,
        p=sampling_probabilities,
    )
    reward_embeddings = torch.stack([
        episode.reward_embedding
        for episode in episodic_sampler.memory
    ])
    combined_embeddings = reward_embeddings[torch.as_tensor(sampled_indices)].mean(dim=1)
    decoded_rewards = episodic_sampler.reward_decoder(
        combined_embeddings.to(episodic_sampler.device),
    ).flatten().detach().cpu().numpy()

    if decoded_rewards.shape != (config['n_retrieval_repetitions'],):
        raise ValueError('decoded reward vector has an unexpected shape')
    if not np.isfinite(decoded_rewards).all():
        raise ValueError('decoded rewards must be finite')

    return {
        'sampling_budget': sampling_budget,
        'retrieval_seed': retrieval_seed,
        'n_retrieval_repetitions': config['n_retrieval_repetitions'],
        'decoded_reconstructed_rewards': decoded_rewards,
    }


def validate_retrieval_distribution(
    *,
    distribution: dict[str, Any],
    query_weights: dict[str, float],
    memory_count: int,
) -> None:
    """
    Validate target-query candidates and probabilities.

    Args:
        distribution (dict[str, Any]): Cached RAG retrieval distribution.
        query_weights (dict[str, float]): Configured component weights.
        memory_count (int): Expected number of retrieval candidates.
    """
    if distribution['active_component_names'] != list(ACTIVE_COMPONENTS):
        raise ValueError(f'active retrieval components must be {ACTIVE_COMPONENTS}')
    if not np.isclose(sum(query_weights.values()), 1.0):
        raise ValueError('query weights must sum to one')

    probabilities = distribution['sampling_probabilities']
    if probabilities.shape != (memory_count,):
        raise ValueError(f'target retrieval distribution must contain {memory_count} candidates')
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError('sampling probabilities must be finite and nonnegative')
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError('sampling probabilities must sum to one')


def build_exact_match_diagnostics(
    *,
    target_row: pd.Series,
    distribution: dict[str, Any],
    sampling_budgets: list[int],
) -> dict[str, Any]:
    """
    Summarize exact-target accessibility in the retrieval distribution.

    Args:
        target_row (pd.Series): Query-target dataset row.
        distribution (dict[str, Any]): Query-specific retrieval distribution.
        sampling_budgets (list[int]): Configured episodic sampling budgets.

    Returns:
        dict[str, Any]: Exact-match identity, score, probability, and rank.
    """
    target_in_memory = bool(target_row['in_memory'])
    diagnostics = {
        'target_in_memory': target_in_memory,
        'memory_position': None,
        'conjunction_id': int(target_row['conjunction_id']),
        'h5_idx': int(target_row['h5_idx']),
        'oracle_reward': float(target_row['oracle_reward']),
        'query_key_score': None,
        'sampling_probability': None,
        'sampling_probability_rank': None,
        'is_highest_probability': False,
        'retrieval_probability_at_least_once': {},
    }
    if not target_in_memory:
        return diagnostics

    target_memory_position = int(target_row['memory_position'])
    probabilities = np.asarray(distribution['sampling_probabilities'])
    scores = np.asarray(distribution['query_key_match_scores'])
    order = np.lexsort((np.arange(len(probabilities)), -probabilities))
    ranks = np.empty(len(probabilities), dtype=int)
    ranks[order] = np.arange(1, len(probabilities) + 1)
    target_probability = float(probabilities[target_memory_position])
    diagnostics.update({
        'memory_position': target_memory_position,
        'query_key_score': float(scores[target_memory_position]),
        'sampling_probability': target_probability,
        'sampling_probability_rank': int(ranks[target_memory_position]),
        'is_highest_probability': bool(ranks[target_memory_position] == 1),
        'retrieval_probability_at_least_once': {
            int(sampling_budget): float(1.0 - (1.0 - target_probability) ** sampling_budget)
            for sampling_budget in sampling_budgets
        },
    })

    return diagnostics


def run_session(
    *,
    session_id: int,
    config: dict[str, Any],
    test_dataset: Any,
    cortex_encoder: CortexEncoder,
    condition: str,
    parallel: bool,
    n_jobs: int,
) -> dict[str, Any]:
    """
    Generate, encode, retrieve, and validate one independent session.

    Args:
        session_id (int): Independent simulation identifier.
        config (dict[str, Any]): Resolved experiment configuration.
        test_dataset (Any): Held-out Shapes3D dataset.
        cortex_encoder (CortexEncoder): Frozen stimulus and reward models.
        condition (str): Episodic-memory inclusion condition.
        parallel (bool): Whether to evaluate budget cells in parallel.
        n_jobs (int): Joblib worker count.

    Returns:
        dict[str, Any]: Compact schema-v2 simulation record.
    """
    session_seed = build_coordinate_seed(config['random_seed'], session_id)
    cube_local_id = test_dataset.global_to_local['object_shape'][int(Shape.CUBE)]
    level_values = np.asarray(config['reward_assignment']['level_values'], dtype=float)
    session = build_session_dataset(
        data_table=test_dataset.data_table,
        cube_local_id=cube_local_id,
        level_values=level_values,
        session_id=session_id,
        session_seed=session_seed,
        condition=condition,
    )
    pixel_values, factors = materialize_stimulus_batch(session=session, dataset=test_dataset)
    stimulus_embeddings = cortex_encoder.encode_stimuli(
        pixel_values=pixel_values,
        factors=factors,
    )
    memory_indices = np.flatnonzero(session['in_memory'].to_numpy())
    target_index = int(np.flatnonzero(session['is_target'].to_numpy())[0])
    memory_count = 124 if condition == 'unseen' else 125
    memory_rewards = torch.tensor(
        session.loc[session['in_memory'], 'oracle_reward'].to_numpy(),
        dtype=torch.float32,
    )
    reward_embeddings = cortex_encoder.encode_rewards(memory_rewards)
    hidden_states = build_hidden_states(
        stimulus_embeddings=stimulus_embeddings,
        reward_embeddings=reward_embeddings,
        memory_indices=memory_indices,
    )
    episodic_sampler = EpisodicSamplingModuleFeatureRecombination(
        reward_decoder=cortex_encoder.reward_model.decoder,
        sample_replacement=config['sample_replacement'],
        query_key_normalization=config['query_key_normalization'],
        device=cortex_encoder.device,
    )
    episodic_sampler.store_session(hidden_states)
    if len(episodic_sampler.memory) != memory_count:
        raise ValueError(f'episodic memory must contain exactly {memory_count} episodes')
    if any(
        getattr(episode, f'{component_name}_embedding') is None
        for episode in episodic_sampler.memory
        for component_name in STIMULUS_COMPONENTS
    ):
        raise ValueError('stored episode is missing a stimulus component')
    if any(episode.reward_embedding is None for episode in episodic_sampler.memory):
        raise ValueError('stored episode is missing a reward embedding')

    query = build_target_query(
        stimulus_embeddings=stimulus_embeddings,
        target_index=target_index,
    )
    distribution = episodic_sampler.get_retrieval_distribution(
        query=query,
        weights=config['query_weights'],
        sampling_temperature=config['sampling_temperature'],
    )
    validate_retrieval_distribution(
        distribution=distribution,
        query_weights=config['query_weights'],
        memory_count=memory_count,
    )

    budget_kwargs = [
        {
            'sampling_budget': sampling_budget,
            'budget_index': budget_index,
            'session_id': session_id,
            'config': config,
            'sampling_probabilities': distribution['sampling_probabilities'],
            'episodic_sampler': episodic_sampler,
        }
        for budget_index, sampling_budget in enumerate(config['grid']['sampling_budget'])
    ]
    if parallel:
        retrieval_grid = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(run_budget)(**kwargs)
            for kwargs in budget_kwargs
        )
    else:
        retrieval_grid = [run_budget(**kwargs) for kwargs in budget_kwargs]

    target_row = session.iloc[target_index]
    memory_table = session.iloc[memory_indices]
    target_image_occurrences = int((memory_table['h5_idx'] == int(target_row['h5_idx'])).sum())
    expected_target_occurrences = 0 if condition == 'unseen' else 1
    if target_image_occurrences != expected_target_occurrences:
        raise ValueError('target HDF5 image membership does not match the condition')
    target_feature_counts = {
        factor_name: int((memory_table[factor_name] == target_row[factor_name]).sum())
        for factor_name in COLOR_FACTORS
    }
    expected_feature_count = 24 if condition == 'unseen' else 25
    if any(count != expected_feature_count for count in target_feature_counts.values()):
        raise ValueError('target feature counts do not match the condition')
    memory_embeddings = {
        component_name: stimulus_embeddings[component_name][memory_indices].numpy()
        for component_name in ACTIVE_COMPONENTS
    }
    target_embeddings = {
        component_name: stimulus_embeddings[component_name][target_index].numpy()
        for component_name in ACTIVE_COMPONENTS
    }
    target_metadata = {
        key: value
        for key, value in target_row.to_dict().items()
        if key not in ('role', 'memory_position')
    }
    target_in_memory = bool(target_row['in_memory'])
    exact_match = build_exact_match_diagnostics(
        target_row=target_row,
        distribution=distribution,
        sampling_budgets=config['grid']['sampling_budget'],
    )
    target_record = {
        **target_metadata,
        'active_embeddings': target_embeddings,
        'active_component_names': distribution['active_component_names'],
        'component_weights': distribution['component_weights'],
        'raw_component_similarities': distribution['raw_component_similarities'],
        'normalized_component_similarities': distribution['normalized_component_similarities'],
        'query_key_scores': distribution['query_key_match_scores'],
        'sampling_probabilities': distribution['sampling_probabilities'],
        'exact_match': exact_match,
    }

    return {
        'schema_version': 2,
        'condition': condition,
        'config': config,
        'session': {
            'session_id': session_id,
            'session_seed': session_seed,
            'target_conjunction_id': int(target_row['conjunction_id']),
            'target_h5_idx': int(target_row['h5_idx']),
            'target_oracle_reward': float(target_row['oracle_reward']),
            'condition': condition,
            'memory_count': memory_count,
            'target_count': 1,
            'target_in_memory': target_in_memory,
            'target_feature_memory_counts': target_feature_counts,
        },
        'dataset': session.to_dict(orient='records'),
        'memory_embeddings': memory_embeddings,
        'target': target_record,
        'retrieval_grid': retrieval_grid,
    }


def resolve_output_dir(*, config: dict[str, Any], condition: str) -> Path:
    """
    Resolve the condition-specific simulation directory.

    Args:
        config (dict[str, Any]): Resolved experiment configuration.
        condition (str): Episodic-memory inclusion condition.

    Returns:
        Path: Condition-specific simulation directory.
    """
    if condition == 'unseen':
        root_dir = filepaths.EXPERIMENTS_DIR
        output_name = config['experiment_name']
    else:
        root_dir = filepaths.EXPERIMENTS_DIR / config['experiment_name']
        output_name = condition

    return get_replacement_output_dir(
        root_dir=root_dir,
        output_name=output_name,
        sample_replacement=config['sample_replacement'],
    )


def save_record(
    *,
    record: dict[str, Any],
    config: dict[str, Any],
    condition: str,
) -> Path:
    """
    Save one compact session record.

    Args:
        record (dict[str, Any]): Schema-v2 session record.
        config (dict[str, Any]): Resolved experiment configuration.
        condition (str): Episodic-memory inclusion condition.

    Returns:
        Path: Saved record path.
    """
    output_dir = resolve_output_dir(config=config, condition=condition)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"001_session-{record['session']['session_id']:03d}.npy"
    np.save(output_path, record, allow_pickle=True)

    return output_path


def run_simulation(args: argparse.Namespace) -> None:
    """
    Run all configured novel-stimulus sessions.

    Args:
        args (argparse.Namespace): Parsed simulation arguments.
    """
    config = load_simulation_config(args.config)
    validate_config(config)
    _, test_dataset, _ = get_Shapes3D_datasets(
        file_path=filepaths.SHAPES3D_FILE_PATH,
        **FEATURE_CONFIG,
    )
    cortex_encoder = CortexEncoder(
        stimulus_model_dir=STIMULUS_MODEL_DIR,
        reward_model_path=REWARD_MODEL_PATH,
        device=DEVICE,
    )

    for session_id in range(config['n_sessions']):
        record = run_session(
            session_id=session_id,
            config=config,
            test_dataset=test_dataset,
            cortex_encoder=cortex_encoder,
            condition=args.condition,
            parallel=args.parallel,
            n_jobs=args.n_jobs,
        )
        output_path = save_record(record=record, config=config, condition=args.condition)
        print(f'Saved session {session_id:03d} to {output_path}')


def main() -> None:
    run_simulation(parse_args())


if __name__ == '__main__':
    main()
