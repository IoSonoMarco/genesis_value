"""
Prepare feature-rich Shape3D sessions for GENESIS reward recombination.
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import torch

from configs import shape3d_feature_config_genesis
from data import Shapes3DInput, get_Shapes3D_datasets
from joblib import Parallel, delayed
from models import (
    AEModelConfig,
    CVAEFilmMultipleFeatures,
    LinearDecoder,
    load_linear_decoder,
)
from models.memory import EpisodicSamplingModuleFeatureRecombination
from models.memory.rag import CompressedEpisodeFactorized, FeatureRecombinationHiddenState
from utils import constants, filepaths
from utils.simulation_config import get_replacement_output_dir, load_simulation_config


SESSION_METADATA_COLUMNS = (
    'session_position',
    'dataset_index',
    'h5_idx',
    'object_shape',
    'object_color',
    'floor_color',
    'wall_color',
    'object_shape_global',
    'object_color_global',
    'floor_color_global',
    'wall_color_global',
    'scalar_reward',
)
FACTOR_COLUMNS = (
    'object_shape',
    'object_color',
    'floor_color',
    'wall_color',
)
COLOR_FACTOR_COLUMNS = (
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
STORED_COMPONENT_FIELDS = {
    'z': 'z_embedding',
    'object_shape': 'object_shape_embedding',
    'object_color': 'object_color_embedding',
    'floor_color': 'floor_color_embedding',
    'wall_color': 'wall_color_embedding',
    'reward': 'reward_embedding',
}
DEVICE = torch.device('cuda')
FEATURE_CONFIG = shape3d_feature_config_genesis.feature_config
DEFAULT_SIMULATION_CONFIG_PATH = filepaths.ROOT / 'configs' / 'simulation_genesis_reward_reconstruction_ablation.yaml'
CUBE_OBJECT_SHAPE = 0
MAX_REWARD_ASSIGNMENT_ATTEMPTS = 100_000
RETRIEVAL_DISTRIBUTION_FIELDS = (
    'candidate_indices',
    'active_component_names',
    'raw_component_similarities',
    'normalized_component_similarities',
    'query_key_match_scores',
    'sampling_probabilities',
)


class CortexEncoder:
    """
    Encode Shape3D sessions and scalar rewards with frozen cortical models.
    """

    def __init__(
        self,
        *,
        stimulus_model_dir: Path,
        reward_model_dir: Path,
        device: torch.device,
    ) -> None:
        self.device = device
        self.feature_config = FEATURE_CONFIG
        self.stimulus_model = self.load_stimulus_model(stimulus_model_dir).to(device)
        self.reward_model = self.load_reward_model(reward_model_dir).to(device)

    def load_stimulus_model(self, stimulus_model_dir: Path) -> CVAEFilmMultipleFeatures:
        """
        Load the frozen Shape3D CVAE.
        """
        model_pars = np.load(
            stimulus_model_dir / 'model_params.npy',
            allow_pickle=True,
        ).item()
        model_weights = torch.load(
            stimulus_model_dir / 'ckpt' / 'best.pt',
            map_location='cpu',
        )
        config = AEModelConfig(
            latent_dim=model_pars['latent_dim'],
            hidden_dim_base=model_pars['hidden_dim_base'],
            m=3,
            kld_beta=model_pars['kld_beta'],
        )
        model = CVAEFilmMultipleFeatures(
            config=config,
            feature_sizes={
                feature_name: len(feature_values)
                for feature_name, feature_values in self.feature_config.items()
            },
            n_cond_channels=model_pars['n_cond_channels'],
            image_size=constants.SHAPE3D_IMAGE_SIZE,
        )
        model.load_state_dict(model_weights)
        model.eval()

        for parameter in model.parameters():
            parameter.requires_grad = False

        return model

    def load_reward_model(self, reward_model_dir: Path) -> LinearDecoder:
        """
        Load the frozen scalar reward encoder and decoder.
        """
        model = load_linear_decoder(
            checkpoint_path=next(reward_model_dir.rglob('weights.pt')),
            device='cpu',
        )

        for parameter in model.parameters():
            parameter.requires_grad = False

        return model

    @torch.no_grad()
    def encode_session(
        self,
        *,
        pixel_values: torch.Tensor,
        factors: object,
        rewards: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Encode one session batch into factorized stimulus and reward embeddings.
        """
        pixel_values = pixel_values.to(self.device)
        features = {
            feature_name: getattr(factors, feature_name).to(self.device)
            for feature_name in self.feature_config
        }
        mu, _ = self.stimulus_model.encode(pixel_values)
        feature_embeddings = self.stimulus_model.embed_features(features).flatten(1).cpu()
        stimulus_embeddings = {'z': mu.flatten(1).cpu()}

        for feature_index, feature_name in enumerate(self.feature_config):
            start = feature_index * self.stimulus_model.cond_dim
            stop = start + self.stimulus_model.cond_dim
            stimulus_embeddings[feature_name] = feature_embeddings[:, start:stop]

        reward_embeddings = self.reward_model.encode(rewards.to(self.device)).cpu()

        return stimulus_embeddings, reward_embeddings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse feature-recombination simulation arguments.
    """
    parser = argparse.ArgumentParser(description='Prepare GENESIS feature-recombination sessions.')
    parser.add_argument('--config', type=Path, default=DEFAULT_SIMULATION_CONFIG_PATH)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--n-jobs', type=int, default=-1)

    return parser.parse_args(argv)


def resolve_simulation_config(args: argparse.Namespace) -> dict[str, Any]:
    """
    Load one feature-recombination simulation definition.
    """
    return load_simulation_config(args.config)


def sample_balanced_session(
    *,
    data_table: pd.DataFrame,
    session_size: int,
    session_seed: int,
) -> pd.DataFrame:
    """
    Sample a session with non-cue factors balanced within each shape.
    """
    factor_values = {
        factor_name: np.sort(data_table[factor_name].unique())
        for factor_name in FACTOR_COLUMNS
    }
    if len(factor_values['object_shape']) != 4 or any(
        len(factor_values[factor_name]) != 5
        for factor_name in COLOR_FACTOR_COLUMNS
    ):
        raise ValueError('balanced sessions require four shapes and five values per color factor')

    random_state = np.random.RandomState(session_seed)
    selected_rows = []
    selected_h5_indices = set()
    shape_assignments = build_near_balanced_factor_assignments(
        factor_values=factor_values['object_shape'],
        session_size=session_size,
        random_state=random_state,
    )
    factor_assignments = {'object_shape': shape_assignments}

    for factor_name in COLOR_FACTOR_COLUMNS:
        assignments = np.empty(session_size, dtype=factor_values[factor_name].dtype)

        for shape_value in factor_values['object_shape']:
            shape_positions = np.flatnonzero(shape_assignments == shape_value)
            assignments[shape_positions] = build_near_balanced_factor_assignments(
                factor_values=factor_values[factor_name],
                session_size=len(shape_positions),
                random_state=random_state,
            )

        factor_assignments[factor_name] = assignments

    for position in range(session_size):
        target_factors = {
            factor_name: factor_assignments[factor_name][position]
            for factor_name in FACTOR_COLUMNS
        }
        candidates = data_table.loc[
            np.logical_and.reduce([
                data_table[factor_name] == factor_value
                for factor_name, factor_value in target_factors.items()
            ])
        ].sort_values('h5_idx')
        candidates = candidates.loc[~candidates['h5_idx'].isin(selected_h5_indices)]
        selected_h5_idx = int(random_state.choice(candidates['h5_idx'].to_numpy()))
        selected_row = data_table.loc[data_table['h5_idx'] == selected_h5_idx].iloc[0]

        selected_rows.append({
            'dataset_index': int(data_table.index.get_loc(selected_row.name)),
            'h5_idx': selected_h5_idx,
            'object_shape': int(selected_row['object_shape']),
            'object_color': int(selected_row['object_color']),
            'floor_color': int(selected_row['floor_color']),
            'wall_color': int(selected_row['wall_color']),
            'object_shape_global': int(selected_row['object_shape_global']),
            'object_color_global': int(selected_row['object_color_global']),
            'floor_color_global': int(selected_row['floor_color_global']),
            'wall_color_global': int(selected_row['wall_color_global']),
            'scalar_reward': 0.0,
        })
        selected_h5_indices.add(selected_h5_idx)

    session = pd.DataFrame(selected_rows).iloc[random_state.permutation(session_size)].reset_index(drop=True)
    session.insert(0, 'session_position', np.arange(session_size))

    return session.loc[:, SESSION_METADATA_COLUMNS]


def assign_session_rewards(
    *,
    session: pd.DataFrame,
    reward_values: np.ndarray,
    cue_feature: str,
    cue_value: int,
    minimum_cue_mean_reward: float,
    random_state: np.random.RandomState,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Assign constrained Nicholas-Mattar rewards to one sampled session.

    Args:
        session (pd.DataFrame): Ordered session metadata.
        reward_values (np.ndarray): Discrete scalar rewards with shape ``[R]``.
        cue_feature (str): Feature defining the semantic retrieval target.
        cue_value (int): Local cue-feature value.
        minimum_cue_mean_reward (float): Minimum accepted cue-conditioned mean.
        random_state (np.random.RandomState): Reward-allocation random source.

    Returns:
        tuple[pd.DataFrame, dict[str, Any]]: Rewarded session and assignment audit.

    Raises:
        RuntimeError: If no valid reward assignment is found.
    """
    repeats, remainder = divmod(len(session), len(reward_values))
    assert remainder == 0, 'session size must allocate every reward equally'
    reward_template = np.repeat(reward_values, repeats)

    for assignment_attempt in range(1, MAX_REWARD_ASSIGNMENT_ATTEMPTS + 1):
        rewarded_session = session.copy()
        rewarded_session['scalar_reward'] = random_state.permutation(reward_template)
        feature_reward_sums = {
            factor_name: {
                str(int(factor_value)): float(reward_sum)
                for factor_value, reward_sum in rewarded_session.groupby(factor_name)['scalar_reward'].sum().items()
            }
            for factor_name in FACTOR_COLUMNS
        }
        feature_sum_values = [
            reward_sum
            for factor_sums in feature_reward_sums.values()
            for reward_sum in factor_sums.values()
        ]
        cue_mean_reward = float(
            rewarded_session.loc[
                rewarded_session[cue_feature] == cue_value,
                'scalar_reward',
            ].mean(),
        )

        if (
            all(reward_sum != 0.0 for reward_sum in feature_sum_values)
            and any(reward_sum > 0.0 for reward_sum in feature_sum_values)
            and any(reward_sum < 0.0 for reward_sum in feature_sum_values)
            and cue_mean_reward >= minimum_cue_mean_reward
        ):
            return rewarded_session, {
                'assignment_attempts': assignment_attempt,
                'reward_values': reward_values.tolist(),
                'feature_reward_sums': feature_reward_sums,
                'cube_session_mean_reward': cue_mean_reward,
                'whole_session_mean_reward': float(rewarded_session['scalar_reward'].mean()),
            }

    raise RuntimeError('failed to assign constrained session rewards')


def build_near_balanced_factor_assignments(
    *,
    factor_values: np.ndarray,
    session_size: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """
    Allocate factor values with per-value counts differing by at most one.
    """
    base_count, remainder = divmod(session_size, len(factor_values))
    assignments = np.repeat(factor_values, base_count)

    if remainder:
        remainder_values = random_state.choice(
            factor_values,
            size=remainder,
            replace=False,
        )
        assignments = np.concatenate([assignments, remainder_values])

    return random_state.permutation(assignments)


def build_hidden_states(
    *,
    stimulus_embeddings: dict[str, Any],
    reward_embeddings: Any,
) -> list[dict[str, Any]]:
    """
    Materialize one ordered session as factorized RAG hidden states.
    """
    hidden_states = []

    for session_position in range(reward_embeddings.shape[0]):
        hidden_states.append({
            'stimulus_embeddings': {
                component_name: stimulus_embeddings[component_name][session_position]
                for component_name in STIMULUS_COMPONENT_NAMES
            },
            'reward_embedding': reward_embeddings[session_position],
        })

    return hidden_states


def build_cube_query(
    *,
    session: pd.DataFrame,
    episodic_sampler: EpisodicSamplingModuleFeatureRecombination,
) -> CompressedEpisodeFactorized:
    """
    Build the fixed cube cue from stored semantic and terminal temporal embeddings.
    """
    cube_position = int(session.loc[
        session['object_shape'] == CUBE_OBJECT_SHAPE,
        'session_position',
    ].iloc[0])

    return CompressedEpisodeFactorized(
        object_shape_embedding=episodic_sampler.memory[cube_position].object_shape_embedding,
        time_embedding=episodic_sampler.query_time_embedding,
    )


def build_retrieval_candidate_indices(
    *,
    session: pd.DataFrame,
    retrieval_scope: str,
    cue_feature: str,
    cue_value: int,
) -> list[int]:
    """
    Resolve eligible session positions for one retrieval scope.

    Args:
        session (pd.DataFrame): Ordered session metadata.
        retrieval_scope (str): All-memory or cue-feature retrieval scope.
        cue_feature (str): Feature used to define the retrieval cue.
        cue_value (int): Target value of the cue feature.

    Returns:
        list[int]: Eligible stored-memory indices.
    """
    if retrieval_scope == 'all_session_items':
        return session['session_position'].tolist()
    if retrieval_scope == 'cue_feature_items':
        return session.loc[session[cue_feature] == cue_value, 'session_position'].tolist()

    raise ValueError(f'unsupported retrieval scope: {retrieval_scope}')


def build_retrieval_weights(temporal_weight: float) -> dict[str, float]:
    """
    Build the fixed time-plus-cube component weights for one grid cell.
    """
    return {
        'time_embedding': temporal_weight,
        'z': 0.0,
        'object_shape': 1.0 - temporal_weight,
        'object_color': 0.0,
        'floor_color': 0.0,
        'wall_color': 0.0,
    }


def build_coordinate_seed(base_seed: int, *coordinates: int) -> int:
    """
    Derive an independent deterministic seed from experiment coordinates.
    """
    return int(np.random.SeedSequence([base_seed, *coordinates]).generate_state(1)[0])


def run_retrieval_grid_cell(
    *,
    config: dict[str, Any],
    query: CompressedEpisodeFactorized,
    episodic_sampler: EpisodicSamplingModuleFeatureRecombination,
    n_retrieval_repetitions: int,
    sampling_temperature: float,
    random_seed: int,
    session_id: int,
    grid_cell_id: int,
    candidate_indices: list[int],
) -> dict[str, Any]:
    """
    Run seeded Monte Carlo retrieval repetitions for one grid cell.
    """
    component_weights = build_retrieval_weights(config['temporal_weight'])
    repetitions = []
    retrieval_distribution = None

    for repetition_id in range(n_retrieval_repetitions):
        retrieval_seed = build_coordinate_seed(
            random_seed,
            session_id,
            grid_cell_id,
            repetition_id,
        )
        retrieval = episodic_sampler.retrieve_and_recombine_reward(
            query=query,
            weights=component_weights,
            k_samples=config['sampling_budget'],
            sampling_temperature=sampling_temperature,
            random_state=np.random.RandomState(retrieval_seed),
            candidate_indices=candidate_indices,
            return_metadata=True,
        )
        if retrieval_distribution is None:
            retrieval_distribution = {
                field_name: retrieval[field_name]
                for field_name in RETRIEVAL_DISTRIBUTION_FIELDS
            }
        compact_retrieval = {
            field_name: field_value
            for field_name, field_value in retrieval.items()
            if field_name not in RETRIEVAL_DISTRIBUTION_FIELDS
            and field_name not in ('combined_reward_embedding', 'component_weights', 'sampling_temperature')
        }
        repetitions.append({
            'repetition_id': repetition_id,
            'retrieval_seed': retrieval_seed,
            **compact_retrieval,
        })

    return {
        'config': dict(config),
        'component_weights': component_weights,
        'retrieval_distribution': retrieval_distribution,
        'repetitions': repetitions,
    }


def validate_stored_session(
    *,
    episodic_sampler: Any,
    expected_session_size: int,
) -> dict[str, Any]:
    """
    Confirm that one factorized RAG session has been stored completely.
    """
    if len(episodic_sampler.memory) != expected_session_size:
        raise ValueError('stored episode count does not match session size')

    for episode in episodic_sampler.memory:
        for field_name in STORED_COMPONENT_FIELDS.values():
            if getattr(episode, field_name) is None:
                raise ValueError(f'missing stored episode component: {field_name}')

    return {
        'stored_episode_count': len(episodic_sampler.memory),
        'stored_component_names': list(STORED_COMPONENT_FIELDS),
    }


def materialize_session_batch(
    *,
    session: pd.DataFrame,
    dataset: Any,
) -> Shapes3DInput:
    """
    Load one ordered session as a batched cortical-model input.
    """
    session_items = [dataset[int(dataset_index)] for dataset_index in session['dataset_index']]
    pixel_values, factor_values, _ = zip(*session_items)
    factor_type = type(factor_values[0])
    factors = factor_type(**{
        feature_name: torch.stack([
            getattr(factor_value, feature_name)
            for factor_value in factor_values
        ])
        for feature_name in FEATURE_CONFIG
    })

    return Shapes3DInput(
        pixel_values=torch.stack(pixel_values),
        factors=factors,
        rewards=torch.tensor(session['scalar_reward'].to_numpy(), dtype=torch.float32),
    )


def prepare_session_knowledge_base(
    *,
    session_id: int,
    session_seed: int,
    session_size: int,
    sample_replacement: bool,
    query_key_normalization: str,
    reward_assignment: dict[str, Any],
    test_ds: Any,
    cortex_encoder: CortexEncoder,
) -> tuple[
    dict[str, Any],
    EpisodicSamplingModuleFeatureRecombination,
    CompressedEpisodeFactorized,
]:
    """
    Build the fixed session, store it once, and construct the cube query.
    """
    session = sample_balanced_session(
        data_table=test_ds.data_table,
        session_size=session_size,
        session_seed=session_seed,
    )
    reward_seed = build_coordinate_seed(session_seed, 1)
    session, reward_audit = assign_session_rewards(
        session=session,
        reward_values=np.asarray(reward_assignment['values'], dtype=float),
        cue_feature=reward_assignment['cue_feature'],
        cue_value=reward_assignment['cue_value'],
        minimum_cue_mean_reward=reward_assignment['minimum_cue_mean_reward'],
        random_state=np.random.RandomState(reward_seed),
    )
    batch = materialize_session_batch(session=session, dataset=test_ds)
    stimulus_embeddings, reward_embeddings = cortex_encoder.encode_session(
        pixel_values=batch.pixel_values,
        factors=batch.factors,
        rewards=batch.rewards,
    )
    hidden_states: list[FeatureRecombinationHiddenState] = build_hidden_states(
        stimulus_embeddings=stimulus_embeddings,
        reward_embeddings=reward_embeddings,
    )
    episodic_sampler = EpisodicSamplingModuleFeatureRecombination(
        reward_decoder=cortex_encoder.reward_model.decoder,
        sample_replacement=sample_replacement,
        query_key_normalization=query_key_normalization,
        device=cortex_encoder.device,
    )
    episodic_sampler.store_session(hidden_states)
    validation = validate_stored_session(
        episodic_sampler=episodic_sampler,
        expected_session_size=session_size,
    )
    query = build_cube_query(
        session=session,
        episodic_sampler=episodic_sampler,
    )
    session_record = {
        'session_id': session_id,
        'session_seed': session_seed,
        'session_metadata': session.to_dict(orient='records'),
        'retrieval_candidate_indices': {
            'all_session_items': build_retrieval_candidate_indices(
                session=session,
                retrieval_scope='all_session_items',
                cue_feature=reward_assignment['cue_feature'],
                cue_value=reward_assignment['cue_value'],
            ),
            'cue_feature_items': build_retrieval_candidate_indices(
                session=session,
                retrieval_scope='cue_feature_items',
                cue_feature=reward_assignment['cue_feature'],
                cue_value=reward_assignment['cue_value'],
            ),
        },
        'reward_assignment': {
            'reward_seed': reward_seed,
            **reward_audit,
        },
        'reward_targets': {
            'cue_feature': reward_assignment['cue_feature'],
            'cue_value': reward_assignment['cue_value'],
            'minimum_cue_mean_reward': reward_assignment['minimum_cue_mean_reward'],
        },
        'storage_validation': validation,
        'query': {
            'object_shape': 'cube',
            'object_shape_local': CUBE_OBJECT_SHAPE,
        },
    }

    return session_record, episodic_sampler, query


def expand_retrieval_grid(simulation_config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Expand the configured retrieval grid into ordered condition records.
    """
    grid = simulation_config['grid']
    grid_keys = list(grid)

    return [
        dict(zip(grid_keys, values))
        for values in product(*[grid[grid_key] for grid_key in grid_keys])
    ]


def run_retrieval_grid(
    *,
    simulation_config: dict[str, Any],
    query: CompressedEpisodeFactorized,
    episodic_sampler: EpisodicSamplingModuleFeatureRecombination,
    session_id: int,
    candidate_indices_by_scope: dict[str, list[int]],
    parallel: bool,
    n_jobs: int,
) -> list[dict[str, Any]]:
    """
    Evaluate every retrieval-grid cell against one fixed stored session.
    """
    configs = expand_retrieval_grid(simulation_config)

    if parallel:
        return Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(run_retrieval_grid_cell)(
                config=config,
                query=query,
                episodic_sampler=episodic_sampler,
                n_retrieval_repetitions=simulation_config['n_retrieval_repetitions'],
                sampling_temperature=simulation_config['sampling_temperature'],
                random_seed=simulation_config['random_seed'],
                session_id=session_id,
                grid_cell_id=grid_cell_id,
                candidate_indices=candidate_indices_by_scope[
                    config.get('retrieval_scope', 'all_session_items')
                ],
            )
            for grid_cell_id, config in enumerate(configs)
        )

    return [
        run_retrieval_grid_cell(
            config=config,
            query=query,
            episodic_sampler=episodic_sampler,
            n_retrieval_repetitions=simulation_config['n_retrieval_repetitions'],
            sampling_temperature=simulation_config['sampling_temperature'],
            random_seed=simulation_config['random_seed'],
            session_id=session_id,
            grid_cell_id=grid_cell_id,
            candidate_indices=candidate_indices_by_scope[
                config.get('retrieval_scope', 'all_session_items')
            ],
        )
        for grid_cell_id, config in enumerate(configs)
    ]


def save_simulation_record(
    *,
    simulation_config: dict[str, Any],
    session: dict[str, Any],
    retrieval_grid: list[dict[str, Any]],
) -> None:
    """
    Save the session metadata and storage validation record.
    """
    savepath = get_replacement_output_dir(
        root_dir=filepaths.EXPERIMENTS_DIR,
        output_name=simulation_config['experiment_name'],
        sample_replacement=simulation_config['sample_replacement'],
    )
    savepath.mkdir(parents=True, exist_ok=True)
    record = {
        'schema_version': 3,
        'config': simulation_config,
        'session': session,
        'retrieval_grid': retrieval_grid,
    }
    np.save(
        savepath / f"001_session-{session['session_id']:03d}.npy",
        record,
        allow_pickle=True,
    )


def run_simulation(
    *,
    args: argparse.Namespace,
    parallel: bool,
) -> None:
    """
    Run one stored session and every configured retrieval-grid condition.
    """
    simulation_config = resolve_simulation_config(args)
    _, test_ds, _ = get_Shapes3D_datasets(
        file_path=filepaths.SHAPES3D_FILE_PATH,
        **FEATURE_CONFIG,
    )
    cortex_encoder = CortexEncoder(
        stimulus_model_dir=filepaths.OUTPUT_DIR / 'cvae' / 'shape3d_genesis_2',
        reward_model_dir=filepaths.OUTPUT_DIR / 'reward_linear_decoder',
        device=DEVICE,
    )
    session_sizes = simulation_config.get('session_sizes', [simulation_config['session_size']])

    for population_index, session_size in enumerate(session_sizes):
        for population_session_id in range(simulation_config['n_sessions']):
            session_id = population_index * simulation_config['n_sessions'] + population_session_id
            session_seed = build_coordinate_seed(simulation_config['random_seed'], session_id)
            session, episodic_sampler, query = prepare_session_knowledge_base(
                session_id=session_id,
                session_seed=session_seed,
                session_size=session_size,
                sample_replacement=simulation_config['sample_replacement'],
                query_key_normalization=simulation_config['query_key_normalization'],
                reward_assignment=simulation_config['reward_assignment'],
                test_ds=test_ds,
                cortex_encoder=cortex_encoder,
            )
            session['population_id'] = f'balanced_{session_size}'
            session['session_size'] = session_size
            session['cue_candidate_count'] = len(session['retrieval_candidate_indices']['cue_feature_items'])
            retrieval_grid = run_retrieval_grid(
                simulation_config=simulation_config,
                query=query,
                episodic_sampler=episodic_sampler,
                session_id=session_id,
                candidate_indices_by_scope=session['retrieval_candidate_indices'],
                parallel=parallel,
                n_jobs=args.n_jobs,
            )
            save_simulation_record(
                simulation_config=simulation_config,
                session=session,
                retrieval_grid=retrieval_grid,
            )


def run_simulation_grid(args: argparse.Namespace) -> None:
    """
    Run the temporary single-session feature-recombination scaffold.
    """
    run_simulation(args=args, parallel=False)


def run_simulation_grid_parallel(args: argparse.Namespace) -> None:
    """
    Run the temporary single-session scaffold through joblib threads.
    """
    run_simulation(args=args, parallel=True)


if __name__ == '__main__':
    parsed_args = parse_args()
    if parsed_args.parallel:
        run_simulation_grid_parallel(parsed_args)
    else:
        run_simulation_grid(parsed_args)
