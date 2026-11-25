"""
Run the GENESIS Nicholas-Mattar feature-choice simulation.
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
from models.memory import EpisodicSamplingModuleFeatureRecombination
from models.memory.rag import CompressedEpisodeFactorized
from scripts.simulation_genesis_feature_recombination import (
    CortexEncoder,
    FACTOR_COLUMNS,
    MAX_REWARD_ASSIGNMENT_ATTEMPTS,
    RETRIEVAL_DISTRIBUTION_FIELDS,
    build_coordinate_seed,
    build_hidden_states,
    expand_retrieval_grid,
    materialize_session_batch,
    sample_balanced_session,
    validate_stored_session,
)
from utils import filepaths
from utils.simulation_config import get_replacement_output_dir, load_simulation_config


DEVICE = torch.device('cuda')
FEATURE_CONFIG = shape3d_feature_config_genesis.feature_config
DEFAULT_SIMULATION_CONFIG_PATH = filepaths.ROOT / 'configs' / 'simulation_genesis_nicholas_mattar_task.yaml'
FEATURE_EMBEDDING_FIELDS = {
    'object_shape': 'object_shape_embedding',
    'object_color': 'object_color_embedding',
    'floor_color': 'floor_color_embedding',
    'wall_color': 'wall_color_embedding',
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse Nicholas-Mattar simulation arguments.
    """
    parser = argparse.ArgumentParser(description='Run GENESIS Nicholas-Mattar feature-choice sessions.')
    parser.add_argument('--config', type=Path, default=DEFAULT_SIMULATION_CONFIG_PATH)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--n-jobs', type=int, default=-1)

    return parser.parse_args(argv)


def resolve_simulation_config(args: argparse.Namespace) -> dict[str, Any]:
    """
    Load one Nicholas-Mattar simulation definition.
    """
    return load_simulation_config(args.config)


def build_target_mean_gap_schedule(
    *,
    n_sessions: int,
    target_mean_gap_range: list[float],
    required_oracle_choice: str,
) -> np.ndarray:
    """
    Build signed, evenly spaced target reward gaps for one oracle cohort.
    """
    target_gaps = np.linspace(*target_mean_gap_range, num=n_sessions)

    if required_oracle_choice == 'cube':
        return target_gaps

    return -target_gaps


def assign_target_rewards(
    *,
    session: pd.DataFrame,
    reward_values: np.ndarray,
    targets: list[dict[str, Any]],
    target_mean_gap: float,
    required_oracle_choice: str,
    random_state: np.random.RandomState,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Assign balanced baseline rewards calibrated to one target-feature gap.
    """
    repeats, remainder = divmod(len(session), len(reward_values))
    assert remainder == 0, 'session size must allocate every reward equally'
    reward_template = np.repeat(reward_values, repeats)

    requested_target_mean_gap = target_mean_gap

    for assignment_attempt in range(1, MAX_REWARD_ASSIGNMENT_ATTEMPTS + 1):
        rewarded_session = session.copy()
        rewarded_session['scalar_reward'] = random_state.permutation(reward_template)
        target_names = [target['name'] for target in targets]
        first_target, second_target = targets
        first_target_mask = rewarded_session[first_target['feature']] == first_target['value']
        second_target_mask = rewarded_session[second_target['feature']] == second_target['value']
        base_target_mean_gap = float(
            rewarded_session.loc[first_target_mask, 'scalar_reward'].mean()
            - rewarded_session.loc[second_target_mask, 'scalar_reward'].mean(),
        )
        reward_offset = (requested_target_mean_gap - base_target_mean_gap) / 2.0
        rewarded_session.loc[first_target_mask, 'scalar_reward'] += reward_offset
        rewarded_session.loc[second_target_mask, 'scalar_reward'] -= reward_offset

        target_mean_rewards = {
            target['name']: float(rewarded_session.loc[
                rewarded_session[target['feature']] == target['value'],
                'scalar_reward',
            ].mean())
            for target in targets
        }
        target_mean_gap = target_mean_rewards[target_names[0]] - target_mean_rewards[target_names[1]]
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

        oracle_choice = target_names[0] if target_mean_gap > 0.0 else target_names[1]

        if (
            all(reward_sum != 0.0 for reward_sum in feature_sum_values)
            and oracle_choice == required_oracle_choice
        ):
            return rewarded_session, {
                'assignment_attempts': assignment_attempt,
                'reward_values': reward_values.tolist(),
                'feature_reward_sums': feature_reward_sums,
                'target_mean_rewards': target_mean_rewards,
                'requested_target_mean_gap': requested_target_mean_gap,
                'target_mean_gap': target_mean_gap,
                'oracle_choice': oracle_choice,
                'whole_session_mean_reward': float(rewarded_session['scalar_reward'].mean()),
            }

    raise RuntimeError('failed to assign target-feature rewards')


def build_feature_query(
    *,
    session: pd.DataFrame,
    episodic_sampler: EpisodicSamplingModuleFeatureRecombination,
    target: dict[str, Any],
) -> CompressedEpisodeFactorized:
    """
    Build one semantic feature query with terminal temporal context.
    """
    session_position = int(session.loc[
        session[target['feature']] == target['value'],
        'session_position',
    ].iloc[0])
    embedding_field = FEATURE_EMBEDDING_FIELDS[target['feature']]

    return CompressedEpisodeFactorized(
        **{
            embedding_field: getattr(episodic_sampler.memory[session_position], embedding_field),
            'time_embedding': episodic_sampler.query_time_embedding,
        },
    )


def build_retrieval_weights(*, target: dict[str, Any], temporal_weight: float) -> dict[str, float]:
    """
    Build semantic-plus-temporal retrieval weights for one target cue.
    """
    weights = {
        'time_embedding': temporal_weight,
        'z': 0.0,
        'object_shape': 0.0,
        'object_color': 0.0,
        'floor_color': 0.0,
        'wall_color': 0.0,
    }
    weights[target['feature']] = 1.0 - temporal_weight

    return weights


def compact_retrieval(retrieval: dict[str, Any]) -> dict[str, Any]:
    """
    Remove repeated distribution arrays and latent tensors from one retrieval record.
    """
    return {
        field_name: field_value
        for field_name, field_value in retrieval.items()
        if field_name not in RETRIEVAL_DISTRIBUTION_FIELDS
        and field_name not in ('combined_reward_embedding', 'component_weights', 'sampling_temperature')
    }


def run_choice_grid_cell(
    *,
    config: dict[str, Any],
    queries: dict[str, CompressedEpisodeFactorized],
    targets: list[dict[str, Any]],
    episodic_sampler: EpisodicSamplingModuleFeatureRecombination,
    n_retrieval_repetitions: int,
    sampling_temperature: float,
    random_seed: int,
    session_id: int,
    grid_cell_id: int,
    candidate_indices: list[int],
    oracle_choice: str,
) -> dict[str, Any]:
    """
    Retrieve two feature rewards and choose the larger reconstruction.
    """
    repetitions = []
    retrieval_distributions = {}
    component_weights = {
        target['name']: build_retrieval_weights(target=target, temporal_weight=config['temporal_weight'])
        for target in targets
    }

    for repetition_id in range(n_retrieval_repetitions):
        target_retrievals = {}

        for target_index, target in enumerate(targets):
            target_name = target['name']
            retrieval_seed = build_coordinate_seed(
                random_seed,
                session_id,
                grid_cell_id,
                repetition_id,
                target_index,
            )
            retrieval = episodic_sampler.retrieve_and_recombine_reward(
                query=queries[target_name],
                weights=component_weights[target_name],
                k_samples=config['sampling_budget'],
                sampling_temperature=sampling_temperature,
                random_state=np.random.RandomState(retrieval_seed),
                candidate_indices=candidate_indices,
                return_metadata=True,
            )
            if target_name not in retrieval_distributions:
                retrieval_distributions[target_name] = {
                    field_name: retrieval[field_name]
                    for field_name in RETRIEVAL_DISTRIBUTION_FIELDS
                }
            target_retrievals[target_name] = {
                'retrieval_seed': retrieval_seed,
                **compact_retrieval(retrieval),
            }

        first_target, second_target = (target['name'] for target in targets)
        first_reward = target_retrievals[first_target]['decoded_recombined_reward']
        second_reward = target_retrievals[second_target]['decoded_recombined_reward']
        estimated_reward_gap = first_reward - second_reward
        if estimated_reward_gap > 0.0:
            chosen_target = first_target
        elif estimated_reward_gap < 0.0:
            chosen_target = second_target
        else:
            chosen_target = 'tie'
        repetitions.append({
            'repetition_id': repetition_id,
            'target_retrievals': target_retrievals,
            'estimated_reward_gap': estimated_reward_gap,
            'chosen_target': chosen_target,
            'oracle_choice': oracle_choice,
            'is_correct': chosen_target == oracle_choice,
        })

    return {
        'config': dict(config),
        'component_weights': component_weights,
        'retrieval_distributions': retrieval_distributions,
        'repetitions': repetitions,
    }


def prepare_session_knowledge_base(
    *,
    session_id: int,
    session_seed: int,
    session_size: int,
    required_oracle_choice: str,
    target_mean_gap: float,
    simulation_config: dict[str, Any],
    test_ds: Any,
    cortex_encoder: CortexEncoder,
) -> tuple[dict[str, Any], EpisodicSamplingModuleFeatureRecombination, dict[str, CompressedEpisodeFactorized]]:
    """
    Encode, store, and query one feature-rich session.
    """
    session = sample_balanced_session(
        data_table=test_ds.data_table,
        session_size=session_size,
        session_seed=session_seed,
    )
    reward_seed = build_coordinate_seed(session_seed, 1)
    reward_assignment = simulation_config['reward_assignment']
    targets = simulation_config['targets']
    session, reward_audit = assign_target_rewards(
        session=session,
        reward_values=np.asarray(reward_assignment['values'], dtype=float),
        targets=targets,
        target_mean_gap=target_mean_gap,
        required_oracle_choice=required_oracle_choice,
        random_state=np.random.RandomState(reward_seed),
    )
    batch = materialize_session_batch(session=session, dataset=test_ds)
    stimulus_embeddings, reward_embeddings = cortex_encoder.encode_session(
        pixel_values=batch.pixel_values,
        factors=batch.factors,
        rewards=batch.rewards,
    )
    episodic_sampler = EpisodicSamplingModuleFeatureRecombination(
        reward_decoder=cortex_encoder.reward_model.decoder,
        sample_replacement=simulation_config['sample_replacement'],
        query_key_normalization=simulation_config['query_key_normalization'],
        device=cortex_encoder.device,
    )
    episodic_sampler.store_session(build_hidden_states(
        stimulus_embeddings=stimulus_embeddings,
        reward_embeddings=reward_embeddings,
    ))
    queries = {
        target['name']: build_feature_query(session=session, episodic_sampler=episodic_sampler, target=target)
        for target in targets
    }
    session_record = {
        'session_id': session_id,
        'session_seed': session_seed,
        'session_metadata': session.to_dict(orient='records'),
        'targets': targets,
        'required_oracle_choice': required_oracle_choice,
        'reward_assignment': {'reward_seed': reward_seed, **reward_audit},
        'storage_validation': validate_stored_session(
            episodic_sampler=episodic_sampler,
            expected_session_size=session_size,
        ),
        'retrieval_candidate_indices': list(range(session_size)),
    }

    return session_record, episodic_sampler, queries


def run_choice_grid(
    *,
    simulation_config: dict[str, Any],
    queries: dict[str, CompressedEpisodeFactorized],
    episodic_sampler: EpisodicSamplingModuleFeatureRecombination,
    session: dict[str, Any],
    parallel: bool,
    n_jobs: int,
) -> list[dict[str, Any]]:
    """
    Evaluate every configured sampling-budget condition for one stored session.
    """
    grid = expand_retrieval_grid(simulation_config)

    def run_grid_cell(grid_cell_id: int, config: dict[str, Any]) -> dict[str, Any]:
        """
        Run one paired-choice retrieval condition.
        """
        return run_choice_grid_cell(
            config=config,
            queries=queries,
            targets=simulation_config['targets'],
            episodic_sampler=episodic_sampler,
            n_retrieval_repetitions=simulation_config['n_retrieval_repetitions'],
            sampling_temperature=simulation_config['sampling_temperature'],
            random_seed=simulation_config['random_seed'],
            session_id=session['session_id'],
            grid_cell_id=grid_cell_id,
            candidate_indices=session['retrieval_candidate_indices'],
            oracle_choice=session['reward_assignment']['oracle_choice'],
        )

    if parallel:
        return Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(run_grid_cell)(grid_cell_id, config)
            for grid_cell_id, config in enumerate(grid)
        )

    return [run_grid_cell(grid_cell_id, config) for grid_cell_id, config in enumerate(grid)]


def save_simulation_record(
    *,
    simulation_config: dict[str, Any],
    session: dict[str, Any],
    choice_grid: list[dict[str, Any]],
) -> None:
    """
    Save one session's paired feature-choice outcomes.
    """
    savepath = get_replacement_output_dir(
        root_dir=filepaths.EXPERIMENTS_DIR,
        output_name=simulation_config['experiment_name'],
        sample_replacement=simulation_config['sample_replacement'],
    )
    savepath.mkdir(parents=True, exist_ok=True)
    np.save(
        savepath / f"001_session-{session['session_id']:03d}.npy",
        {
            'schema_version': 1,
            'config': simulation_config,
            'session': session,
            'choice_grid': choice_grid,
        },
        allow_pickle=True,
    )


def run_simulation(*, args: argparse.Namespace, parallel: bool) -> None:
    """
    Run paired feature choices across configured sessions and sampling budgets.
    """
    simulation_config = resolve_simulation_config(args)
    _, test_ds, _ = get_Shapes3D_datasets(file_path=filepaths.SHAPES3D_FILE_PATH, **FEATURE_CONFIG)
    cortex_encoder = CortexEncoder(
        stimulus_model_dir=filepaths.OUTPUT_DIR / 'cvae' / 'shape3d_genesis_2',
        reward_model_dir=filepaths.OUTPUT_DIR / 'reward_linear_decoder',
        device=DEVICE,
    )
    session_size = simulation_config['session_size']

    for oracle_choice_index, required_oracle_choice in enumerate(simulation_config['oracle_choices']):
        target_gaps = build_target_mean_gap_schedule(
            n_sessions=simulation_config['n_sessions'],
            target_mean_gap_range=simulation_config['reward_assignment']['target_mean_gap_range'],
            required_oracle_choice=required_oracle_choice,
        )
        for condition_session_id, target_mean_gap in enumerate(target_gaps):
            session_id = oracle_choice_index * simulation_config['n_sessions'] + condition_session_id
            session_seed = build_coordinate_seed(
                simulation_config['random_seed'],
                oracle_choice_index,
                condition_session_id,
            )
            session, episodic_sampler, queries = prepare_session_knowledge_base(
                session_id=session_id,
                session_seed=session_seed,
                session_size=session_size,
                required_oracle_choice=required_oracle_choice,
                target_mean_gap=float(target_mean_gap),
                simulation_config=simulation_config,
                test_ds=test_ds,
                cortex_encoder=cortex_encoder,
            )
            session['population_id'] = f'balanced_{session_size}'
            session['session_size'] = session_size
            choice_grid = run_choice_grid(
                simulation_config=simulation_config,
                queries=queries,
                episodic_sampler=episodic_sampler,
                session=session,
                parallel=parallel,
                n_jobs=args.n_jobs,
            )
            save_simulation_record(
                simulation_config=simulation_config,
                session=session,
                choice_grid=choice_grid,
            )


if __name__ == '__main__':
    parsed_args = parse_args()
    run_simulation(args=parsed_args, parallel=parsed_args.parallel)
