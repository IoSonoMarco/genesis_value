"""
Run the supplementary GENESIS-budget versus TD two-armed bandit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path.cwd()))

from joblib import Parallel, delayed
import numpy as np
import torch
import yaml

from configs import shape3d_feature_config_genesis
from data import get_Shapes3D_datasets
from models.agent import SimpleTwoArmedBanditPolicy
from models.memory import CompressedEpisode, EpisodicSamplingModuleTwoArmedBandit
from models.memory.utils import generate_temporal_embeddings
from scripts.simulation_genesis_reward_bandit_task import (
    BanditTaskConfig,
    CortexEncoder,
    EnvironmentSimulationCache,
    build_environment_cache,
    build_task_trials_schedule,
)
from utils import filepaths


DEFAULT_CONFIG_PATH = (
    filepaths.ROOT / 'configs' / 'supp_simulation_genesis_reward_bandit_task_temporal_only.yaml'
)
MODEL_PREFIX = 'GENESIS_'
TD_MODEL_ID = 'TD_RL'
STREAM_NAMES = (
    'environment',
    'temporal',
    'warmup',
    'potential_outcomes',
    'policy',
    'retrieval',
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse supplementary simulation arguments.

    Args:
        argv (list[str] | None): Optional command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run supplementary GENESIS-budget versus TD bandit.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--n-environments', type=int)
    parser.add_argument('--n-agent-simulations', type=int)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument('--parallel', action='store_true', dest='parallel')
    execution.add_argument('--serial', action='store_false', dest='parallel')
    parser.set_defaults(parallel=None)
    parser.add_argument('--n-jobs', type=int)
    return parser.parse_args(argv)


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load the supplementary simulation YAML.

    Args:
        config_path (Path): Root-relative or absolute YAML path.

    Returns:
        dict[str, Any]: Parsed configuration.
    """
    path = config_path if config_path.is_absolute() else filepaths.ROOT / config_path
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def resolve_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """
    Apply CLI overrides to a copied configuration.

    Args:
        config (dict[str, Any]): Original YAML configuration.
        args (argparse.Namespace): Parsed CLI overrides.

    Returns:
        dict[str, Any]: Resolved configuration.
    """
    resolved = deepcopy(config)
    if args.n_environments is not None:
        resolved['n_environments'] = args.n_environments
    if args.n_agent_simulations is not None:
        resolved['n_agent_simulations'] = args.n_agent_simulations
    if args.parallel is not None:
        resolved['execution']['parallel'] = args.parallel
    if args.n_jobs is not None:
        resolved['execution']['n_jobs'] = args.n_jobs
    return resolved


def model_ids_from_config(config: dict[str, Any]) -> list[str]:
    """
    Construct the ordered model identifiers.

    Args:
        config (dict[str, Any]): Resolved simulation configuration.

    Returns:
        list[str]: Seven GENESIS identifiers followed by TD.
    """
    labels = [
        f'{MODEL_PREFIX}{str(budget).upper()}'
        for budget in config['genesis']['sampling_budgets']
    ]
    return [*labels, TD_MODEL_ID]


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate the fixed primary protocol.

    Args:
        config (dict[str, Any]): Resolved simulation configuration.

    Raises:
        ValueError: If the configuration violates the supplementary protocol.
    """
    if config['n_environments'] <= 0 or config['n_agent_simulations'] <= 0:
        raise ValueError('Environment and agent counts must be positive.')

    task = config['task']
    if task['n_choice_trials'] <= 0:
        raise ValueError('task.n_choice_trials must be positive.')
    if not 0 <= task['n_random_warmup_trials'] < task['n_choice_trials']:
        raise ValueError('The warm-up count must be non-negative and smaller than the trial count.')
    if task['n_probes'] != 0:
        raise ValueError('The temporal-only primary comparison requires task.n_probes=0.')
    if not 0 < task['p_win_low'] < task['p_win_high'] < 1:
        raise ValueError('Win-probability bounds must be ordered inside (0, 1).')
    if not np.isclose(task['p_win_low'] + task['p_win_high'], 1.0):
        raise ValueError('Win-probability bounds must be complementary around 0.5.')
    if not task['p_win_low'] <= task['p_win_start_high'] <= task['p_win_high']:
        raise ValueError('task.p_win_start_high must lie within the reflecting bounds.')
    if task['random_walk_sigma'] < 0:
        raise ValueError('task.random_walk_sigma must be non-negative.')
    if task['reward_win'] != 1.0 or task['reward_loss'] != -1.0:
        raise ValueError('The primary protocol requires binary +1/-1 rewards.')

    genesis = config['genesis']
    budgets = genesis['sampling_budgets']
    if len(budgets) != 7 or budgets[-1] != 'all':
        raise ValueError('GENESIS budgets must contain six finite entries followed by all.')
    finite_budgets = budgets[:-1]
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in finite_budgets):
        raise ValueError('Finite GENESIS budgets must be positive integers.')
    if len(set(finite_budgets)) != len(finite_budgets):
        raise ValueError('Finite GENESIS budgets must be unique.')
    if genesis['sampling_temperature'] <= 0:
        raise ValueError('GENESIS sampling temperature must be positive.')
    if genesis['sample_replacement'] is not True:
        raise ValueError('The primary comparison requires replacement sampling.')
    if genesis['temporal_weight'] != 1.0 or genesis['query_key_normalization'] != 'legacy':
        raise ValueError('The primary comparison requires temporal-only legacy retrieval.')
    if genesis['all_mode'] != 'literal_unweighted':
        raise ValueError('GENESIS ALL must use literal unweighted retrieval.')

    td_config = config['td']
    if td_config['enabled'] is not True:
        raise ValueError('The eight-condition primary comparison requires TD.')
    if not 0 < td_config['learning_rate'] <= 1:
        raise ValueError('TD learning rate must lie in (0, 1].')
    if td_config['perseveration'] != 0.0:
        raise ValueError('The primary comparison does not include perseveration.')
    if config['choice_policy']['beta_choice'] != 2.0:
        raise ValueError('The primary comparison requires beta_choice=2.0.')
    if not all(config['pairing'].values()):
        raise ValueError('All paired-randomness features are required.')
    if config['temporal']['embedding_dim'] <= 0 or config['temporal']['embedding_angle'] <= 0:
        raise ValueError('Temporal embedding settings must be positive.')
    if config['execution']['device'] not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('execution.device must be auto, cpu, or cuda.')
    if config['execution']['n_jobs'] != -1 and config['execution']['n_jobs'] <= 0:
        raise ValueError('execution.n_jobs must be -1 or a positive integer.')
    if config['output']['schema_version'] != 1:
        raise ValueError('Only supplementary schema version 1 is supported.')
    if model_ids_from_config(config) != [
        'GENESIS_1',
        'GENESIS_3',
        'GENESIS_5',
        'GENESIS_10',
        'GENESIS_20',
        'GENESIS_50',
        'GENESIS_ALL',
        'TD_RL',
    ]:
        raise ValueError('The primary model order must contain the specified eight conditions.')


def resolve_device(device_name: str) -> torch.device:
    """
    Resolve the configured inference device.

    Args:
        device_name (str): Configured device name.

    Returns:
        torch.device: Runtime inference device.
    """
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable.')
    return torch.device(device_name)


def derive_seed(base_seed: int, stream: str, *identifiers: int | str) -> int:
    """
    Derive a stable 32-bit seed for one independent random stream.

    Args:
        base_seed (int): Experiment base seed.
        stream (str): Stream name.
        identifiers (int | str): Environment, agent, or model identifiers.

    Returns:
        int: Stable seed accepted by NumPy random generators.
    """
    if stream not in STREAM_NAMES:
        raise ValueError(f'Unknown random stream: {stream}')
    payload = ':'.join(map(str, (base_seed, stream, *identifiers))).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder='little')


def config_fingerprint(config: dict[str, Any]) -> str:
    """
    Hash the resolved scientific and execution configuration.

    Args:
        config (dict[str, Any]): Resolved configuration.

    Returns:
        str: Twelve-character configuration fingerprint.
    """
    payload = json.dumps(config, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:12]


def build_task_config(config: dict[str, Any], environment_seed: int) -> BanditTaskConfig:
    """
    Convert YAML task settings into the reused bandit task dataclass.

    Args:
        config (dict[str, Any]): Resolved simulation configuration.
        environment_seed (int): Environment schedule seed.

    Returns:
        BanditTaskConfig: Schedule builder configuration.
    """
    task = config['task']
    return BanditTaskConfig(
        n_choice_trials=task['n_choice_trials'],
        n_probes=task['n_probes'],
        p_win_low=task['p_win_low'],
        p_win_high=task['p_win_high'],
        p_win_start_high=task['p_win_start_high'],
        random_walk_sigma=task['random_walk_sigma'],
        random_seed=environment_seed,
        first_reminder_trial=task['n_random_warmup_trials'],
    )


def build_shared_agent_inputs(
    *,
    config: dict[str, Any],
    schedule: list[dict[str, Any]],
    environment_id: int,
    agent_id: int,
) -> dict[str, Any]:
    """
    Generate paired actions, outcomes, and choice uniforms for one agent.

    Args:
        config (dict[str, Any]): Resolved configuration.
        schedule (list[dict[str, Any]]): Environment trial schedule.
        environment_id (int): Environment identifier.
        agent_id (int): Agent replicate identifier.

    Returns:
        dict[str, Any]: Shared random inputs and their seeds.
    """
    base_seed = config['base_random_seed']
    warmup_seed = derive_seed(base_seed, 'warmup', environment_id, agent_id)
    outcome_seed = derive_seed(base_seed, 'potential_outcomes', environment_id, agent_id)
    policy_seed = derive_seed(base_seed, 'policy', environment_id, agent_id)
    warmup_rng = np.random.default_rng(warmup_seed)
    outcome_rng = np.random.default_rng(outcome_seed)
    policy_rng = np.random.default_rng(policy_seed)
    n_trials = len(schedule)
    n_warmup = config['task']['n_random_warmup_trials']
    warmup_actions = np.where(warmup_rng.random(n_warmup) < 0.5, 'RIGHT', 'LEFT')
    outcome_uniforms = outcome_rng.random((n_trials, 2))
    potential_outcomes = []
    for trial_id, trial in enumerate(schedule):
        potential_outcomes.append({
            'LEFT': 1.0 if outcome_uniforms[trial_id, 0] < trial['p_win_left'] else -1.0,
            'RIGHT': 1.0 if outcome_uniforms[trial_id, 1] < trial['p_win_right'] else -1.0,
        })
    return {
        'seeds': {
            'warmup': warmup_seed,
            'potential_outcomes': outcome_seed,
            'policy': policy_seed,
        },
        'warmup_actions': warmup_actions.tolist(),
        'policy_choice_uniforms': policy_rng.random(n_trials).tolist(),
        'potential_outcomes': potential_outcomes,
    }


def budget_from_model_id(model_id: str) -> int | Literal['all']:
    """
    Recover a GENESIS sampling budget from its model identifier.

    Args:
        model_id (str): GENESIS model identifier.

    Returns:
        int | Literal['all']: Episodic sampling budget.
    """
    value = model_id.removeprefix(MODEL_PREFIX)
    return 'all' if value == 'ALL' else int(value)


def run_genesis_history(
    *,
    model_id: str,
    config: dict[str, Any],
    environment_cache: EnvironmentSimulationCache,
    temporal_embeddings: torch.Tensor,
    shared_inputs: dict[str, Any],
    cortex_encoder: CortexEncoder,
    environment_id: int,
    agent_id: int,
    device: torch.device,
) -> dict[str, Any]:
    """
    Run one GENESIS condition with an independent episodic history.

    Args:
        model_id (str): GENESIS condition identifier.
        config (dict[str, Any]): Resolved configuration.
        environment_cache (EnvironmentSimulationCache): Static schedule and embeddings.
        temporal_embeddings (torch.Tensor): Shared environment temporal contexts.
        shared_inputs (dict[str, Any]): Paired random inputs.
        cortex_encoder (CortexEncoder): Frozen cortical encoders.
        environment_id (int): Environment identifier.
        agent_id (int): Agent replicate identifier.
        device (torch.device): Reward decoder device.

    Returns:
        dict[str, Any]: Model metadata and trial history.
    """
    budget = budget_from_model_id(model_id)
    retrieval_seed = derive_seed(
        config['base_random_seed'],
        'retrieval',
        environment_id,
        agent_id,
        model_id,
    )
    sampler = EpisodicSamplingModuleTwoArmedBandit(
        weights={
            'stimulus_embedding': 0.0,
            'reward_embedding': 0.0,
            'time_embedding': config['genesis']['temporal_weight'],
        },
        k_samples=budget,
        sample_replacement=config['genesis']['sample_replacement'],
        sampling_temperature=config['genesis']['sampling_temperature'],
        query_key_normalization=config['genesis']['query_key_normalization'],
        device=device,
        reward_decoder=cortex_encoder.reward_model.decoder,
        random_state=np.random.RandomState(retrieval_seed),
    )
    policy = SimpleTwoArmedBanditPolicy(beta_choice=config['choice_policy']['beta_choice'])
    cumulative_reward = 0.0
    raw_rewards = {'LEFT': [], 'RIGHT': []}
    trials = []
    n_warmup = config['task']['n_random_warmup_trials']

    for trial in environment_cache.schedule:
        trial_id = trial['trial_id']
        is_warmup = trial_id < n_warmup
        time_embedding = temporal_embeddings[trial_id]
        metadata = {
            arm: {
                'candidate_indices': list(sampler.candidate_indices_by_arm[arm]),
                'sampled_episode_indices': [],
            }
            for arm in ['LEFT', 'RIGHT']
        }
        action_value = {'LEFT': np.nan, 'RIGHT': np.nan}
        p_right = np.nan
        logit_right = np.nan

        if is_warmup:
            chosen_arm = shared_inputs['warmup_actions'][trial_id]
        else:
            query = CompressedEpisode(time_embedding=time_embedding)
            action_value, metadata = sampler.estimate_action_values(query=query, return_metadata=True)
            logit_right = policy.beta_choice * (action_value['RIGHT'] - action_value['LEFT'])
            p_right = policy.sigmoid(logit_right)
            chosen_arm = (
                'RIGHT'
                if shared_inputs['policy_choice_uniforms'][trial_id] < p_right
                else 'LEFT'
            )

        exact_mean = {
            arm: float(np.mean(raw_rewards[arm])) if raw_rewards[arm] else 0.0
            for arm in ['LEFT', 'RIGHT']
        }
        reward = shared_inputs['potential_outcomes'][trial_id][chosen_arm]
        cumulative_reward += reward
        stimulus_embedding = environment_cache.stimulus_embeddings[trial['choice_image']['h5_idx']]
        reward_embedding = environment_cache.reward_embeddings[reward]
        sampler.store_episode({
            'hidden_states': {
                'stimulus_embedding': stimulus_embedding,
                'reward_embedding': reward_embedding,
                'time_embedding': time_embedding,
            },
            'chosen_arm': chosen_arm,
            'reward': torch.tensor(reward, dtype=torch.float32),
            'trial_metadata': trial,
        })
        raw_rewards[chosen_arm].append(reward)
        trials.append({
            'trial_id': trial_id,
            'model_id': model_id,
            'is_random_warmup': is_warmup,
            'chosen_arm': chosen_arm,
            'reward': reward,
            'cumulative_reward': cumulative_reward,
            'action_value_left': float(action_value['LEFT']),
            'action_value_right': float(action_value['RIGHT']),
            'p_right': float(p_right),
            'logit_right': float(logit_right),
            'p_win_left': trial['p_win_left'],
            'p_win_right': trial['p_win_right'],
            'memory_count_left': len(metadata['LEFT']['candidate_indices']),
            'memory_count_right': len(metadata['RIGHT']['candidate_indices']),
            'retrieved_indices_left': list(metadata['LEFT']['sampled_episode_indices']),
            'retrieved_indices_right': list(metadata['RIGHT']['sampled_episode_indices']),
            'sampling_budget': budget,
            'sampling_temperature': config['genesis']['sampling_temperature'],
            'sample_replacement': config['genesis']['sample_replacement'],
            'td_learning_rate': None,
            'exact_sample_mean_left': exact_mean['LEFT'] if budget == 'all' else None,
            'exact_sample_mean_right': exact_mean['RIGHT'] if budget == 'all' else None,
        })

    return {
        'model_id': model_id,
        'sampling_budget': budget,
        'sampling_temperature': config['genesis']['sampling_temperature'],
        'sample_replacement': config['genesis']['sample_replacement'],
        'retrieval_seed': retrieval_seed if budget != 'all' else None,
        'trials': trials,
    }


def run_td_history(
    *,
    config: dict[str, Any],
    schedule: list[dict[str, Any]],
    shared_inputs: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the prespecified chosen-arm TD baseline.

    Args:
        config (dict[str, Any]): Resolved configuration.
        schedule (list[dict[str, Any]]): Environment trial schedule.
        shared_inputs (dict[str, Any]): Paired random inputs.

    Returns:
        dict[str, Any]: TD metadata and trial history.
    """
    td_config = config['td']
    policy = SimpleTwoArmedBanditPolicy(beta_choice=config['choice_policy']['beta_choice'])
    values = {'LEFT': td_config['initial_q_value'], 'RIGHT': td_config['initial_q_value']}
    n_warmup = config['task']['n_random_warmup_trials']
    cumulative_reward = 0.0
    trials = []

    for trial in schedule:
        trial_id = trial['trial_id']
        is_warmup = trial_id < n_warmup
        pre_choice_values = dict(values)
        p_right = np.nan
        logit_right = np.nan
        if is_warmup:
            chosen_arm = shared_inputs['warmup_actions'][trial_id]
        else:
            logit_right = policy.beta_choice * (values['RIGHT'] - values['LEFT'])
            p_right = policy.sigmoid(logit_right)
            chosen_arm = (
                'RIGHT'
                if shared_inputs['policy_choice_uniforms'][trial_id] < p_right
                else 'LEFT'
            )
        reward = shared_inputs['potential_outcomes'][trial_id][chosen_arm]
        cumulative_reward += reward
        values[chosen_arm] += td_config['learning_rate'] * (reward - values[chosen_arm])
        trials.append({
            'trial_id': trial_id,
            'model_id': TD_MODEL_ID,
            'is_random_warmup': is_warmup,
            'chosen_arm': chosen_arm,
            'reward': reward,
            'cumulative_reward': cumulative_reward,
            'action_value_left': float(pre_choice_values['LEFT']),
            'action_value_right': float(pre_choice_values['RIGHT']),
            'p_right': float(p_right),
            'logit_right': float(logit_right),
            'p_win_left': trial['p_win_left'],
            'p_win_right': trial['p_win_right'],
            'memory_count_left': None,
            'memory_count_right': None,
            'retrieved_indices_left': None,
            'retrieved_indices_right': None,
            'sampling_budget': None,
            'sampling_temperature': None,
            'sample_replacement': None,
            'td_learning_rate': td_config['learning_rate'],
            'exact_sample_mean_left': None,
            'exact_sample_mean_right': None,
        })
    return {
        'model_id': TD_MODEL_ID,
        'sampling_budget': None,
        'sampling_temperature': None,
        'sample_replacement': None,
        'td_learning_rate': td_config['learning_rate'],
        'trials': trials,
    }


def run_environment(
    *,
    environment_id: int,
    config: dict[str, Any],
    test_ds: Any,
    cortex_encoder: CortexEncoder,
    device: torch.device,
) -> dict[str, Any]:
    """
    Run every paired agent and model in one environment.

    Args:
        environment_id (int): Environment identifier.
        config (dict[str, Any]): Resolved configuration.
        test_ds (Any): Shapes3D test dataset.
        cortex_encoder (CortexEncoder): Frozen cortical encoders.
        device (torch.device): Inference device.

    Returns:
        dict[str, Any]: Nested environment simulation record.
    """
    base_seed = config['base_random_seed']
    environment_seed = derive_seed(base_seed, 'environment', environment_id)
    temporal_seed = derive_seed(base_seed, 'temporal', environment_id)
    schedule = build_task_trials_schedule(
        test_ds.data_table,
        build_task_config(config, environment_seed),
    )
    environment_cache = build_environment_cache(
        schedule=schedule,
        test_ds=test_ds,
        cortex_encoder=cortex_encoder,
    )
    temporal_config = config['temporal']
    temporal_embeddings = generate_temporal_embeddings(
        [temporal_config['embedding_angle']] * len(schedule),
        dim=temporal_config['embedding_dim'],
        random_state=np.random.RandomState(temporal_seed),
    )
    temporal_embeddings = torch.tensor(temporal_embeddings, dtype=torch.float32)
    agent_records = []
    model_ids = model_ids_from_config(config)

    for agent_id in range(config['n_agent_simulations']):
        shared_inputs = build_shared_agent_inputs(
            config=config,
            schedule=schedule,
            environment_id=environment_id,
            agent_id=agent_id,
        )
        histories = []
        for model_id in model_ids[:-1]:
            histories.append(run_genesis_history(
                model_id=model_id,
                config=config,
                environment_cache=environment_cache,
                temporal_embeddings=temporal_embeddings,
                shared_inputs=shared_inputs,
                cortex_encoder=cortex_encoder,
                environment_id=environment_id,
                agent_id=agent_id,
                device=device,
            ))
        histories.append(run_td_history(
            config=config,
            schedule=schedule,
            shared_inputs=shared_inputs,
        ))
        agent_records.append({
            'agent_id': agent_id,
            'shared_inputs': shared_inputs,
            'model_histories': histories,
        })
    return {
        'environment_id': environment_id,
        'environment_seed': environment_seed,
        'temporal_seed': temporal_seed,
        'schedule': schedule,
        'agent_records': agent_records,
    }


def validate_complete_record(record: dict[str, Any]) -> None:
    """
    Validate exact environment, agent, model, and trial completeness.

    Args:
        record (dict[str, Any]): Supplementary nested simulation record.

    Raises:
        ValueError: If any required combination is missing or duplicated.
    """
    config = record['resolved_config']
    expected_models = record['model_ids']
    expected_trials = config['task']['n_choice_trials']
    if record['config_fingerprint'] != config_fingerprint(config):
        raise ValueError('Resolved configuration does not match the saved fingerprint.')
    if record['actual_n_environments'] != config['n_environments']:
        raise ValueError('Actual environment count does not match the resolved configuration.')
    if record['actual_n_agent_simulations'] != config['n_agent_simulations']:
        raise ValueError('Actual agent count does not match the resolved configuration.')
    if len(record['environments']) != config['n_environments']:
        raise ValueError('The environment list is incomplete.')
    keys = []
    for environment in record['environments']:
        if len(environment['agent_records']) != config['n_agent_simulations']:
            raise ValueError('An environment has an incomplete agent set.')
        for agent in environment['agent_records']:
            if [history['model_id'] for history in agent['model_histories']] != expected_models:
                raise ValueError('A model history set is missing, duplicated, or out of order.')
            for history in agent['model_histories']:
                if len(history['trials']) != expected_trials:
                    raise ValueError('A model history has an incomplete trial sequence.')
                keys.extend(
                    (environment['environment_id'], agent['agent_id'], history['model_id'], trial['trial_id'])
                    for trial in history['trials']
                )
    expected_count = (
        config['n_environments']
        * config['n_agent_simulations']
        * len(expected_models)
        * expected_trials
    )
    expected_runs = config['n_environments'] * config['n_agent_simulations'] * len(expected_models)
    if record['expected_complete_runs'] != expected_runs or record['expected_trial_records'] != expected_count:
        raise ValueError('Saved expected counts do not match the resolved configuration.')
    if len(keys) != expected_count or len(set(keys)) != expected_count:
        raise ValueError('Simulation records are incomplete or contain duplicate keys.')


def run_simulation(args: argparse.Namespace) -> Path:
    """
    Run or reuse one exact supplementary simulation configuration.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        Path: Saved schema-v1 record path.
    """
    original_config = load_config(args.config)
    resolved_config = resolve_config(original_config, args)
    validate_config(resolved_config)
    device = resolve_device(resolved_config['execution']['device'])
    fingerprint = config_fingerprint(resolved_config)
    output_dir = (
        filepaths.EXPERIMENTS_DIR
        / resolved_config['experiment_name']
        / resolved_config['output']['sample_replacement_subdir']
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'simulation_{fingerprint}.npy'
    if output_path.exists():
        record = np.load(output_path, allow_pickle=True).item()
        if record.get('config_fingerprint') != fingerprint:
            raise ValueError('Existing simulation fingerprint does not match its filename.')
        validate_complete_record(record)
        print(f'Reusing complete simulation: {output_path}')
        return output_path

    _, test_ds, _ = get_Shapes3D_datasets(
        file_path=filepaths.SHAPES3D_FILE_PATH,
        **shape3d_feature_config_genesis.feature_config,
    )
    stimulus_checkpoint = filepaths.OUTPUT_DIR / 'cvae' / 'shape3d_genesis_2' / 'ckpt' / 'best.pt'
    reward_checkpoint = next((filepaths.OUTPUT_DIR / 'reward_linear_decoder').rglob('weights.pt'))
    cortex_encoder = CortexEncoder(
        stimulus_model_dir=filepaths.OUTPUT_DIR / 'cvae' / 'shape3d_genesis_2',
        reward_model_dir=filepaths.OUTPUT_DIR / 'reward_linear_decoder',
        device=device,
    )
    environment_ids = range(resolved_config['n_environments'])
    if resolved_config['execution']['parallel']:
        environments = Parallel(
            n_jobs=resolved_config['execution']['n_jobs'],
            prefer='threads',
        )(
            delayed(run_environment)(
                environment_id=environment_id,
                config=resolved_config,
                test_ds=test_ds,
                cortex_encoder=cortex_encoder,
                device=device,
            )
            for environment_id in environment_ids
        )
    else:
        environments = [
            run_environment(
                environment_id=environment_id,
                config=resolved_config,
                test_ds=test_ds,
                cortex_encoder=cortex_encoder,
                device=device,
            )
            for environment_id in environment_ids
        ]
    environments.sort(key=lambda environment: environment['environment_id'])
    model_ids = model_ids_from_config(resolved_config)
    expected_runs = resolved_config['n_environments'] * resolved_config['n_agent_simulations'] * len(model_ids)
    record = {
        'schema_version': resolved_config['output']['schema_version'],
        'experiment_name': resolved_config['experiment_name'],
        'config_fingerprint': fingerprint,
        'original_config': original_config,
        'resolved_config': resolved_config,
        'actual_n_environments': len(environments),
        'actual_n_agent_simulations': resolved_config['n_agent_simulations'],
        'model_ids': model_ids,
        'sampling_budgets': resolved_config['genesis']['sampling_budgets'],
        'expected_complete_runs': expected_runs,
        'expected_trial_records': expected_runs * resolved_config['task']['n_choice_trials'],
        'checkpoints': {
            'stimulus': str(stimulus_checkpoint),
            'reward': str(reward_checkpoint),
        },
        'random_stream_policy': {
            'base_random_seed': resolved_config['base_random_seed'],
            'derivation': 'sha256(base_seed:stream:identifiers) first 32 bits',
            'stream_names': list(STREAM_NAMES),
        },
        'task_distinction': (
            'Existing complementary-probability reflected-walk GENESIS bandit; '
            'not Bornstein independent drifts.'
        ),
        'environments': environments,
    }
    validate_complete_record(record)
    np.save(output_path, record, allow_pickle=True)
    print(f'Saved {expected_runs} complete model histories to {output_path}')
    return output_path


def main(argv: list[str] | None = None) -> None:
    """
    Run the supplementary simulation from command-line arguments.

    Args:
        argv (list[str] | None): Optional command-line arguments.
    """
    run_simulation(parse_args(argv))


if __name__ == '__main__':
    main()
