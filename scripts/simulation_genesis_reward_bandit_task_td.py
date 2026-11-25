"""
Run TD baselines on the completed temporal-only GENESIS bandit environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.cwd()))

from joblib import Parallel, delayed
import numpy as np
import yaml

from models.agent import SimpleTwoArmedBanditPolicy
from utils import filepaths


DEFAULT_CONFIG_PATH = (
    filepaths.ROOT / 'configs' / 'simulation_genesis_reward_bandit_task_temporal_only_td.yaml'
)
SOURCE_CONFIG_PATH = (
    filepaths.ROOT / 'configs' / 'simulation_genesis_reward_bandit_task_temporal_only.yaml'
)
SOURCE_SUBDIR = 'sample_replacement_true'
OUTPUT_SUBDIR = 'td_model'
N_RANDOM_WARMUP_TRIALS = 20
INITIAL_Q_VALUE = 0.0
BETA_CHOICE = 2.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse TD simulation arguments.

    Args:
        argv (list[str] | None): Optional command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run TD models on completed temporal-only bandit environments.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--n-jobs', type=int, default=-1)
    return parser.parse_args(argv)


def load_yaml(config_path: Path) -> dict[str, Any]:
    """
    Load a root-relative or absolute YAML file.

    Args:
        config_path (Path): YAML path.

    Returns:
        dict[str, Any]: Parsed YAML mapping.
    """
    path = config_path if config_path.is_absolute() else filepaths.ROOT / config_path
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def validate_td_config(config: dict[str, Any]) -> list[float]:
    """
    Validate and return the ordered TD learning-rate grid.

    Args:
        config (dict[str, Any]): TD configuration.

    Returns:
        list[float]: Ordered learning rates.

    Raises:
        ValueError: If the configuration contains extra or invalid parameters.
    """
    if set(config) != {'learning_rate'}:
        raise ValueError('The TD config must contain only the learning_rate list.')
    learning_rates = config['learning_rate']
    if not isinstance(learning_rates, list) or len(learning_rates) == 0:
        raise ValueError('learning_rate must be a non-empty list.')
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in learning_rates):
        raise ValueError('Every TD learning rate must be numeric.')
    learning_rates = [float(value) for value in learning_rates]
    if any(not 0 < value <= 1 for value in learning_rates):
        raise ValueError('Every TD learning rate must lie in (0, 1].')
    if len(set(learning_rates)) != len(learning_rates):
        raise ValueError('TD learning rates must be unique.')
    return learning_rates


def format_float(value: float) -> str:
    """
    Format a numeric value for labels and filenames.

    Args:
        value (float): Numeric value.

    Returns:
        str: Compact decimal representation.
    """
    return f'{value:g}'


def source_record_identity(record: dict[str, Any]) -> str:
    """
    Hash environment schedules, seeds, and warm-up experiences.

    Args:
        record (dict[str, Any]): One completed temperature record.

    Returns:
        str: SHA-256 source identity.
    """
    identity = []
    for environment in record['environments']:
        agents = []
        for agent in environment['agent_runs']:
            warmup = agent['simulation_log'][:N_RANDOM_WARMUP_TRIALS]
            agents.append({
                'agent_simulation_id': agent['agent_simulation_id'],
                'agent_seed': agent['agent_seed'],
                'warmup': [(trial['chosen_arm'], trial['reward']) for trial in warmup],
            })
        identity.append({
            'environment_id': environment['environment_id'],
            'environment_seed': environment['environment_seed'],
            'schedule': environment['schedule'],
            'agents': agents,
        })
    payload = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def validate_source_record(
    *,
    record: dict[str, Any],
    source_config: dict[str, Any],
) -> None:
    """
    Validate one completed temperature simulation record.

    Args:
        record (dict[str, Any]): Temperature record.
        source_config (dict[str, Any]): Existing simulation YAML.

    Raises:
        ValueError: If the record is incomplete or incompatible.
    """
    if len(record['environments']) != source_config['n_environments']:
        raise ValueError('A source record has an incompatible environment count.')
    for environment_id, environment in enumerate(record['environments']):
        if environment['environment_id'] != environment_id:
            raise ValueError('Source environments are missing, duplicated, or out of order.')
        if len(environment['schedule']) != 130:
            raise ValueError('A source environment does not contain 130 trials.')
        if len(environment['agent_runs']) != source_config['n_agent_simulations']:
            raise ValueError('A source environment has an incompatible agent count.')
        for agent_id, agent in enumerate(environment['agent_runs']):
            if agent['agent_simulation_id'] != agent_id or len(agent['simulation_log']) != 130:
                raise ValueError('A source agent history is incomplete or out of order.')


def load_completed_temperature_experiment(
    source_config: dict[str, Any],
) -> tuple[dict[str, Any], str, list[Path]]:
    """
    Load and cross-validate the completed temperature grid.

    Args:
        source_config (dict[str, Any]): Existing temperature simulation configuration.

    Returns:
        tuple[dict[str, Any], str, list[Path]]: Canonical record, source signature, and source paths.

    Raises:
        ValueError: If a configured temperature is missing or source identities differ.
    """
    source_dir = filepaths.EXPERIMENTS_DIR / source_config['experiment_name'] / SOURCE_SUBDIR
    records_by_temperature = {}
    paths_by_temperature = {}
    for path in sorted(source_dir.glob('*.npy')):
        record = np.load(path, allow_pickle=True).item()
        config = record.get('config', {})
        if config.get('sample_replacement') is not True or 'sampling_temperature' not in config:
            continue
        temperature = float(config['sampling_temperature'])
        records_by_temperature[temperature] = record
        paths_by_temperature[temperature] = path

    temperatures = [float(value) for value in source_config['sampling_temperature']]
    if set(records_by_temperature) != set(temperatures):
        raise ValueError('The completed source records do not exactly match the configured temperature grid.')

    identities = []
    source_paths = []
    for temperature in temperatures:
        record = records_by_temperature[temperature]
        validate_source_record(record=record, source_config=source_config)
        identities.append(source_record_identity(record))
        source_paths.append(paths_by_temperature[temperature])
    if len(set(identities)) != 1:
        raise ValueError('Temperature records do not share identical schedules, seeds, and warm-up histories.')

    signature_payload = json.dumps({
        'source_config': source_config,
        'paths': [path.name for path in source_paths],
        'identity': identities[0],
    }, sort_keys=True, separators=(',', ':')).encode('utf-8')
    signature = hashlib.sha256(signature_payload).hexdigest()
    return records_by_temperature[temperatures[0]], signature, source_paths


def run_td_agent(
    *,
    schedule: list[dict[str, Any]],
    source_agent: dict[str, Any],
    learning_rate: float,
) -> dict[str, Any]:
    """
    Run one TD model with source-matched warm-up and reward uniforms.

    Args:
        schedule (list[dict[str, Any]]): Saved environment schedule.
        source_agent (dict[str, Any]): Canonical source agent history.
        learning_rate (float): Constant chosen-arm TD learning rate.

    Returns:
        dict[str, Any]: Nested agent record in the existing simulation style.
    """
    agent_seed = source_agent['agent_seed']
    reward_rng = np.random.default_rng(agent_seed)
    choice_rng = np.random.RandomState(agent_seed)
    policy = SimpleTwoArmedBanditPolicy(beta_choice=BETA_CHOICE)
    values = {'LEFT': INITIAL_Q_VALUE, 'RIGHT': INITIAL_Q_VALUE}
    cumulative_reward = 0.0
    simulation_log = []

    for trial in schedule:
        trial_id = trial['trial_id']
        is_warmup = trial_id < N_RANDOM_WARMUP_TRIALS
        pre_choice_values = dict(values)
        if is_warmup:
            chosen_arm = source_agent['simulation_log'][trial_id]['chosen_arm']
            policy_output = {'p_right': np.nan, 'logit_right': np.nan}
            recorded_values = {'LEFT': np.nan, 'RIGHT': np.nan}
        else:
            policy_output = policy.choose_arm(pre_choice_values, random_state=choice_rng)
            chosen_arm = policy_output['chosen_arm']
            recorded_values = pre_choice_values

        p_win = trial['p_win_left'] if chosen_arm == 'LEFT' else trial['p_win_right']
        reward = 1.0 if reward_rng.random() < p_win else -1.0
        if is_warmup and reward != source_agent['simulation_log'][trial_id]['reward']:
            raise ValueError('Source warm-up reward does not match the reconstructed reward stream.')
        cumulative_reward += reward
        values[chosen_arm] += learning_rate * (reward - values[chosen_arm])
        simulation_log.append({
            'trial_id': trial_id,
            'is_reminder_trial': False,
            'reminder_target_trial_id': None,
            'reminder_lag': None,
            'reminder_target_chosen_arm': None,
            'reminder_target_reward': None,
            'reminder_target_candidate_count': None,
            'reminder_target_sampling_probability': None,
            'reminder_target_similarity_rank': None,
            'reminder_target_sampled': None,
            'chosen_arm': chosen_arm,
            'reward': reward,
            'cumulative_reward': cumulative_reward,
            'p_win': p_win,
            'p_win_left': trial['p_win_left'],
            'p_win_right': trial['p_win_right'],
            'action_value_left': float(recorded_values['LEFT']),
            'action_value_right': float(recorded_values['RIGHT']),
            'p_right': policy_output['p_right'],
            'logit_right': policy_output['logit_right'],
            'choice_class_id': trial['choice_image']['class_id'],
            'choice_h5_idx': trial['choice_image']['h5_idx'],
            'td_learning_rate': learning_rate,
            'metadata': None,
        })

    return {
        'agent_simulation_id': source_agent['agent_simulation_id'],
        'agent_seed': agent_seed,
        'simulation_log': simulation_log,
    }


def run_td_environment(
    *,
    source_environment: dict[str, Any],
    learning_rate: float,
) -> dict[str, Any]:
    """
    Run every TD agent in one saved environment.

    Args:
        source_environment (dict[str, Any]): Canonical source environment.
        learning_rate (float): TD learning rate.

    Returns:
        dict[str, Any]: Environment record with TD agent histories.
    """
    return {
        'environment_id': source_environment['environment_id'],
        'environment_seed': source_environment['environment_seed'],
        'schedule': source_environment['schedule'],
        'agent_runs': [
            run_td_agent(
                schedule=source_environment['schedule'],
                source_agent=source_agent,
                learning_rate=learning_rate,
            )
            for source_agent in source_environment['agent_runs']
        ],
    }


def validate_td_record(
    *,
    record: dict[str, Any],
    learning_rate: float,
    source_signature: str,
    source_config: dict[str, Any],
) -> None:
    """
    Validate a saved TD model record before reuse.

    Args:
        record (dict[str, Any]): Saved TD record.
        learning_rate (float): Expected learning rate.
        source_signature (str): Expected source identity.
        source_config (dict[str, Any]): Existing simulation configuration.

    Raises:
        ValueError: If the saved TD record is stale or incomplete.
    """
    expected_config = {
        'model': 'TD_RL',
        'learning_rate': learning_rate,
        'initial_q_value': INITIAL_Q_VALUE,
        'beta_choice': BETA_CHOICE,
        'n_random_warmup_trials': N_RANDOM_WARMUP_TRIALS,
    }
    if record.get('config') != expected_config or record.get('source_signature') != source_signature:
        raise ValueError('An existing TD record was generated from incompatible settings.')
    validate_source_record(record=record, source_config=source_config)


def output_filename(index: int, learning_rate: float) -> str:
    """
    Build a deterministic TD record filename.

    Args:
        index (int): One-based learning-rate index.
        learning_rate (float): TD learning rate.

    Returns:
        str: Output filename.
    """
    rate = format_float(learning_rate).replace('.', 'p')
    return f'{index:03d}_td_learning_rate-{rate}.npy'


def run_simulation(args: argparse.Namespace) -> None:
    """
    Run every configured TD model on the completed source experiment.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
    """
    if args.n_jobs != -1 and args.n_jobs <= 0:
        raise ValueError('--n-jobs must be -1 or a positive integer.')
    td_config = load_yaml(args.config)
    learning_rates = validate_td_config(td_config)
    source_config = load_yaml(SOURCE_CONFIG_PATH)
    source_record, source_signature, source_paths = load_completed_temperature_experiment(source_config)
    output_dir = filepaths.EXPERIMENTS_DIR / source_config['experiment_name'] / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, learning_rate in enumerate(learning_rates, start=1):
        output_path = output_dir / output_filename(index, learning_rate)
        if output_path.exists():
            record = np.load(output_path, allow_pickle=True).item()
            validate_td_record(
                record=record,
                learning_rate=learning_rate,
                source_signature=source_signature,
                source_config=source_config,
            )
            print(f'Reusing complete TD record: {output_path}')
            continue
        if args.parallel:
            environments = Parallel(n_jobs=args.n_jobs, prefer='threads')(
                delayed(run_td_environment)(
                    source_environment=environment,
                    learning_rate=learning_rate,
                )
                for environment in source_record['environments']
            )
        else:
            environments = [
                run_td_environment(
                    source_environment=environment,
                    learning_rate=learning_rate,
                )
                for environment in source_record['environments']
            ]
        environments.sort(key=lambda environment: environment['environment_id'])
        record = {
            'config': {
                'model': 'TD_RL',
                'learning_rate': learning_rate,
                'initial_q_value': INITIAL_Q_VALUE,
                'beta_choice': BETA_CHOICE,
                'n_random_warmup_trials': N_RANDOM_WARMUP_TRIALS,
            },
            'source_signature': source_signature,
            'source_files': [str(path) for path in source_paths],
            'environments': environments,
        }
        validate_td_record(
            record=record,
            learning_rate=learning_rate,
            source_signature=source_signature,
            source_config=source_config,
        )
        np.save(output_path, record, allow_pickle=True)
        print(f'Saved TD alpha={format_float(learning_rate)} to {output_path}')


def main(argv: list[str] | None = None) -> None:
    """
    Run TD simulation from command-line arguments.

    Args:
        argv (list[str] | None): Optional command-line arguments.
    """
    run_simulation(parse_args(argv))


if __name__ == '__main__':
    main()
