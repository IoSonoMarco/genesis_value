"""
Simulate the fixed-parameter PVL-Delta reference model in the shared IGT environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path.cwd()))

from models.agent import PVLDeltaModel, get_linear_choice_temperature
from scripts.simulation_genesis_igt_task_temporal_valence import load_igt_schedule
from utils import filepaths
from utils.simulation_config import load_simulation_config


DEFAULT_PVL_CONFIG_PATH = filepaths.ROOT / 'configs' / 'simulation_pvl_delta_igt_task.yaml'
DEFAULT_SHARED_CONFIG_PATH = filepaths.ROOT / 'configs' / 'simulation_genesis_igt_task_models.yaml'
MODEL_NAME = 'pvl_delta_healthy_reference'
MODEL_METADATA = {
    'name': MODEL_NAME,
    'family': 'reference',
    'attention': 'fixed',
    'model_type': 'pvl_delta',
}
REFERENCE_PUBLICATION = {
    'authors': 'Steingroever, Wetzels, and Wagenmakers',
    'year': 2013,
    'title': 'Validating the PVL-Delta model for the Iowa gambling task',
    'doi': '10.3389/fpsyg.2013.00898',
    'parameter_source': 'Figure 3',
}
DECK_NAMES = ('A', 'B', 'C', 'D')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse standalone PVL-Delta simulation arguments.

    Args:
        argv (list[str] | None): Optional CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run the fixed PVL-Delta IGT reference simulation.')
    parser.add_argument('--config', type=Path, default=DEFAULT_PVL_CONFIG_PATH)
    parser.add_argument('--shared-config', type=Path, default=DEFAULT_SHARED_CONFIG_PATH)
    return parser.parse_args(argv)


def build_model(parameter_config: dict[str, float]) -> PVLDeltaModel:
    """
    Build the fixed PVL-Delta model from its four-parameter YAML.

    Args:
        parameter_config (dict[str, float]): Parameters ``A``, ``w_loss``, ``a``, and ``c``.

    Returns:
        PVLDeltaModel: Zero-initialized four-deck learner.
    """
    return PVLDeltaModel(
        outcome_sensitivity=parameter_config['A'],
        loss_aversion=parameter_config['w_loss'],
        learning_rate=parameter_config['a'],
        response_consistency=parameter_config['c'],
    )


def run_agent(
    *,
    shared_config: dict[str, Any],
    parameter_config: dict[str, float],
    schedule: np.ndarray,
    agent_id: int,
) -> dict[str, Any]:
    """
    Run one stochastic fixed-parameter PVL-Delta agent.

    Args:
        shared_config (dict[str, Any]): Shared GENESIS IGT task and seed settings.
        parameter_config (dict[str, float]): Fixed PVL-Delta parameter vector.
        schedule (np.ndarray): Scaled payoff schedule with shape ``[100, 4]``.
        agent_id (int): Stable agent identifier shared with GENESIS.

    Returns:
        dict[str, Any]: Choice seed and complete trial history.
    """
    model = build_model(parameter_config)
    choice_seed = shared_config['base_random_seed'] + agent_id * 2 + 1
    choice_random_state = np.random.RandomState(choice_seed)
    choice_policy_config = shared_config['choice_policy']
    min_choice_temperature = choice_policy_config['min_choice_temperature']
    deck_draw_indices = dict.fromkeys(DECK_NAMES, 0)
    cumulative_reward = 0.0
    history = []

    for trial_id in range(shared_config['task']['n_trials']):
        warmup_temperature = get_linear_choice_temperature(
            trial_id=trial_id,
            max_choice_temperature=choice_policy_config['max_choice_temperature'],
            min_choice_temperature=min_choice_temperature,
            n_anneal_trials=choice_policy_config['n_anneal_trials'],
        )
        warmup_multiplier = min_choice_temperature / warmup_temperature
        response_sensitivity = model.base_response_sensitivity * warmup_multiplier
        action_values = {
            deck: float(model.expectancies[index])
            for index, deck in enumerate(DECK_NAMES)
        }
        probabilities = model.choice_probabilities(response_sensitivity)
        deck_index = int(choice_random_state.choice(len(DECK_NAMES), p=probabilities))
        chosen_deck = DECK_NAMES[deck_index]
        draw_index = deck_draw_indices[chosen_deck]
        reward = float(schedule[draw_index, deck_index])
        deck_draw_indices[chosen_deck] += 1
        cumulative_reward += reward
        subjective_utility = model.subjective_utility(reward)
        prediction_error = model.update(
            deck_index=deck_index,
            subjective_utility=subjective_utility,
        )
        updated_action_values = {
            deck: float(model.expectancies[index])
            for index, deck in enumerate(DECK_NAMES)
        }
        history.append({
            'agent_id': agent_id,
            'trial_id': trial_id,
            'chosen_deck': chosen_deck,
            'deck_draw_index': draw_index,
            'reward': reward,
            'cumulative_reward': cumulative_reward,
            'action_values': action_values,
            'updated_action_values': updated_action_values,
            'p_a': float(probabilities[0]),
            'p_b': float(probabilities[1]),
            'p_c': float(probabilities[2]),
            'p_d': float(probabilities[3]),
            'choice_temperature': 1.0 / response_sensitivity,
            'warmup_temperature': warmup_temperature,
            'warmup_multiplier': warmup_multiplier,
            'base_response_sensitivity': model.base_response_sensitivity,
            'response_sensitivity': response_sensitivity,
            'subjective_utility': subjective_utility,
            'prediction_error': prediction_error,
            'model_name': MODEL_NAME,
            'model_family': MODEL_METADATA['family'],
            'attention_condition': MODEL_METADATA['attention'],
        })

    return {
        'agent_id': agent_id,
        'choice_seed': choice_seed,
        'simulation_log': history,
    }


def main(args: argparse.Namespace) -> None:
    """
    Run and save the standalone PVL-Delta reference simulation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    parameter_config = load_simulation_config(args.config)
    shared_config = load_simulation_config(args.shared_config)
    schedule_path = filepaths.ROOT / shared_config['task']['schedule_path']
    schedule = load_igt_schedule(
        schedule_path=schedule_path,
        reward_scaling_factor=shared_config['task']['reward_scaling_factor'],
    )
    output_dir = filepaths.EXPERIMENTS_DIR / shared_config['experiment_name'] / MODEL_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / '001_fixed_parameters.npy'

    agent_runs = [
        run_agent(
            shared_config=shared_config,
            parameter_config=parameter_config,
            schedule=schedule,
            agent_id=agent_id,
        )
        for agent_id in range(shared_config['n_agents'])
    ]
    model = build_model(parameter_config)
    np.save(output_path, {
        'config': shared_config,
        'model': dict(MODEL_METADATA),
        'pvl_parameters': dict(parameter_config),
        'base_response_sensitivity': model.base_response_sensitivity,
        'warmup': {
            'schedule': 'matched_genesis_choice_temperature_multiplier',
            'max_choice_temperature': shared_config['choice_policy']['max_choice_temperature'],
            'min_choice_temperature': shared_config['choice_policy']['min_choice_temperature'],
            'n_anneal_trials': shared_config['choice_policy']['n_anneal_trials'],
        },
        'reference_publication': dict(REFERENCE_PUBLICATION),
        'source_schedule_path': str(schedule_path),
        'reward_scaling_factor': shared_config['task']['reward_scaling_factor'],
        'agent_runs': agent_runs,
    }, allow_pickle=True)


if __name__ == '__main__':
    main(parse_args())
