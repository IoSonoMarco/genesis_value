"""
Simulate the GENESIS reminder-bandit task with episodic sampling.

The script separates memory-based valuation from action selection. A frozen
stimulus CVAE encodes Shape3D images, a frozen reward model encodes scalar
outcomes, and the episodic sampling module stores those encoded experiences.
On each trial, the agent builds a memory cue from temporal context alone or
from temporal context plus a reminder image. The episodic module estimates
LEFT and RIGHT values from sampled memories, and the policy turns those values
into a stochastic arm choice.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.cwd()))
warnings.simplefilter('ignore')

from dataclasses import dataclass
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from configs import shape3d_feature_config_genesis
from data import Shapes3DInput, get_Shapes3D_datasets
from models import (
    AEModelConfig,
    CVAEFilmMultipleFeatures,
    LinearDecoder,
    load_linear_decoder,
)
from models.agent import SimpleTwoArmedBanditPolicy
from models.memory import CompressedEpisode, EpisodicSamplingModuleTwoArmedBandit
from models.memory.utils import generate_temporal_embeddings
from utils import constants, filepaths
from utils.simulation_config import get_replacement_output_dir, load_simulation_config


DEFAULT_SIMULATION_CONFIG_PATH = filepaths.ROOT / 'configs' / 'simulation_genesis_reward_bandit_task.yaml'
DEVICE = torch.device('cuda')
FEATURE_CONFIG = shape3d_feature_config_genesis.feature_config
FACTOR_COLUMNS = (
    'object_shape_global',
    'object_color_global',
    'floor_color_global',
    'wall_color_global',
)
N_RANDOM_WARMUP_TRIALS = 20
BASE_RANDOM_SEED = 44
SAVE_METADATA = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse simulation grid arguments.

    Args:
        argv (list[str] | None): Optional CLI argument list.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Run GENESIS reminder-bandit simulation grid.')
    parser.add_argument('--config', type=Path, default=DEFAULT_SIMULATION_CONFIG_PATH)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--n-jobs', type=int, default=-1)

    return parser.parse_args(argv)


def resolve_simulation_config(args: argparse.Namespace) -> dict[str, Any]:
    """
    Resolve simulation metadata and grid values from YAML and CLI overrides.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict[str, Any]: Experiment name and resolved simulation grid.
    """
    return load_simulation_config(args.config)


def resolve_simulation_grid(args: argparse.Namespace) -> dict[str, list[Any]]:
    """
    Resolve the simulation grid from YAML with optional CLI overrides.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict[str, list[Any]]: Simulation grid keyed by parameter name.
    """
    return resolve_simulation_config(args)['grid']


def expand_simulation_configs(selection_config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Expand the simulation grid into per-run configs.

    Args:
        selection_config (dict[str, Any]): YAML-style config with a ``grid`` mapping.

    Returns:
        list[dict[str, Any]]: One config dictionary per grid cell.
    """
    grid = selection_config['grid']
    replacement_mode = selection_config.get('sample_replacement')
    if replacement_mode is True and 'sampling_temperature' in selection_config:
        grid = {
            'sampling_budget': [selection_config.get('replacement_sampling_budget', 20)],
            'temporal_weight': grid['temporal_weight'],
            'sampling_temperature': selection_config['sampling_temperature'],
        }
    grid_keys = list(grid.keys())

    configs = []
    for values in product(*[grid[key] for key in grid_keys]):
        config = dict(zip(grid_keys, values))
        if replacement_mode is not None:
            config['sample_replacement'] = replacement_mode

        if 'normalize_query_key_components' in selection_config:
            config['normalize_query_key_components'] = selection_config[
                'normalize_query_key_components'
            ]

        if 'query_key_normalization' in selection_config:
            config['query_key_normalization'] = selection_config[
                'query_key_normalization'
            ]

        configs.append(config)

    return configs


@dataclass
class BanditTaskConfig:
    """
    Configure the Shape3D reminder-bandit schedule.

    Attributes:
        n_choice_trials (int): Number of choice trials and unique semantic image classes.
        n_probes (int): Number of reminder trials inserted into the schedule.
        probe_min_lag (int): Minimum target-to-reminder trial distance.
        probe_max_lag (int): Maximum target-to-reminder trial distance.
        p_win_low (float): Lower bound of the drifting LEFT win probability.
        p_win_high (float): Upper bound of the drifting LEFT win probability.
        p_win_start_high (float): Initial high probability before complement sampling.
        random_walk_sigma (float): Standard deviation of probability drift.
        random_seed (int): Seed controlling schedule construction.
        first_reminder_trial (int): minimum number of first trials without a reminder.
    """

    n_choice_trials: int = 130
    n_probes: int = 24
    probe_min_lag: int = 20
    probe_max_lag: int = 70
    p_win_low: float = 0.3
    p_win_high: float = 0.7
    p_win_start_high: float = 0.6
    random_walk_sigma: float = 0.025
    random_seed: int = 44
    first_reminder_trial: int = N_RANDOM_WARMUP_TRIALS

    def __post_init__(self) -> None:
        """
        Validate probability and probe-lag relationships.

        Raises:
            ValueError: If the configuration cannot define the task.
        """
        if self.probe_min_lag < 0:
            raise ValueError('probe_min_lag must be non-negative.')
        if self.probe_max_lag < self.probe_min_lag:
            raise ValueError('probe_max_lag must be greater than or equal to probe_min_lag.')
        if not np.isclose(self.p_win_low + self.p_win_high, 1.0):
            raise ValueError('Bandit probability bounds must be complementary around 0.5.')
        if not self.p_win_low < self.p_win_high:
            raise ValueError('p_win_low must be smaller than p_win_high.')
        if not self.p_win_low <= self.p_win_start_high <= self.p_win_high:
            raise ValueError('p_win_start_high must lie within the probability bounds.')
        if self.first_reminder_trial < 0:
            raise ValueError('first_reminder_trial must be non-negative.')
        if self.first_reminder_trial >= self.n_choice_trials:
            raise ValueError('first_reminder_trial must be smaller than n_choice_trials.')


@dataclass
class EnvironmentSimulationCache:
    """
    Store deterministic environment inputs reused by every agent and grid cell.

    Attributes:
        schedule (list[dict[str, Any]]): Fixed trial schedule for one environment.
        stimulus_embeddings (dict[int, torch.Tensor]): Stimulus embeddings keyed by HDF5 image index.
        reward_embeddings (dict[float, torch.Tensor]): Reward embeddings keyed by binary reward value.
    """

    schedule: list[dict[str, Any]]
    stimulus_embeddings: dict[int, torch.Tensor]
    reward_embeddings: dict[float, torch.Tensor]


def build_task_trials_schedule(
    data_table: pd.DataFrame,
    config: BanditTaskConfig | None = None,
) -> list[dict]:
    """
    Build a simplified Shape3D reminder-bandit task.

    The returned schedule has one dictionary per choice trial. Each trial always
    contains a choice image and drifting LEFT/RIGHT reward probabilities. Some
    trials also contain a same-class reminder image shown before the choice.

    Args:
        data_table (pd.DataFrame): Shape3D metadata with ``h5_idx`` and semantic factor columns.
        config (BanditTaskConfig | None): Task schedule configuration.

    Returns:
        list[dict]: Choice-trial dictionaries in chronological order.
    """
    config = config or BanditTaskConfig()
    rng = np.random.default_rng(config.random_seed)

    # Group images into semantic classes, so each trial can use one choice image
    # and reserve one same-class image as a possible future reminder.
    groups = []
    required_columns = ['h5_idx', *FACTOR_COLUMNS]

    grouped = data_table.loc[:, required_columns].groupby(
        list(FACTOR_COLUMNS),
        sort=True,
    )

    for factor_values, group in grouped:
        if len(group) < 2:
            continue

        factors = dict(zip(FACTOR_COLUMNS, map(int, factor_values)))
        h5_indices = np.sort(group['h5_idx'].to_numpy(dtype=np.int64))
        groups.append((factors, h5_indices))

    # Select one unique semantic class per choice trial.
    selected_group_ids = rng.choice(
        len(groups),
        size=config.n_choice_trials,
        replace=False,
    )

    trials = []

    for trial_id, group_id in enumerate(selected_group_ids):
        factors, h5_indices = groups[int(group_id)]

        choice_h5_idx, reminder_h5_idx = rng.choice(
            h5_indices,
            size=2,
            replace=False,
        )

        choice_image = {
            'class_id': int(trial_id),
            'h5_idx': int(choice_h5_idx),
            **factors,
        }

        heldout_reminder_image = {
            'class_id': int(trial_id),
            'h5_idx': int(reminder_h5_idx),
            **factors,
        }

        trials.append({
            'trial_id': int(trial_id),
            'choice_image': choice_image,
            'heldout_reminder_image': heldout_reminder_image,
            'reminder_image': None,
        })

    # Generate a reflected random walk for LEFT reward probability. RIGHT is
    # always the complementary probability.
    p_win_left = np.empty(config.n_choice_trials, dtype=np.float64)

    p_win_left[0] = (
        config.p_win_start_high
        if rng.random() < 0.5
        else 1.0 - config.p_win_start_high
    )

    probability_width = config.p_win_high - config.p_win_low
    reflection_period = 2 * probability_width

    for trial_id in range(1, config.n_choice_trials):
        candidate = p_win_left[trial_id - 1] + rng.normal(
            loc=0.0,
            scale=config.random_walk_sigma,
        )

        offset = (
            candidate % reflection_period
            - config.p_win_low % reflection_period
        ) % reflection_period

        p_win_left[trial_id] = (
            config.p_win_low + offset
            if offset <= probability_width
            else config.p_win_high - (offset - probability_width)
        )

    for trial_id, trial in enumerate(trials):
        trial['p_win_left'] = float(p_win_left[trial_id])
        trial['p_win_right'] = float(1.0 - p_win_left[trial_id])

    # Assign reminder images to later trials. A reminder is a held-out image
    # from a previous trial's semantic class, so it cues that earlier episode.
    if config.n_probes:
        
        first_valid_reminder_trial = max(
            config.probe_min_lag + 1,
            config.first_reminder_trial,
        )

        valid_reminder_trial_ids = np.arange(
            first_valid_reminder_trial,
            config.n_choice_trials,
            dtype=np.int64,
        )

        if len(valid_reminder_trial_ids) < config.n_probes:
            raise ValueError(
                'Not enough valid reminder trials after applying '
                f'first_reminder_trial={config.first_reminder_trial}. '
                f'Available={len(valid_reminder_trial_ids)}, requested={config.n_probes}.'
            )

        reminder_trial_ids = np.sort(rng.choice(
            valid_reminder_trial_ids,
            size=config.n_probes,
            replace=False,
        ))

        used_target_trials = set()

        for reminder_trial_id in reminder_trial_ids:
            candidate_targets = np.arange(
                max(0, int(reminder_trial_id) - config.probe_max_lag),
                int(reminder_trial_id) - config.probe_min_lag + 1,
                dtype=np.int64,
            )

            candidate_targets = [
                int(target_trial_id)
                for target_trial_id in rng.permutation(candidate_targets)
                if int(target_trial_id) not in used_target_trials
            ]

            target_trial_id = candidate_targets[0]
            used_target_trials.add(target_trial_id)

            target_trial = trials[target_trial_id]

            reminder_image = dict(target_trial['heldout_reminder_image'])
            reminder_image['target_trial_id'] = int(target_trial_id)
            reminder_image['lag'] = int(reminder_trial_id - target_trial_id)

            trials[int(reminder_trial_id)]['reminder_image'] = reminder_image

    for trial in trials:
        trial.pop('heldout_reminder_image')

    return trials


class CortexEncoder:
    """
    Encode task stimuli and scalar rewards with frozen cortical models.

    The stimulus model supplies a compact image-plus-factor representation.
    The reward model supplies the latent reward representation stored in
    episodic memory and later decoded back into scalar action values.
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

    def load_stimulus_model(
        self,
        stimulus_model_dir: Path,
    ) -> CVAEFilmMultipleFeatures:
        """
        Load the frozen Shape3D CVAE.

        Args:
            stimulus_model_dir (Path): Trained CVAE output directory.

        Returns:
            CVAEFilmMultipleFeatures: Frozen stimulus model.
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

    def load_reward_model(
        self,
        reward_model_dir: Path,
    ) -> LinearDecoder:
        """
        Load the frozen scalar reward encoder.

        Args:
            reward_model_dir (Path): Reward model output directory.

        Returns:
            LinearDecoder: Frozen reward model.
        """
        model = load_linear_decoder(
            checkpoint_path=next(reward_model_dir.rglob('weights.pt')),
            device='cpu',
        )
        for parameter in model.parameters():
            parameter.requires_grad = False

        return model

    @torch.no_grad()
    def encode_stimulus(
        self,
        pixel_values: torch.Tensor,
        factors: object,
    ) -> torch.Tensor:
        """
        Encode one Shape3D image and its semantic factors.

        Args:
            pixel_values (torch.Tensor): Image tensor with shape ``[C, H, W]``.
            factors (object): Shape3D factor object with one scalar tensor per configured feature.

        Returns:
            torch.Tensor: Concatenated stimulus embedding with shape ``[D]``.
        """
        pixel_values = pixel_values.to(self.device)[None]
        features = {
            feature_name: getattr(factors, feature_name)[None].to(self.device)
            for feature_name in self.feature_config
        }

        mu, _ = self.stimulus_model.encode(pixel_values)
        feature_embeddings = self.stimulus_model.embed_features(features)

        stimulus_embedding = torch.hstack([
            mu.flatten(1).cpu()[0],
            feature_embeddings.flatten(1).cpu()[0],
        ])

        return stimulus_embedding

    @torch.no_grad()
    def encode_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """
        Encode one scalar reward for episodic storage.

        Args:
            reward (torch.Tensor): Scalar reward tensor.

        Returns:
            torch.Tensor: Reward embedding with shape ``[D]``.
        """
        reward = reward.to(self.device)[None]
        reward_embedding = self.reward_model.encode(reward)
        reward_embedding = reward_embedding.cpu()[0]

        return reward_embedding


def from_h5idx_to_batch(
    h5_idx: int,
    data_table: pd.DataFrame,
    dataset: torch.utils.data.Dataset,
) -> Shapes3DInput:
    """
    Load one Shape3D item by HDF5 image index.

    Args:
        h5_idx (int): Shape3D HDF5 image identifier.
        data_table (pd.DataFrame): Dataset metadata containing ``h5_idx``.
        dataset (torch.utils.data.Dataset): Shape3D dataset indexed like ``data_table``.

    Returns:
        Shapes3DInput: One-image batch container used by the cortical encoder.
    """
    idx = data_table.index[data_table.h5_idx == h5_idx][0]
    data = dataset[idx]

    batch = Shapes3DInput(
        pixel_values=data[0],
        factors=data[1],
        rewards=torch.tensor(0).float(),
    )
    return batch


def build_environment_cache(
    *,
    schedule: list[dict[str, Any]],
    test_ds: Any,
    cortex_encoder: CortexEncoder,
) -> EnvironmentSimulationCache:
    """
    Encode deterministic environment inputs once for reuse across simulations.

    Args:
        schedule (list[dict[str, Any]]): Fixed trial schedule for one environment.
        test_ds (Any): Shape3D test dataset used to load scheduled images.
        cortex_encoder (CortexEncoder): Frozen stimulus and reward encoder.

    Returns:
        EnvironmentSimulationCache: Cached schedule, stimulus embeddings, and reward embeddings.
    """
    stimulus_embeddings = {}

    for trial in schedule:
        images = [trial['choice_image']]
        if trial['reminder_image'] is not None:
            images.append(trial['reminder_image'])

        for image in images:
            h5_idx = image['h5_idx']
            if h5_idx in stimulus_embeddings:
                continue

            batch = from_h5idx_to_batch(
                h5_idx,
                test_ds.data_table,
                test_ds,
            )
            stimulus_embeddings[h5_idx] = cortex_encoder.encode_stimulus(
                batch.pixel_values,
                batch.factors,
            )

    reward_embeddings = {}
    for reward_value in [-1.0, 1.0]:
        reward = torch.tensor(reward_value, dtype=torch.float32)
        reward_embeddings[reward_value] = cortex_encoder.encode_reward(reward)

    return EnvironmentSimulationCache(
        schedule=schedule,
        stimulus_embeddings=stimulus_embeddings,
        reward_embeddings=reward_embeddings,
    )


def build_environment_caches(
    *,
    n_environments: int,
    test_ds: Any,
    cortex_encoder: CortexEncoder,
) -> list[EnvironmentSimulationCache]:
    """
    Build deterministic schedules and embedding caches for every environment.

    Args:
        n_environments (int): Number of independent environments.
        test_ds (Any): Shape3D test dataset used to load scheduled images.
        cortex_encoder (CortexEncoder): Frozen stimulus and reward encoder.

    Returns:
        list[EnvironmentSimulationCache]: One cache per environment in identifier order.
    """
    environment_caches = []

    for environment_id in tqdm(range(n_environments), desc='Caching environments'):
        environment_seed = BASE_RANDOM_SEED + environment_id
        schedule = build_task_trials_schedule(
            test_ds.data_table,
            BanditTaskConfig(random_seed=environment_seed),
        )
        environment_caches.append(build_environment_cache(
            schedule=schedule,
            test_ds=test_ds,
            cortex_encoder=cortex_encoder,
        ))

    return environment_caches


def summarize_reminder_retrieval(
    *,
    metadata: dict[str, dict[str, Any]],
    target_trial_id: int,
    target_arm: str,
) -> dict[str, int | float | bool]:
    """
    Summarize retrieval of the episode targeted by a reminder.

    Args:
        metadata (dict[str, dict[str, Any]]): Per-arm retrieval metadata.
        target_trial_id (int): Memory index of the reminded episode.
        target_arm (str): Arm chosen in the reminded episode.

    Returns:
        dict[str, int | float | bool]: Compact target accessibility and capture fields.
    """
    arm_metadata = metadata[target_arm]
    candidate_indices = arm_metadata['candidate_indices']
    target_position = candidate_indices.index(target_trial_id)
    target_score = arm_metadata['qk_scores'][target_position]
    target_rank = int(np.sum(arm_metadata['qk_scores'] > target_score) + 1)

    return {
        'reminder_target_candidate_count': len(candidate_indices),
        'reminder_target_sampling_probability': float(arm_metadata['probs'][target_position]),
        'reminder_target_similarity_rank': target_rank,
        'reminder_target_sampled': target_trial_id in arm_metadata['sampled_episode_indices'],
    }


def main(
    *,
    config: dict[str, Any],
    test_ds: Any,
    cortex_encoder: CortexEncoder,
    schedule: list[dict[str, Any]],
    agent_seed: int,
    environment_cache: EnvironmentSimulationCache | None = None,
) -> list[dict[str, Any]]:
    """
    Run the reminder-bandit simulation.

    Normal trials query memory with temporal context only. Reminder trials add
    an encoded same-class reminder image to the temporal cue. The reminder can
    bias behavior only by changing which episodes are sampled before action
    values are decoded and passed to the policy.

    Args:
        config (dict[str, Any]): One simulation-grid parameter combination.
        test_ds (Any): Shape3D test dataset used to load scheduled images.
        cortex_encoder (CortexEncoder): Frozen stimulus and reward encoder.
        schedule (list[dict[str, Any]]): Fixed environment trial schedule.
        agent_seed (int): Seed controlling agent choices, retrieval, and reward sampling.
        environment_cache (EnvironmentSimulationCache | None): Optional deterministic embedding cache.

    Returns:
        list[dict[str, Any]]: Trial-level behavioral records for one agent.
    """
    
    random_state = np.random.RandomState(agent_seed)
    rng = np.random.default_rng(agent_seed)

    episodic_sampler = EpisodicSamplingModuleTwoArmedBandit(
        reward_decoder=cortex_encoder.reward_model.decoder,
        weights=dict(
            stimulus_embedding=1 - config['temporal_weight'],
            reward_embedding=0.0,
            time_embedding=config['temporal_weight'],
        ),
        k_samples=config['sampling_budget'],
        sample_replacement=config['sample_replacement'],
        sampling_temperature=config.get('sampling_temperature', 1.0),
        normalize_query_key_components=config.get('normalize_query_key_components', False),
        query_key_normalization=config.get('query_key_normalization'),
        device=DEVICE,
        random_state=random_state,
    )

    policy = SimpleTwoArmedBanditPolicy(beta_choice=2.0)

    data_table = test_ds.data_table.copy() if environment_cache is None else None

    # Temporal context is available on every trial and always participates in
    # episodic retrieval, even when no reminder image is shown.
    time_embeds = generate_temporal_embeddings(
        [np.pi / 100] * len(schedule),
        dim=64,
        random_state=random_state,
    )
    time_embeds = torch.tensor(time_embeds).float()

    simulation_log = []

    for trial in schedule:
        trial_id = trial['trial_id']
        time_embedding = time_embeds[trial_id]
        
        if trial_id < N_RANDOM_WARMUP_TRIALS:
            is_warmup_trial = True
        else:
            is_warmup_trial = False

        # 1. Build the current memory query.
        if trial['reminder_image'] is not None:
            if environment_cache is None:
                reminder_batch = from_h5idx_to_batch(
                    trial['reminder_image']['h5_idx'],
                    data_table,
                    test_ds,
                )
                reminder_stimulus_embedding = cortex_encoder.encode_stimulus(
                    reminder_batch.pixel_values,
                    reminder_batch.factors,
                )
            else:
                reminder_stimulus_embedding = environment_cache.stimulus_embeddings[
                    trial['reminder_image']['h5_idx']
                ]

            query = CompressedEpisode(
                stimulus_embedding=reminder_stimulus_embedding,
                time_embedding=time_embedding,
            )

            reminder_trial_id = trial['reminder_image']['target_trial_id']
            reminder_lag = trial['reminder_image']['lag']
            
            # 2. Estimate action values by sampling memories separately for each arm.
            action_value, metadata = episodic_sampler.estimate_action_values(
                query=query,
                return_metadata=True
            )

            reminder_target = simulation_log[reminder_trial_id]
            reminder_target_chosen_arm = reminder_target['chosen_arm']
            reminder_target_reward = reminder_target['reward']
            reminder_retrieval = summarize_reminder_retrieval(
                metadata=metadata,
                target_trial_id=reminder_trial_id,
                target_arm=reminder_target_chosen_arm,
            )
            
        else:
            query = CompressedEpisode(time_embedding=time_embedding)
            reminder_trial_id = None
            reminder_lag = None
            reminder_target_chosen_arm = None
            reminder_target_reward = None
            reminder_retrieval = {
                'reminder_target_candidate_count': None,
                'reminder_target_sampling_probability': None,
                'reminder_target_similarity_rank': None,
                'reminder_target_sampled': None,
            }
            
            if is_warmup_trial: 
                # exploration
                chosen_arm = "RIGHT" if random_state.random() < 0.5 else "LEFT"
                action_value = {"LEFT": np.nan, "RIGHT": np.nan}
                metadata = None
            else:
                # 2. Estimate action values by sampling memories separately for each arm.
                action_value, metadata = episodic_sampler.estimate_action_values(
                    query=query,
                    weights={'time_embedding': 1.0}, # If no reminder is used, sampling is anchored to recency
                    return_metadata=True
                )

        if is_warmup_trial:
            # exploration default
            policy_output = {
                "chosen_arm": chosen_arm,
                "p_right": np.nan,
                "logit_right": np.nan,
            }
        else:
            # 3. Convert action values into a stochastic LEFT/RIGHT policy choice.
            policy_output = policy.choose_arm(
                action_value,
                random_state=random_state,
            )
            chosen_arm = policy_output['chosen_arm']

        # 4. Sample the environment reward from the drifting bandit probability.
        p_win = (
            trial['p_win_left']
            if chosen_arm == 'LEFT'
            else trial['p_win_right']
        )

        reward = torch.tensor(
            1.0 if rng.random() < p_win else -1.0,
            dtype=torch.float32,
        )

        # 5. Reveal the choice image and encode the full episode into memory.
        if environment_cache is None:
            choice_batch = from_h5idx_to_batch(
                trial['choice_image']['h5_idx'],
                data_table,
                test_ds,
            )
            stimulus_embedding = cortex_encoder.encode_stimulus(
                choice_batch.pixel_values,
                choice_batch.factors,
            )
            reward_embedding = cortex_encoder.encode_reward(reward)
        else:
            stimulus_embedding = environment_cache.stimulus_embeddings[
                trial['choice_image']['h5_idx']
            ]
            reward_embedding = environment_cache.reward_embeddings[float(reward)]

        episode = dict(
            hidden_states=dict(
                stimulus_embedding=stimulus_embedding,
                reward_embedding=reward_embedding,
                time_embedding=time_embedding,
            ),
            chosen_arm=chosen_arm,
            reward=reward,
            trial_metadata=trial,
        )

        episodic_sampler.store_episode(episode)

        # 6. Store a compact behavioral record for later analysis.
        simulation_log.append(dict(
            trial_id=trial_id,
            is_reminder_trial=trial['reminder_image'] is not None,
            reminder_target_trial_id=reminder_trial_id,
            reminder_lag=reminder_lag,
            reminder_target_chosen_arm=reminder_target_chosen_arm,
            reminder_target_reward=reminder_target_reward,
            **reminder_retrieval,
            chosen_arm=chosen_arm,
            reward=float(reward),
            p_win=float(p_win),
            p_win_left=float(trial['p_win_left']),
            action_value_left=float(action_value['LEFT']),
            action_value_right=float(action_value['RIGHT']),
            p_right=policy_output['p_right'],
            logit_right=policy_output['logit_right'],
            choice_class_id=trial['choice_image']['class_id'],
            choice_h5_idx=trial['choice_image']['h5_idx'],
            metadata=metadata if SAVE_METADATA else None
        ))
        
    return simulation_log


def config_to_filename_suffix(config: dict[str, Any]) -> str:
    """
    Build the parameter-specific portion of a simulation output filename.

    Args:
        config (dict[str, Any]): One simulation-grid parameter combination.

    Returns:
        str: Parameter-specific output filename suffix.
    """
    sb = str(config['sampling_budget'])
    tw = str(config['temporal_weight']).replace('.', 'p')
    repl = str(config['sample_replacement']).lower()
    normalization = config.get('query_key_normalization')

    if normalization is None:
        normalization = 'rank' if config.get('normalize_query_key_components', False) else 'legacy'

    sampling_temperature = config.get('sampling_temperature')
    temperature_suffix = ''
    if config['sample_replacement'] and sampling_temperature is not None:
        temperature = str(sampling_temperature).replace('.', 'p')
        temperature_suffix = f'sampling_temperature-{temperature}_'

    return (
        f'sampling_budget-{sb}_temporal_weight-{tw}_sample_replacement-{repl}_'
        f'{temperature_suffix}query_key_normalization-{normalization}.npy'
    )


def config_to_filename(config: dict[str, Any], config_index: int) -> str:
    """
    Build the indexed filename for one simulation-grid output.

    Args:
        config (dict[str, Any]): One simulation-grid parameter combination.
        config_index (int): One-based grid-cell index.

    Returns:
        str: Indexed simulation output filename.
    """
    suffix = config_to_filename_suffix(config)

    return f'{config_index:03d}_{suffix}'


def grid_cell_output_exists(*, output_dir: Path, config: dict[str, Any]) -> bool:
    """
    Return whether a simulation output already exists for one grid cell.

    Args:
        output_dir (Path): Directory containing simulation-grid outputs.
        config (dict[str, Any]): One simulation-grid parameter combination.

    Returns:
        bool: Whether an output with the same grid parameters exists.
    """
    return any(output_dir.glob(f'*_{config_to_filename_suffix(config)}'))
    
    
def run_simulation_environment(
    *,
    environment_id: int,
    config: dict[str, Any],
    test_ds: Any,
    cortex_encoder: CortexEncoder,
    n_agent_simulations: int,
    environment_cache: EnvironmentSimulationCache | None = None,
) -> dict[str, Any]:
    """
    Run one environment and all nested agent simulations.

    Args:
        environment_id (int): Environment index in the simulation grid cell.
        config (dict[str, Any]): One simulation-grid parameter combination.
        test_ds (Any): Shape3D test dataset used to load scheduled images.
        cortex_encoder (CortexEncoder): Frozen stimulus and reward encoder.
        n_agent_simulations (int): Number of agent simulations to run in the environment.
        environment_cache (EnvironmentSimulationCache | None): Optional deterministic schedule and embedding cache.

    Returns:
        dict[str, Any]: Environment record containing schedule and agent runs.
    """
    environment_seed = BASE_RANDOM_SEED + environment_id
    if environment_cache is None:
        schedule = build_task_trials_schedule(
            test_ds.data_table,
            BanditTaskConfig(random_seed=environment_seed),
        )
    else:
        schedule = environment_cache.schedule
    environment = dict(
        environment_id=environment_id,
        environment_seed=environment_seed,
        schedule=schedule,
        agent_runs=[],
    )

    for agent_simulation_id in range(n_agent_simulations):
        agent_seed = (
            BASE_RANDOM_SEED
            + environment_id * n_agent_simulations
            + agent_simulation_id
        )
        simulation_log = main(
            config=config,
            test_ds=test_ds,
            cortex_encoder=cortex_encoder,
            schedule=schedule,
            agent_seed=agent_seed,
            environment_cache=environment_cache,
        )
        environment['agent_runs'].append(dict(
            agent_simulation_id=agent_simulation_id,
            agent_seed=agent_seed,
            simulation_log=simulation_log,
        ))

    return environment


def run_simulation_grid(args: argparse.Namespace) -> None:
    """
    Run every simulation config in the resolved grid.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    
    # Load data and frozen cortical models used by GENESIS.
    dataset_start = time.perf_counter()
    _, test_ds, _ = get_Shapes3D_datasets(
        file_path=filepaths.SHAPES3D_FILE_PATH,
        **FEATURE_CONFIG,
    )
    print(f'Loaded HDF5 datasets in {time.perf_counter() - dataset_start:.1f}s')
    
    cortex_encoder = CortexEncoder(
        stimulus_model_dir=filepaths.OUTPUT_DIR / 'cvae' / 'shape3d_genesis_2',
        reward_model_dir=filepaths.OUTPUT_DIR / 'reward_linear_decoder',
        device=DEVICE,
    )
    
    ### Run
    
    simulation_config = resolve_simulation_config(args)
    n_environments = simulation_config['n_environments']
    n_agent_simulations = simulation_config['n_agent_simulations']
    configs = expand_simulation_configs(simulation_config)
    environment_caches = None
    total_cell_simulation_start = None
    
    print(f'Running {len(configs)} simulation config(s).')
    
    for config_index, config in enumerate(configs, start=1):
        savepath = get_replacement_output_dir(
            root_dir=filepaths.EXPERIMENTS_DIR,
            output_name=simulation_config['experiment_name'],
            sample_replacement=config['sample_replacement'],
        )
        savepath.mkdir(parents=True, exist_ok=True)
        if grid_cell_output_exists(output_dir=savepath, config=config):
            print(f'Skipping existing simulation {config_index}/{len(configs)} | {config}')
            continue

        if environment_caches is None:
            cache_start = time.perf_counter()
            environment_caches = build_environment_caches(
                n_environments=n_environments,
                test_ds=test_ds,
                cortex_encoder=cortex_encoder,
            )
            print(f'Built environment cache in {time.perf_counter() - cache_start:.1f}s')

        cell_start = time.perf_counter()
        if total_cell_simulation_start is None:
            total_cell_simulation_start = cell_start

        print(f'Simulation {config_index}/{len(configs)} | {config}')
        
        temp = dict(config=config, environments=[])

        for environment_id, environment_cache in enumerate(environment_caches):
            environment = run_simulation_environment(
                environment_id=environment_id,
                config=config,
                test_ds=test_ds,
                cortex_encoder=cortex_encoder,
                n_agent_simulations=n_agent_simulations,
                environment_cache=environment_cache,
            )
            temp['environments'].append(environment)
    
        np.save(
            savepath/config_to_filename(config, config_index),
            temp,
            allow_pickle=True
        )
        print(f'Completed simulation {config_index}/{len(configs)} in {time.perf_counter() - cell_start:.1f}s')

    total_cell_simulation_time = (
        0.0
        if total_cell_simulation_start is None
        else time.perf_counter() - total_cell_simulation_start
    )
    print(f'Total cell simulation time: {total_cell_simulation_time:.1f}s')
    print("Done!")


def run_simulation_grid_parallel(args: argparse.Namespace) -> None:
    """
    Run every simulation config with environments parallelized by joblib.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    
    # Load data and frozen cortical models used by GENESIS.
    dataset_start = time.perf_counter()
    _, test_ds, _ = get_Shapes3D_datasets(
        file_path=filepaths.SHAPES3D_FILE_PATH,
        **FEATURE_CONFIG,
    )
    print(f'Loaded HDF5 datasets in {time.perf_counter() - dataset_start:.1f}s')
    
    cortex_encoder = CortexEncoder(
        stimulus_model_dir=filepaths.OUTPUT_DIR / 'cvae' / 'shape3d_genesis_2',
        reward_model_dir=filepaths.OUTPUT_DIR / 'reward_linear_decoder',
        device=DEVICE,
    )
    
    ### Run
    
    simulation_config = resolve_simulation_config(args)
    n_environments = simulation_config['n_environments']
    n_agent_simulations = simulation_config['n_agent_simulations']
    configs = expand_simulation_configs(simulation_config)
    environment_caches = None
    total_cell_simulation_start = None
    
    print(f'Running {len(configs)} simulation config(s).')
    
    for config_index, config in enumerate(configs, start=1):
        savepath = get_replacement_output_dir(
            root_dir=filepaths.EXPERIMENTS_DIR,
            output_name=simulation_config['experiment_name'],
            sample_replacement=config['sample_replacement'],
        )
        savepath.mkdir(parents=True, exist_ok=True)
        if grid_cell_output_exists(output_dir=savepath, config=config):
            print(f'Skipping existing simulation {config_index}/{len(configs)} | {config}')
            continue

        if environment_caches is None:
            cache_start = time.perf_counter()
            environment_caches = build_environment_caches(
                n_environments=n_environments,
                test_ds=test_ds,
                cortex_encoder=cortex_encoder,
            )
            print(f'Built environment cache in {time.perf_counter() - cache_start:.1f}s')

        cell_start = time.perf_counter()
        if total_cell_simulation_start is None:
            total_cell_simulation_start = cell_start

        print(f'Simulation {config_index}/{len(configs)} | {config}')
        
        environments = Parallel(n_jobs=args.n_jobs, prefer='threads')(
            delayed(run_simulation_environment)(
                environment_id=environment_id,
                config=config,
                test_ds=test_ds,
                cortex_encoder=cortex_encoder,
                n_agent_simulations=n_agent_simulations,
                environment_cache=environment_cache,
            )
            for environment_id, environment_cache in enumerate(environment_caches)
        )

        temp = dict(config=config, environments=environments)
    
        np.save(
            savepath/config_to_filename(config, config_index),
            temp,
            allow_pickle=True
        )
        print(f'Completed simulation {config_index}/{len(configs)} in {time.perf_counter() - cell_start:.1f}s')

    total_cell_simulation_time = (
        0.0
        if total_cell_simulation_start is None
        else time.perf_counter() - total_cell_simulation_start
    )
    print(f'Total cell simulation time: {total_cell_simulation_time:.1f}s')
    print("Done!")


if __name__ == '__main__':
    parsed_args = parse_args()
    if parsed_args.parallel:
        run_simulation_grid_parallel(parsed_args)
    else:
        run_simulation_grid(parsed_args)
