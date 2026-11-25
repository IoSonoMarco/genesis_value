from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import numpy as np
import torch
import torch.nn.functional as F

from models.utils import calibrated_softmax, softmax
from models.memory.utils import generate_temporal_embeddings


@dataclass
class CompressedEpisode:
    """
    Compact representation of an episode or memory cue.

    A compressed episode can hold stimulus, reward, temporal information, or any
    subset of these components. Stored memories use it as a key/value container,
    while current trial cues use it as a retrieval query.

    Attributes:
        stimulus_embedding (torch.Tensor | None): Stimulus representation with shape ``[D]``.
        reward_embedding (torch.Tensor | None): Reward representation with shape ``[D]``.
        time_embedding (torch.Tensor | None): Temporal-context representation with shape ``[D]``.
    """

    stimulus_embedding: torch.Tensor | None = None
    reward_embedding: torch.Tensor | None = None
    time_embedding: torch.Tensor | None = None

    def get_similarity(self, key: CompressedEpisode) -> dict[str, float]:
        """
        Compare this cue against a stored memory key.

        Only components present in both objects are compared. Each available
        component contributes one cosine-similarity score, so the caller can
        decide how to weight stimulus, reward, and temporal matches.

        Args:
            key (CompressedEpisode): Stored memory key to compare against.

        Returns:
            dict[str, float]: Cosine similarity by compared component.
        """
        scores = {}

        for field_name, query_value in self.__dict__.items():
            if query_value is None:
                continue

            key_value = getattr(key, field_name)

            if key_value is None:
                continue

            score = F.cosine_similarity(
                query_value.flatten(),
                key_value.flatten(),
                dim=0,
            )

            scores[field_name] = float(score)

        return scores

###

class EmbeddingWeight(TypedDict, total=False):
    """
    Scalar values estimated for the two bandit arms.
    """

    stimulus_embedding: float 
    reward_embedding: float 
    time_embedding: float 


QueryKeyNormalization = Literal['legacy', 'rank', 'zscore']


class FactorizedEmbeddingWeight(TypedDict, total=False):
    """
    Similarity weights for factorized episode components.
    """

    time_embedding: float
    z: float
    object_shape: float
    object_color: float
    floor_color: float
    wall_color: float


class FeatureRecombinationHiddenState(TypedDict):
    """
    Encoded image and reward representations for one session position.
    """

    stimulus_embeddings: dict[str, torch.Tensor]
    reward_embedding: torch.Tensor


@dataclass
class CompressedEpisodeFactorized:
    """
    Factorized representation of an episode or retrieval cue.

    Attributes:
        z_embedding (torch.Tensor | None): Image latent representation with shape ``[D]``.
        object_shape_embedding (torch.Tensor | None): Shape representation with shape ``[D]``.
        object_color_embedding (torch.Tensor | None): Object-color representation with shape ``[D]``.
        floor_color_embedding (torch.Tensor | None): Floor-color representation with shape ``[D]``.
        wall_color_embedding (torch.Tensor | None): Wall-color representation with shape ``[D]``.
        reward_embedding (torch.Tensor | None): Reward representation with shape ``[D]``.
        time_embedding (torch.Tensor | None): Temporal representation with shape ``[D]``.
    """

    z_embedding: torch.Tensor | None = None
    object_shape_embedding: torch.Tensor | None = None
    object_color_embedding: torch.Tensor | None = None
    floor_color_embedding: torch.Tensor | None = None
    wall_color_embedding: torch.Tensor | None = None
    reward_embedding: torch.Tensor | None = None
    time_embedding: torch.Tensor | None = None

    def get_similarity(
        self,
        key: CompressedEpisodeFactorized,
        component_names: list[str],
    ) -> dict[str, float]:
        """
        Compare active factorized query components against one stored key.

        Args:
            key (CompressedEpisodeFactorized): Stored episode key.
            component_names (list[str]): Active query-key component names.

        Returns:
            dict[str, float]: Cosine similarity for each active component.
        """
        similarities = {}

        for component_name in component_names:
            field_name = FACTORIZED_COMPONENT_FIELDS[component_name]
            query_value = getattr(self, field_name)
            key_value = getattr(key, field_name)

            if query_value is None or key_value is None:
                raise KeyError(f'missing active query-key component: {component_name}')

            similarities[component_name] = float(F.cosine_similarity(
                query_value.flatten(),
                key_value.flatten(),
                dim=0,
            ))

        return similarities


FACTORIZED_COMPONENT_FIELDS = {
    'time_embedding': 'time_embedding',
    'z': 'z_embedding',
    'object_shape': 'object_shape_embedding',
    'object_color': 'object_color_embedding',
    'floor_color': 'floor_color_embedding',
    'wall_color': 'wall_color_embedding',
}
REQUIRED_STIMULUS_COMPONENTS = (
    'z',
    'object_shape',
    'object_color',
    'floor_color',
    'wall_color',
)


class EpisodicSamplingModuleFeatureRecombination:
    """
    Factorized episodic retrieval and reward recombination for one session.
    """

    def __init__(
        self,
        *,
        reward_decoder: torch.nn.Module,
        temporal_embedding_angle: float = np.pi/100,
        temporal_embedding_dim: int = 64,
        temporal_random_seed: int = 44,
        sample_replacement: bool = True,
        query_key_normalization: QueryKeyNormalization = 'legacy',
        device: torch.device | str = 'cpu',
    ) -> None:
        """
        Initialize the factorized retrieval module.

        Args:
            reward_decoder (torch.nn.Module): Decoder from reward embeddings to scalar rewards.
            temporal_embedding_angle (float): Angular step between consecutive temporal embeddings.
            temporal_embedding_dim (int): Temporal embedding dimensionality.
            temporal_random_seed (int): Seed used to regenerate temporal trajectories.
            sample_replacement (bool): Whether finite retrieval samples may repeat episodes.
            query_key_normalization (QueryKeyNormalization): Component normalization strategy.
            device (torch.device | str): Device used by the reward decoder.
        """
        if query_key_normalization not in ('legacy', 'rank', 'zscore'):
            raise ValueError(f'unsupported query-key normalization: {query_key_normalization}')

        self.reward_decoder = reward_decoder
        self.temporal_embedding_angle = temporal_embedding_angle
        self.temporal_embedding_dim = temporal_embedding_dim
        self.temporal_random_seed = temporal_random_seed
        self.sample_replacement = sample_replacement
        self.query_key_normalization = query_key_normalization
        self.device = torch.device(device)
        self.memory: list[CompressedEpisodeFactorized] = []

    def store_session(self, hidden_states: list[FeatureRecombinationHiddenState]) -> None:
        """
        Replace memory with one ordered factorized session.

        Args:
            hidden_states (list[FeatureRecombinationHiddenState]): Ordered encoded session states.
        """
        if not hidden_states:
            raise ValueError('hidden_states must not be empty')

        component_shapes: dict[str, tuple[int, ...]] = {}
        reward_shape: tuple[int, ...] | None = None

        for hidden_state in hidden_states:
            stimulus_embeddings = hidden_state['stimulus_embeddings']
            stimulus_component_names = set(stimulus_embeddings)

            if stimulus_component_names != set(REQUIRED_STIMULUS_COMPONENTS):
                missing_components = set(REQUIRED_STIMULUS_COMPONENTS) - stimulus_component_names
                if missing_components:
                    raise KeyError(f'missing stimulus components: {sorted(missing_components)}')
                unexpected_components = stimulus_component_names - set(REQUIRED_STIMULUS_COMPONENTS)
                raise KeyError(f'unexpected stimulus components: {sorted(unexpected_components)}')

            for component_name in REQUIRED_STIMULUS_COMPONENTS:
                component_embedding = stimulus_embeddings[component_name]
                if not isinstance(component_embedding, torch.Tensor):
                    raise TypeError(f'{component_name} embedding must be a tensor')

                component_shape = tuple(component_embedding.shape)
                if component_name in component_shapes:
                    if component_shape != component_shapes[component_name]:
                        raise ValueError(f'inconsistent {component_name} embedding shape')
                else:
                    component_shapes[component_name] = component_shape

            reward_embedding = hidden_state['reward_embedding']
            if not isinstance(reward_embedding, torch.Tensor):
                raise TypeError('reward_embedding must be a tensor')

            if reward_shape is not None and tuple(reward_embedding.shape) != reward_shape:
                raise ValueError('inconsistent reward embedding shape')
            reward_shape = tuple(reward_embedding.shape)

        temporal_random_state = np.random.RandomState(self.temporal_random_seed)
        temporal_embeddings = generate_temporal_embeddings(
            [self.temporal_embedding_angle] * (len(hidden_states) + 1),
            dim=self.temporal_embedding_dim,
            random_state=temporal_random_state,
        )
        query_time_embedding = torch.tensor(temporal_embeddings[-1]).float()

        self.memory = [
            CompressedEpisodeFactorized(
                z_embedding=hidden_state['stimulus_embeddings']['z'],
                object_shape_embedding=hidden_state['stimulus_embeddings']['object_shape'],
                object_color_embedding=hidden_state['stimulus_embeddings']['object_color'],
                floor_color_embedding=hidden_state['stimulus_embeddings']['floor_color'],
                wall_color_embedding=hidden_state['stimulus_embeddings']['wall_color'],
                reward_embedding=hidden_state['reward_embedding'],
                time_embedding=torch.tensor(temporal_embedding).float(),
            )
            for hidden_state, temporal_embedding in zip(hidden_states, temporal_embeddings[:-1])
        ]
        self.query_time_embedding = query_time_embedding

    @torch.no_grad()
    def get_retrieval_distribution(
        self,
        *,
        query: CompressedEpisodeFactorized,
        weights: FactorizedEmbeddingWeight,
        sampling_temperature: float,
        candidate_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate one factorized query distribution without sampling memories.

        Args:
            query (CompressedEpisodeFactorized): Factorized retrieval cue.
            weights (FactorizedEmbeddingWeight): Nonnegative component weights summing to one.
            sampling_temperature (float): Retrieval softmax temperature.
            candidate_indices (list[int] | None): Eligible indices in the stored session.

        Returns:
            dict[str, Any]: Query similarities, scores, and sampling probabilities.
        """
        if not self.memory:
            raise ValueError('store a session before retrieval')

        active_component_names = self._validate_weights(weights)
        resolved_candidate_indices = self._resolve_candidate_indices(candidate_indices)
        raw_component_similarities = self._get_component_similarities(
            query=query,
            component_names=active_component_names,
        )
        candidate_raw_component_similarities = {
            component_name: similarities[resolved_candidate_indices]
            for component_name, similarities in raw_component_similarities.items()
        }
        normalized_component_similarities = self._normalize_component_similarities(
            candidate_raw_component_similarities,
            active_component_names,
        )
        candidate_query_key_match_scores = sum(
            weights[component_name] * normalized_component_similarities[component_name]
            for component_name in active_component_names
        )
        candidate_sampling_probabilities = self._get_sampling_probabilities(
            query_key_match_scores=candidate_query_key_match_scores,
            sampling_temperature=sampling_temperature,
        )
        full_normalized_component_similarities = {
            component_name: np.zeros(len(self.memory))
            for component_name in active_component_names
        }
        for component_name in active_component_names:
            full_normalized_component_similarities[component_name][resolved_candidate_indices] = (
                normalized_component_similarities[component_name]
            )

        query_key_match_scores = np.full(len(self.memory), -np.inf)
        query_key_match_scores[resolved_candidate_indices] = candidate_query_key_match_scores
        sampling_probabilities = np.zeros(len(self.memory))
        sampling_probabilities[resolved_candidate_indices] = candidate_sampling_probabilities

        return {
            'component_weights': dict(weights),
            'sampling_temperature': sampling_temperature,
            'candidate_indices': resolved_candidate_indices,
            'active_component_names': active_component_names,
            'raw_component_similarities': raw_component_similarities,
            'normalized_component_similarities': full_normalized_component_similarities,
            'query_key_match_scores': query_key_match_scores,
            'sampling_probabilities': sampling_probabilities,
        }

    @torch.no_grad()
    def retrieve_and_recombine_reward(
        self,
        *,
        query: CompressedEpisodeFactorized,
        weights: FactorizedEmbeddingWeight,
        k_samples: int | Literal['all'],
        sampling_temperature: float,
        random_state: np.random.RandomState,
        candidate_indices: list[int] | None = None,
        return_metadata: bool = False,
    ) -> dict[str, Any]:
        """
        Sample matching memories and decode their recombined reward embedding.

        Args:
            query (CompressedEpisodeFactorized): Factorized retrieval cue.
            weights (FactorizedEmbeddingWeight): Nonnegative component weights summing to one.
            k_samples (int | Literal['all']): Requested finite sample count or all episodes.
            sampling_temperature (float): Retrieval softmax temperature.
            random_state (np.random.RandomState): Random source for this retrieval call.
            candidate_indices (list[int] | None): Eligible indices in the stored session.
            return_metadata (bool): Whether to include retrieval diagnostics.

        Returns:
            dict[str, Any]: Decoded recombined reward, selected indices, and optional diagnostics.
        """
        distribution = self.get_retrieval_distribution(
            query=query,
            weights=weights,
            sampling_temperature=sampling_temperature,
            candidate_indices=candidate_indices,
        )
        resolved_candidate_indices = distribution['candidate_indices']
        candidate_sampling_probabilities = distribution['sampling_probabilities'][resolved_candidate_indices]
        sampled_episode_indices = self._sample_episode_indices(
            k_samples=k_samples,
            sampling_probabilities=candidate_sampling_probabilities,
            random_state=random_state,
            candidate_indices=resolved_candidate_indices,
        )
        combined_reward_embedding = self._combine_reward_embeddings(sampled_episode_indices)
        decoded_reward = self.reward_decoder(combined_reward_embedding.unsqueeze(0).to(self.device))

        result = {
            'decoded_recombined_reward': float(decoded_reward.squeeze().detach().cpu()),
            'combined_reward_embedding': combined_reward_embedding.detach().cpu(),
            'sampled_episode_indices': sampled_episode_indices,
        }

        if return_metadata:
            result.update({
                'requested_sampling_budget': k_samples,
                'effective_sample_count': len(sampled_episode_indices),
                **distribution,
            })

        return result

    def _resolve_candidate_indices(self, candidate_indices: list[int] | None) -> list[int]:
        """
        Resolve the eligible memory positions for one retrieval call.

        Args:
            candidate_indices (list[int] | None): Requested eligible memory positions.

        Returns:
            list[int]: Eligible memory positions.
        """
        if candidate_indices is None:
            return list(range(len(self.memory)))
        if not candidate_indices:
            raise ValueError('candidate_indices must not be empty')

        return candidate_indices

    def _validate_weights(self, weights: FactorizedEmbeddingWeight) -> list[str]:
        """
        Validate component weights and resolve active component names.

        Args:
            weights (FactorizedEmbeddingWeight): Requested component weights.

        Returns:
            list[str]: Positive-weight component names in a stable order.
        """
        unknown_component_names = set(weights) - set(FACTORIZED_COMPONENT_FIELDS)
        if unknown_component_names:
            raise KeyError(f'unknown weight components: {sorted(unknown_component_names)}')
        if any(weight < 0.0 for weight in weights.values()):
            raise ValueError('weights must be nonnegative')
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError('weights must sum to one')

        return [
            component_name
            for component_name in FACTORIZED_COMPONENT_FIELDS
            if weights.get(component_name, 0.0) > 0.0
        ]

    def _get_component_similarities(
        self,
        *,
        query: CompressedEpisodeFactorized,
        component_names: list[str],
    ) -> dict[str, np.ndarray]:
        """
        Compute raw cosine similarities across all stored episodes.

        Args:
            query (CompressedEpisodeFactorized): Factorized retrieval cue.
            component_names (list[str]): Positive-weight component names.

        Returns:
            dict[str, np.ndarray]: Raw similarity arrays with shape ``[N]``.
        """
        similarities_by_episode = [
            query.get_similarity(episode, component_names)
            for episode in self.memory
        ]

        return {
            component_name: np.array([
                similarities[component_name]
                for similarities in similarities_by_episode
            ])
            for component_name in component_names
        }

    def _normalize_component_similarities(
        self,
        raw_component_similarities: dict[str, np.ndarray],
        component_names: list[str],
    ) -> dict[str, np.ndarray]:
        """
        Apply the configured query-key component normalization.

        Args:
            raw_component_similarities (dict[str, np.ndarray]): Raw similarities with shape ``[N]``.
            component_names (list[str]): Positive-weight component names.

        Returns:
            dict[str, np.ndarray]: Normalized similarities with shape ``[N]``.
        """
        if self.query_key_normalization == 'legacy':
            return {
                component_name: similarities.copy()
                for component_name, similarities in raw_component_similarities.items()
            }

        similarities_by_episode = [
            {
                component_name: raw_component_similarities[component_name][episode_index]
                for component_name in component_names
            }
            for episode_index in range(len(next(iter(raw_component_similarities.values()))))
        ]

        if self.query_key_normalization == 'rank':
            normalized_by_episode = EpisodicSamplingModuleTwoArmedBandit.normalize_component_similarity_ranks(
                similarities_by_episode,
            )
        else:
            normalized_by_episode = EpisodicSamplingModuleTwoArmedBandit.normalize_component_similarity_zscores(
                similarities_by_episode,
            )

        return {
            component_name: np.array([
                similarities[component_name]
                for similarities in normalized_by_episode
            ])
            for component_name in component_names
        }

    def _get_sampling_probabilities(
        self,
        *,
        query_key_match_scores: np.ndarray,
        sampling_temperature: float,
    ) -> np.ndarray:
        """
        Transform retrieval scores into sampling probabilities.

        Args:
            query_key_match_scores (np.ndarray): Combined query-key scores with shape ``[N]``.
            sampling_temperature (float): Retrieval softmax temperature.

        Returns:
            np.ndarray: Sampling probabilities with shape ``[N]``.
        """
        if len(query_key_match_scores) == 1:
            return np.ones(1)

        if self.query_key_normalization == 'legacy':
            return calibrated_softmax(query_key_match_scores, sampling_temperature)

        return softmax(query_key_match_scores, sampling_temperature)

    def _sample_episode_indices(
        self,
        *,
        k_samples: int | Literal['all'],
        sampling_probabilities: np.ndarray,
        random_state: np.random.RandomState,
        candidate_indices: list[int],
    ) -> list[int]:
        """
        Sample stored episode indices for one retrieval call.

        Args:
            k_samples (int | Literal['all']): Requested finite sample count or all episodes.
            sampling_probabilities (np.ndarray): Candidate probabilities with shape ``[C]``.
            random_state (np.random.RandomState): Random source for this retrieval call.
            candidate_indices (list[int]): Eligible memory positions with shape ``[C]``.

        Returns:
            list[int]: Sampled session indices.
        """
        if k_samples == 'all':
            return list(candidate_indices)
        if not isinstance(k_samples, int) or k_samples <= 0:
            raise ValueError('k_samples must be a positive integer or all')

        sampled_positions = random_state.choice(
            len(candidate_indices),
            size=min(k_samples, len(candidate_indices)),
            replace=self.sample_replacement,
            p=sampling_probabilities,
        )
        return [candidate_indices[int(position)] for position in sampled_positions]

    def _combine_reward_embeddings(self, sampled_episode_indices: list[int]) -> torch.Tensor:
        """
        Average reward embeddings from one retrieved episode sample.

        Args:
            sampled_episode_indices (list[int]): Retrieved session indices.

        Returns:
            torch.Tensor: Combined reward embedding with shape ``[D]``.
        """
        reward_embeddings = torch.vstack([
            self.memory[episode_index].reward_embedding
            for episode_index in sampled_episode_indices
        ])
        return reward_embeddings.mean(dim=0, keepdim=False)


class EpisodicSamplingModuleTwoArmedBandit:
    """
    Episodic memory system for a two-armed bandit agent.

    The module stores experienced episodes and estimates action values by
    sampling remembered outcomes. It does not choose actions directly. Instead,
    it retrieves episodes for each arm, averages their reward representations,
    and decodes the averaged representation into a scalar value.

    Retrieval is cue-dependent: reminder stimuli and temporal context change
    which past episodes are likely to be sampled.
    """

    def __init__(
        self,
        *,
        weights: EmbeddingWeight = dict(
            stimulus_embedding=1.0 / 3,
            reward_embedding=1.0 / 3,
            time_embedding=1.0 / 3,
        ),
        k_samples: int | Literal['all'] = 1,
        sample_replacement: bool = True,
        sampling_temperature: float = 1.0,
        normalize_query_key_components: bool = False,
        query_key_normalization: QueryKeyNormalization | None = None,
        device: torch.device | str = 'cpu',
        reward_decoder: torch.nn.Module,
        random_state: np.random.RandomState | None = None,
    ) -> None:
        """
        Initialize episodic retrieval and reward decoding.

        Args:
            weights (dict[str, float]): Similarity weights for available episode components.
            k_samples (int): Number of retrieved episodes used for each arm value.
            sample_replacement (bool): allow sample same episodes (True), or only distinct episodes (False).
            sampling_temperature (float): Softmax temperature for memory sampling.
            normalize_query_key_components (bool): Normalize component-wise query-key ranks before weighting.
            query_key_normalization (str | None): Query-key normalization strategy.
            default_q_value (float): Value returned when no memory exists for an arm.
            device (torch.device | str): Device used by the reward decoder.
            reward_decoder (torch.nn.Module): Decoder mapping reward embeddings to scalar rewards.
            random_state (np.random.RandomState | None): Optional seeded random state for memory sampling.
        """
        assert sum([weights[i] for i in weights]) == 1.0, 'weights must sum to 1...'

        self.reward_decoder = reward_decoder
        self.weights = weights
        self.k_samples = k_samples
        self.sample_replacement = sample_replacement
        self.sampling_temperature = sampling_temperature
        self.normalize_query_key_components = normalize_query_key_components
        self.query_key_normalization = query_key_normalization
        self.default_q_value = 0.
        self.device = torch.device(device)
        self.random_state = random_state

        self.memory = []
        self.memory_log = []
        self.candidate_indices_by_arm = {'LEFT': [], 'RIGHT': []}

    def resolve_query_key_normalization(self) -> QueryKeyNormalization:
        """
        Resolve the active query-key normalization strategy.

        Returns:
            str: Active query-key normalization strategy.
        """
        query_key_normalization = getattr(self, 'query_key_normalization', None)

        if query_key_normalization is not None:
            return query_key_normalization

        if getattr(self, 'normalize_query_key_components', False):
            return 'rank'

        return 'legacy'

    def store_episode(self, episode: dict[str, Any]) -> None:
        """
        Store a newly experienced episode.

        The memory key contains the information that can make the episode
        searchable later. The memory value contains the information recovered
        after sampling, especially the reward representation used for valuation.

        Args:
            episode (dict[str, Any]): Trial episode with encoded hidden states,
                chosen arm, scalar reward, and trial metadata.
        """
        self.memory_log.append(episode['trial_metadata'])

        self.memory.append(dict(
            key=CompressedEpisode(**episode['hidden_states']),
            value=CompressedEpisode(
                stimulus_embedding=episode['hidden_states']['stimulus_embedding'],
                reward_embedding=episode['hidden_states']['reward_embedding'],
            ),
            reward=episode['reward'],
            chosen_arm=episode['chosen_arm'],
        ))
        self.candidate_indices_by_arm[episode['chosen_arm']].append(len(self.memory) - 1)

    def query_similarity_scores(
        self,
        query: CompressedEpisode,
        candidate_indices: list[int],
        weights: EmbeddingWeight = None
    ) -> np.ndarray:
        """
        Score how strongly a query matches candidate memories.

        Each candidate memory key is compared with the current cue. Available
        component similarities are combined with this module's weighting scheme,
        producing one accessibility score per candidate memory.

        Args:
            query (CompressedEpisode): Current retrieval cue.
            candidate_indices (list[int]): Memory indices eligible for retrieval.
            weights (EmbeddingWeight): If provided, it overrides the module's 
            default weights for this query

        Returns:
            np.ndarray: Weighted retrieval score for each candidate.
        """
        weights = weights if weights is not None else self.weights
        similarities_by_candidate = []

        for memory_idx in candidate_indices:
            key = self.memory[memory_idx]['key']
            similarities = query.get_similarity(key)
            similarities_by_candidate.append(similarities)

        normalization = self.resolve_query_key_normalization()

        if normalization == 'rank':
            similarities_by_candidate = self.normalize_component_similarity_ranks(
                similarities_by_candidate,
            )
        elif normalization == 'zscore':
            similarities_by_candidate = self.normalize_component_similarity_zscores(
                similarities_by_candidate,
            )

        scores = []

        for similarities in similarities_by_candidate:
            score = sum(
                weights.get(k, 0) * similarities[k] 
                for k in similarities
            )
            
            scores.append(score)

        return np.array(scores)

    @staticmethod
    def normalize_component_similarity_ranks(
        similarities_by_candidate: list[dict[str, float]],
    ) -> list[dict[str, float]]:
        """
        Map each query-key component to a common candidate-rank scale.

        Args:
            similarities_by_candidate (list[dict[str, float]]): Component similarities per candidate.

        Returns:
            list[dict[str, float]]: Rank-normalized component similarities per candidate.
        """
        n_candidates = len(similarities_by_candidate)
        component_names = similarities_by_candidate[0].keys()
        normalized = [dict() for _ in similarities_by_candidate]

        for component_name in component_names:
            values = np.array([
                similarities[component_name]
                for similarities in similarities_by_candidate
            ])
            descending_ranks = np.argsort(
                np.argsort(-values, kind='stable'),
                kind='stable',
            )
            rank_scores = 2.0 * (1.0 - (descending_ranks + 0.5) / n_candidates) - 1.0

            for candidate_index, rank_score in enumerate(rank_scores):
                normalized[candidate_index][component_name] = float(rank_score)

        return normalized

    @staticmethod
    def normalize_component_similarity_zscores(
        similarities_by_candidate: list[dict[str, float]],
    ) -> list[dict[str, float]]:
        """
        Standardize each query-key component over the current candidate set.

        Args:
            similarities_by_candidate (list[dict[str, float]]): Component similarities per candidate.

        Returns:
            list[dict[str, float]]: Component-wise z-scored similarities per candidate.
        """
        component_names = similarities_by_candidate[0].keys()
        normalized = [dict() for _ in similarities_by_candidate]

        for component_name in component_names:
            values = np.array([
                similarities[component_name]
                for similarities in similarities_by_candidate
            ])
            standard_deviation = values.std()

            if standard_deviation == 0.0:
                zscores = np.zeros_like(values)
            else:
                zscores = (values - values.mean()) / standard_deviation

            for candidate_index, zscore in enumerate(zscores):
                normalized[candidate_index][component_name] = float(zscore)

        return normalized

    def sample_episode_indices(
        self,
        query: CompressedEpisode,
        candidate_indices: list[int],
        weights: EmbeddingWeight = None,
        return_metadata: bool = False 
    ) -> list[int]:
        """
        Sample memories from cue-dependent accessibility scores.

        Retrieval scores are converted into a probability distribution. Higher
        scoring episodes are more likely to be sampled, but the selected memory
        set remains stochastic.

        Args:
            query (CompressedEpisode): Current retrieval cue.
            candidate_indices (list[int]): Memory indices eligible for retrieval.
            weights (EmbeddingWeight): If provided, it overrides the module's 
            default weights for this query-based sampling
            return_metadata: if True, return additional computed quantities

        Returns:
            list[int]: Sampled memory indices.
        """
        if len(candidate_indices) == 0:
            return []
        
        if self.k_samples == "all":
            sampled_episode_indices = list(candidate_indices)
            metadata = {}

        else:
            qk_scores = self.query_similarity_scores(query, candidate_indices, weights)
            if self.resolve_query_key_normalization() == 'legacy':
                probs = calibrated_softmax(qk_scores, self.sampling_temperature)
            else:
                probs = softmax(qk_scores, self.sampling_temperature)

            k = min(self.k_samples, len(candidate_indices))
            rng = self.random_state if self.random_state is not None else np.random
            idx = rng.choice(
                len(candidate_indices), 
                size=k, 
                replace=self.sample_replacement, 
                p=probs
            )
            sampled_episode_indices = [candidate_indices[i] for i in idx]
            
            metadata = dict(
                qk_scores=qk_scores,
                probs=probs
            )
        
        if return_metadata:
            return sampled_episode_indices, metadata
        return sampled_episode_indices
        

    @torch.no_grad()
    def decode_sampled_reward(
        self,
        sampled_indices: list[int],
    ) -> float:
        """
        Decode sampled reward memories into one scalar action value.

        The sampled reward embeddings are averaged and passed through the reward
        decoder. This implements valuation from retrieved episodic reward
        information.

        Args:
            sampled_indices (list[int]): Memory indices selected for one arm.

        Returns:
            float: Decoded scalar reward estimate.
        """
        if len(sampled_indices) == 0:
            return float(self.default_q_value)

        reward_embeddings = torch.vstack([
            self.memory[i]['value'].reward_embedding
            for i in sampled_indices
        ])

        avg_reward_embedding = reward_embeddings.mean(dim=0, keepdim=True)
        avg_reward_embedding = avg_reward_embedding.to(self.device)

        decoded_reward = self.reward_decoder(avg_reward_embedding)

        return float(decoded_reward.squeeze().detach().cpu())

    def estimate_action_values(
        self,
        query: CompressedEpisode,
        weights: EmbeddingWeight = None,     
        return_metadata: bool = False   
    ) -> dict[str, float]:
        """
        Estimate LEFT and RIGHT values from separate episodic samples.

        For each arm, only memories in which that arm was chosen are eligible.
        The module samples from those memories under the current query and
        decodes sampled reward information into one scalar action value.

        Args:
            query (CompressedEpisode): Current retrieval cue.
            weights (EmbeddingWeight): If provided, it overrides the module's 
            default weights for this query-based sampling and sampling-based action values
            return_metadata: if True, return additional computed quantities
            
        Returns:
            dict[str, float]: Scalar action values for ``'LEFT'`` and ``'RIGHT'``.
        """
        metadata = dict()
        
        action_value = {}
        
        for arm in ['LEFT', 'RIGHT']:
            candidate_indices = self.candidate_indices_by_arm[arm]

            sampled_episode_indices, episodic_sample_metadata = self.sample_episode_indices(
                query, 
                candidate_indices, 
                weights,
                return_metadata=True
            )
            reward_avg_score = self.decode_sampled_reward(sampled_episode_indices)
            action_value[arm] = reward_avg_score
            
            metadata[arm] = dict(
                candidate_indices=candidate_indices,
                sampled_episode_indices=sampled_episode_indices,
                **episodic_sample_metadata
            )

        if return_metadata:
            return action_value, metadata
        return action_value
