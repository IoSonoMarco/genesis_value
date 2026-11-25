from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import numpy as np
import torch
import torch.nn.functional as F

from models.memory.rag import EpisodicSamplingModuleTwoArmedBandit, QueryKeyNormalization
from models.utils import calibrated_softmax, softmax


DeckName = Literal['A', 'B', 'C', 'D']
DECK_NAMES: tuple[DeckName, ...] = ('A', 'B', 'C', 'D')


class IGTEmbeddingWeight(TypedDict, total=False):
    """
    Similarity weights for IGT retrieval keys.
    """

    time_embedding: float
    valence_embedding: float
    negative_attention: float
    positive_attention: float


IGTEpisodeKey = Literal['time_embedding', 'valence_embedding', 'reward_embedding']


@dataclass(frozen=True)
class IGTQueryComponent:
    """
    One named IGT query embedding and its stored episode key.

    Attributes:
        embedding (torch.Tensor): Query representation with shape ``[D]``.
        episode_key (IGTEpisodeKey): Stored episode field compared with the query.
    """

    embedding: torch.Tensor
    episode_key: IGTEpisodeKey


class IGTActionValue(TypedDict):
    """
    Scalar values estimated for the four IGT decks.
    """

    A: float
    B: float
    C: float
    D: float


@dataclass
class IGTQuery:
    """
    Retrieval cue for one IGT decision.

    Attributes:
        time_embedding (torch.Tensor | None): Temporal representation with shape ``[D]``.
        valence_embedding (torch.Tensor | None): Legacy valence representation with shape ``[D]``.
        components (dict[str, IGTQueryComponent] | None): General named query components.
    """

    time_embedding: torch.Tensor | None = None
    valence_embedding: torch.Tensor | None = None
    components: dict[str, IGTQueryComponent] | None = None


@dataclass
class IGTEpisode:
    """
    Stored IGT episode with separate retrieval key and reward value.

    Attributes:
        time_embedding (torch.Tensor): Temporal representation with shape ``[D]``.
        valence_embedding (torch.Tensor | None): Valence representation with shape ``[D]``.
        reward_embedding (torch.Tensor): Reward representation with shape ``[D]``.
    """

    time_embedding: torch.Tensor
    valence_embedding: torch.Tensor | None
    reward_embedding: torch.Tensor


class EpisodicSamplingModuleIowaGamblingTask:
    """
    Four-deck episodic retrieval and reward reconstruction for the IGT.
    """

    def __init__(
        self,
        *,
        reward_decoder: torch.nn.Module,
        k_samples: int,
        sampling_temperature: float,
        query_key_normalization: QueryKeyNormalization = 'legacy',
        device: torch.device | str = 'cpu',
    ) -> None:
        """
        Initialize four-deck episodic reward reconstruction.

        Args:
            reward_decoder (torch.nn.Module): Decoder from reward embeddings to scalar rewards.
            k_samples (int): Number of replacement samples per nonempty deck.
            sampling_temperature (float): Retrieval softmax temperature.
            query_key_normalization (QueryKeyNormalization): Query-key normalization strategy.
            device (torch.device | str): Device used by the reward decoder.
        """
        self.reward_decoder = reward_decoder
        self.k_samples = k_samples
        self.sampling_temperature = sampling_temperature
        self.query_key_normalization = query_key_normalization
        self.device = torch.device(device)
        self.memory: list[IGTEpisode] = []
        self.memory_log: list[dict[str, Any]] = []
        self.candidate_indices_by_deck: dict[DeckName, list[int]] = {
            deck: []
            for deck in DECK_NAMES
        }

    def store_episode(
        self,
        *,
        chosen_deck: DeckName,
        time_embedding: torch.Tensor,
        valence_embedding: torch.Tensor | None,
        reward_embedding: torch.Tensor,
        reward: float,
        trial_metadata: dict[str, Any],
    ) -> None:
        """
        Store one experienced IGT outcome in its deck-specific memory pool.

        Args:
            chosen_deck (DeckName): Deck selected on the experienced trial.
            time_embedding (torch.Tensor): Temporal representation with shape ``[D]``.
            valence_embedding (torch.Tensor | None): Valence representation with shape ``[D]``.
            reward_embedding (torch.Tensor): Reward representation with shape ``[D]``.
            reward (float): Observed scalar reward.
            trial_metadata (dict[str, Any]): Trial information retained for inspection.
        """
        self.memory.append(IGTEpisode(
            time_embedding=time_embedding,
            valence_embedding=valence_embedding,
            reward_embedding=reward_embedding,
        ))
        self.memory_log.append({
            'chosen_deck': chosen_deck,
            'reward': reward,
            'trial_metadata': trial_metadata,
        })
        self.candidate_indices_by_deck[chosen_deck].append(len(self.memory) - 1)

    @torch.no_grad()
    def estimate_action_values(
        self,
        *,
        query: IGTQuery,
        weights: IGTEmbeddingWeight,
        random_state: np.random.RandomState,
        return_metadata: bool = False,
    ) -> IGTActionValue | tuple[IGTActionValue, dict[DeckName, dict[str, Any]]]:
        """
        Reconstruct one action value for each IGT deck from sampled memories.

        Args:
            query (IGTQuery): Current temporal-only or temporal-valence cue.
            weights (IGTEmbeddingWeight): Nonnegative query-component weights summing to one.
            random_state (np.random.RandomState): Random source for retrieval samples.
            return_metadata (bool): Whether to return deck-level retrieval diagnostics.

        Returns:
            IGTActionValue | tuple[IGTActionValue, dict[DeckName, dict[str, Any]]]: Four reconstructed
                values and optional retrieval metadata.
        """
        active_components = self._resolve_active_components(query=query, weights=weights)
        action_values: IGTActionValue = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
        metadata: dict[DeckName, dict[str, Any]] = {}

        for deck in DECK_NAMES:
            candidate_indices = self.candidate_indices_by_deck[deck]
            if not candidate_indices:
                metadata[deck] = self._empty_metadata(weights)
                continue

            raw_component_similarities = self._get_component_similarities(
                query=query,
                candidate_indices=candidate_indices,
                component_names=active_components,
            )
            normalized_component_similarities = self._normalize_component_similarities(
                raw_component_similarities=raw_component_similarities,
                component_names=active_components,
            )
            query_key_scores = sum(
                weights[component_name] * normalized_component_similarities[component_name]
                for component_name in active_components
            )
            sampling_probabilities = self._get_sampling_probabilities(query_key_scores)
            sampled_episode_indices = self._sample_episode_indices(
                candidate_indices=candidate_indices,
                sampling_probabilities=sampling_probabilities,
                random_state=random_state,
            )
            action_values[deck] = self._decode_reward(sampled_episode_indices)
            metadata[deck] = {
                'candidate_indices': list(candidate_indices),
                'available_memory_count': len(candidate_indices),
                'raw_component_similarities': raw_component_similarities,
                'normalized_component_similarities': normalized_component_similarities,
                'query_key_scores': query_key_scores,
                'sampling_probabilities': sampling_probabilities,
                'sampled_episode_indices': sampled_episode_indices,
                'sampling_budget': self.k_samples,
                'sampling_temperature': self.sampling_temperature,
                'component_weights': dict(weights),
            }

        if return_metadata:
            return action_values, metadata

        return action_values

    @torch.no_grad()
    def estimate_expected_action_values(
        self,
        *,
        query: IGTQuery,
        weights: IGTEmbeddingWeight,
        return_metadata: bool = False,
    ) -> IGTActionValue | tuple[IGTActionValue, dict[DeckName, dict[str, Any]]]:
        """
        Reconstruct deck values from the expected retrieved reward embedding.

        Args:
            query (IGTQuery): Current temporal-valence cue.
            weights (IGTEmbeddingWeight): Nonnegative query-component weights summing to one.
            return_metadata (bool): Whether to return deck-level retrieval diagnostics.

        Returns:
            IGTActionValue | tuple[IGTActionValue, dict[DeckName, dict[str, Any]]]: Four expected
                values and optional retrieval metadata.
        """
        active_components = self._resolve_active_components(query=query, weights=weights)
        action_values: IGTActionValue = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
        metadata: dict[DeckName, dict[str, Any]] = {}

        for deck in DECK_NAMES:
            candidate_indices = self.candidate_indices_by_deck[deck]
            if not candidate_indices:
                metadata[deck] = self._empty_metadata(weights)
                continue

            raw_component_similarities = self._get_component_similarities(
                query=query,
                candidate_indices=candidate_indices,
                component_names=active_components,
            )
            normalized_component_similarities = self._normalize_component_similarities(
                raw_component_similarities=raw_component_similarities,
                component_names=active_components,
            )
            query_key_scores = sum(
                weights[component_name] * normalized_component_similarities[component_name]
                for component_name in active_components
            )
            sampling_probabilities = self._get_sampling_probabilities(query_key_scores)
            action_values[deck] = self._decode_expected_reward(
                candidate_indices=candidate_indices,
                sampling_probabilities=sampling_probabilities,
            )
            metadata[deck] = {
                'candidate_indices': list(candidate_indices),
                'available_memory_count': len(candidate_indices),
                'raw_component_similarities': raw_component_similarities,
                'normalized_component_similarities': normalized_component_similarities,
                'query_key_scores': query_key_scores,
                'sampling_probabilities': sampling_probabilities,
                'sampled_episode_indices': [],
                'sampling_budget': self.k_samples,
                'sampling_temperature': self.sampling_temperature,
                'component_weights': dict(weights),
            }

        if return_metadata:
            return action_values, metadata

        return action_values

    def _resolve_active_components(
        self,
        *,
        query: IGTQuery,
        weights: IGTEmbeddingWeight,
    ) -> list[str]:
        """
        Resolve positive-weight IGT retrieval components.

        Args:
            query (IGTQuery): Current retrieval cue.
            weights (IGTEmbeddingWeight): Query-component weights.

        Returns:
            list[str]: Active component names in stable order.
        """
        query_components = getattr(query, 'components', None)
        component_names = tuple(weights) if query_components is not None else ('time_embedding', 'valence_embedding')
        if set(weights) - set(component_names):
            raise KeyError('unknown IGT query-key component')
        if any(weight < 0.0 for weight in weights.values()):
            raise ValueError('IGT query-key weights must be nonnegative')
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError('IGT query-key weights must sum to one')

        return [
            component_name
            for component_name in component_names
            if weights.get(component_name, 0.0) > 0.0
        ]

    def _get_component_similarities(
        self,
        *,
        query: IGTQuery,
        candidate_indices: list[int],
        component_names: list[str],
    ) -> dict[str, np.ndarray]:
        """
        Compute cosine similarity for each active component and candidate.

        Args:
            query (IGTQuery): Current retrieval cue.
            candidate_indices (list[int]): Eligible episode indices for one deck.
            component_names (list[str]): Positive-weight component names.

        Returns:
            dict[str, np.ndarray]: Component similarities with shape ``[N]``.
        """
        similarities = {
            component_name: []
            for component_name in component_names
        }

        for episode_index in candidate_indices:
            episode = self.memory[episode_index]
            for component_name in component_names:
                if query.components is None:
                    query_embedding = getattr(query, component_name)
                    episode_embedding = getattr(episode, component_name)
                else:
                    query_component = query.components[component_name]
                    query_embedding = query_component.embedding
                    episode_embedding = getattr(episode, query_component.episode_key)
                if query_embedding is None or episode_embedding is None:
                    raise KeyError(f'missing active query-key component: {component_name}')
                similarity = F.cosine_similarity(
                    query_embedding.flatten(),
                    episode_embedding.flatten(),
                    dim=0,
                )
                similarities[component_name].append(float(similarity))

        return {
            component_name: np.asarray(values)
            for component_name, values in similarities.items()
        }

    def _normalize_component_similarities(
        self,
        *,
        raw_component_similarities: dict[str, np.ndarray],
        component_names: list[str],
    ) -> dict[str, np.ndarray]:
        """
        Normalize candidate similarities using the configured GENESIS strategy.

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

        similarities_by_candidate = [
            {
                component_name: raw_component_similarities[component_name][candidate_index]
                for component_name in component_names
            }
            for candidate_index in range(len(next(iter(raw_component_similarities.values()))))
        ]
        normalizer = (
            EpisodicSamplingModuleTwoArmedBandit.normalize_component_similarity_ranks
            if self.query_key_normalization == 'rank'
            else EpisodicSamplingModuleTwoArmedBandit.normalize_component_similarity_zscores
        )
        normalized_by_candidate = normalizer(similarities_by_candidate)

        return {
            component_name: np.asarray([
                similarities[component_name]
                for similarities in normalized_by_candidate
            ])
            for component_name in component_names
        }

    def _get_sampling_probabilities(self, query_key_scores: np.ndarray) -> np.ndarray:
        """
        Convert resolved query-key scores into retrieval probabilities.

        Args:
            query_key_scores (np.ndarray): Combined scores with shape ``[N]``.

        Returns:
            np.ndarray: Retrieval probabilities with shape ``[N]``.
        """
        if self.query_key_normalization == 'legacy':
            return calibrated_softmax(query_key_scores, self.sampling_temperature)

        return softmax(query_key_scores, self.sampling_temperature)

    def _sample_episode_indices(
        self,
        *,
        candidate_indices: list[int],
        sampling_probabilities: np.ndarray,
        random_state: np.random.RandomState,
    ) -> list[int]:
        """
        Draw the configured replacement sample from one deck memory pool.

        Args:
            candidate_indices (list[int]): Eligible episode indices for one deck.
            sampling_probabilities (np.ndarray): Candidate probabilities with shape ``[N]``.
            random_state (np.random.RandomState): Random source for retrieval samples.

        Returns:
            list[int]: Sampled global episode indices.
        """
        sampled_positions = random_state.choice(
            len(candidate_indices),
            size=self.k_samples,
            replace=True,
            p=sampling_probabilities,
        )
        return [candidate_indices[int(position)] for position in sampled_positions]

    def _decode_reward(self, sampled_episode_indices: list[int]) -> float:
        """
        Average retrieved reward embeddings and decode one scalar value.

        Args:
            sampled_episode_indices (list[int]): Sampled global episode indices.

        Returns:
            float: Decoded scalar action value.
        """
        reward_embeddings = torch.vstack([
            self.memory[episode_index].reward_embedding
            for episode_index in sampled_episode_indices
        ])
        mean_reward_embedding = reward_embeddings.mean(dim=0, keepdim=True).to(self.device)
        decoded_reward = self.reward_decoder(mean_reward_embedding)

        return float(decoded_reward.squeeze().detach().cpu())

    def _decode_expected_reward(
        self,
        *,
        candidate_indices: list[int],
        sampling_probabilities: np.ndarray,
    ) -> float:
        """
        Decode the probability-weighted reward embedding for one deck.

        Args:
            candidate_indices (list[int]): Eligible global episode indices.
            sampling_probabilities (np.ndarray): Retrieval probabilities with shape ``[N]``.

        Returns:
            float: Expected decoded scalar action value.
        """
        reward_embeddings = torch.vstack([
            self.memory[episode_index].reward_embedding
            for episode_index in candidate_indices
        ])
        probabilities = torch.tensor(
            sampling_probabilities,
            dtype=reward_embeddings.dtype,
            device=reward_embeddings.device,
        )
        expected_reward_embedding = (probabilities.unsqueeze(-1) * reward_embeddings).sum(dim=0, keepdim=True)
        decoded_reward = self.reward_decoder(expected_reward_embedding.to(self.device))

        return float(decoded_reward.squeeze().detach().cpu())

    def _empty_metadata(self, weights: IGTEmbeddingWeight) -> dict[str, Any]:
        """
        Build retrieval metadata for an unseen deck.

        Args:
            weights (IGTEmbeddingWeight): Active query-component weights.

        Returns:
            dict[str, Any]: Empty deck retrieval diagnostics.
        """
        return {
            'candidate_indices': [],
            'available_memory_count': 0,
            'raw_component_similarities': {},
            'normalized_component_similarities': {},
            'query_key_scores': np.asarray([]),
            'sampling_probabilities': np.asarray([]),
            'sampled_episode_indices': [],
            'sampling_budget': self.k_samples,
            'sampling_temperature': self.sampling_temperature,
            'component_weights': dict(weights),
        }
