"""
Prospect Valence Learning model with Delta updating for the IGT.
"""

from __future__ import annotations

import numpy as np


class PVLDeltaModel:
    """
    Fixed-parameter four-deck PVL-Delta learner.
    """

    def __init__(
        self,
        *,
        outcome_sensitivity: float,
        loss_aversion: float,
        learning_rate: float,
        response_consistency: float,
    ) -> None:
        """
        Initialize the PVL-Delta parameters and zero deck expectancies.

        Args:
            outcome_sensitivity (float): Prospect-utility exponent ``A``.
            loss_aversion (float): Negative-outcome weight ``w_loss``.
            learning_rate (float): Delta learning rate ``a``.
            response_consistency (float): Choice-consistency parameter ``c``.
        """
        self.outcome_sensitivity = outcome_sensitivity
        self.loss_aversion = loss_aversion
        self.learning_rate = learning_rate
        self.response_consistency = response_consistency
        self.base_response_sensitivity = 3.0 ** response_consistency - 1.0
        self.expectancies = np.zeros(4, dtype=np.float64)

    def subjective_utility(self, reward: float) -> float:
        """
        Transform one experienced net reward through prospect utility.

        Args:
            reward (float): Scaled net payoff from the chosen deck.

        Returns:
            float: Subjective utility of the net payoff.
        """
        if reward >= 0.0:
            return reward ** self.outcome_sensitivity
        return -self.loss_aversion * abs(reward) ** self.outcome_sensitivity

    def choice_probabilities(self, response_sensitivity: float) -> np.ndarray:
        """
        Convert current deck expectancies to choice probabilities.

        Args:
            response_sensitivity (float): Effective response sensitivity ``theta_t``.

        Returns:
            np.ndarray: Deck probabilities with shape ``[4]``.
        """
        logits = response_sensitivity * self.expectancies
        logits = logits - logits.max()
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum()

    def update(self, *, deck_index: int, subjective_utility: float) -> float:
        """
        Apply one chosen-deck Delta update.

        Args:
            deck_index (int): Zero-based chosen deck index.
            subjective_utility (float): Utility of the experienced net payoff.

        Returns:
            float: Prediction error before the expectancy update.
        """
        prediction_error = subjective_utility - self.expectancies[deck_index]
        self.expectancies[deck_index] += self.learning_rate * prediction_error
        return float(prediction_error)
