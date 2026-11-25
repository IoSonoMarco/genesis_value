from typing import Literal, TypedDict

import numpy as np


ArmName = Literal['LEFT', 'RIGHT']
DeckName = Literal['A', 'B', 'C', 'D']


class ActionValue(TypedDict):
    """
    Scalar values estimated for the two bandit arms.
    """
    LEFT: float
    RIGHT: float


class PolicyOutput(TypedDict):
    """
    Stochastic choice result returned by the two-armed bandit policy.
    """
    chosen_arm: ArmName
    logit_right: float
    p_right: float


class IGTActionValue(TypedDict):
    """
    Scalar values estimated for the four IGT decks.
    """

    A: float
    B: float
    C: float
    D: float


class IGTPolicyOutput(TypedDict):
    """
    Stochastic choice result returned by the IGT policy.
    """

    chosen_deck: DeckName
    choice_temperature: float
    p_a: float
    p_b: float
    p_c: float
    p_d: float


def get_linear_choice_temperature(
    *,
    trial_id: int,
    max_choice_temperature: float,
    min_choice_temperature: float,
    n_anneal_trials: int,
) -> float:
    """
    Resolve the shared linearly annealed IGT choice temperature.

    Args:
        trial_id (int): Zero-based global IGT trial index.
        max_choice_temperature (float): Temperature on the first trial.
        min_choice_temperature (float): Temperature after annealing.
        n_anneal_trials (int): Number of trials in the linear schedule.

    Returns:
        float: Choice temperature for the requested trial.
    """
    if trial_id >= n_anneal_trials:
        return min_choice_temperature

    return max_choice_temperature + (
        trial_id / (n_anneal_trials - 1)
    ) * (min_choice_temperature - max_choice_temperature)


class SimpleTwoArmedBanditPolicy:
    """
    Minimal stochastic policy for a two-armed bandit.

    The policy receives action values from the episodic memory system and turns
    them into a probabilistic LEFT/RIGHT choice. It does not store memories,
    retrieve episodes, encode stimuli, or process reminders.
    """

    def __init__(
        self,
        beta_choice: float = 2.0,
    ) -> None:
        """
        Initialize choice sensitivity.

        Args:
            beta_choice (float): Scaling applied to the RIGHT-minus-LEFT value difference.
        """
        self.beta_choice = beta_choice

    @staticmethod
    def sigmoid(x: float) -> float:
        """
        Apply the logistic function.

        Args:
            x (float): Scalar logit.

        Returns:
            float: Logistic transform of ``x``.
        """
        return 1.0 / (1.0 + np.exp(-x))

    def choose_arm(
        self,
        action_value: ActionValue,
        *,
        random_state: np.random.RandomState | None = None,
    ) -> PolicyOutput:
        """
        Choose between LEFT and RIGHT from episodic action values.

        The value difference is scaled by ``beta_choice`` and converted into the
        probability of choosing RIGHT. The final arm is sampled stochastically
        from that probability.

        Args:
            action_value (ActionValue): Scalar values for the LEFT and RIGHT arms.
            random_state (np.random.RandomState | None): Optional seeded random state for action sampling.

        Returns:
            PolicyOutput: Chosen arm, RIGHT-choice logit, and RIGHT-choice probability.
        """
        logit_right = self.beta_choice * (action_value['RIGHT'] - action_value['LEFT'])
        p_right = self.sigmoid(logit_right)

        rng = random_state if random_state is not None else np.random
        chosen_arm = 'RIGHT' if rng.random() < p_right else 'LEFT'

        return dict(
            chosen_arm=chosen_arm,
            logit_right=logit_right,
            p_right=p_right,
        )


class IowaGamblingTaskPolicy:
    """
    Stochastic four-deck policy with a linearly annealed temperature.
    """

    def __init__(
        self,
        *,
        max_choice_temperature: float,
        min_choice_temperature: float,
        n_anneal_trials: int,
    ) -> None:
        """
        Initialize the IGT choice-temperature schedule.

        Args:
            max_choice_temperature (float): Temperature on the first trial.
            min_choice_temperature (float): Temperature after annealing.
            n_anneal_trials (int): Number of trials in the linear schedule.
        """
        self.max_choice_temperature = max_choice_temperature
        self.min_choice_temperature = min_choice_temperature
        self.n_anneal_trials = n_anneal_trials

    def get_choice_temperature(self, trial_id: int) -> float:
        """
        Resolve the scheduled choice temperature for one trial.

        Args:
            trial_id (int): Zero-based global IGT trial index.

        Returns:
            float: Choice softmax temperature for the trial.
        """
        return get_linear_choice_temperature(
            trial_id=trial_id,
            max_choice_temperature=self.max_choice_temperature,
            min_choice_temperature=self.min_choice_temperature,
            n_anneal_trials=self.n_anneal_trials,
        )

    def choose_deck(
        self,
        *,
        action_values: IGTActionValue,
        trial_id: int,
        random_state: np.random.RandomState | None = None,
    ) -> IGTPolicyOutput:
        """
        Sample one IGT deck from episodic action values.

        Args:
            action_values (IGTActionValue): Scalar values for decks A through D.
            trial_id (int): Zero-based global IGT trial index.
            random_state (np.random.RandomState | None): Optional choice random source.

        Returns:
            IGTPolicyOutput: Chosen deck, temperature, and deck probabilities.
        """
        choice_temperature = self.get_choice_temperature(trial_id)
        decks = ('A', 'B', 'C', 'D')
        logits = np.array([action_values[deck] for deck in decks]) / choice_temperature
        logits = logits - logits.max()
        probabilities = np.exp(logits)
        probabilities = probabilities / probabilities.sum()
        rng = random_state if random_state is not None else np.random
        chosen_deck = decks[int(rng.choice(len(decks), p=probabilities))]

        return {
            'chosen_deck': chosen_deck,
            'choice_temperature': choice_temperature,
            'p_a': float(probabilities[0]),
            'p_b': float(probabilities[1]),
            'p_c': float(probabilities[2]),
            'p_d': float(probabilities[3]),
        }
