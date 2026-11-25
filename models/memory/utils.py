import numpy as np


def l2normalize(x: np.ndarray, dim: int = 0) -> np.ndarray:
    return x/np.linalg.norm(x, axis=dim, keepdims=True)


def combine_embeddings(x: np.ndarray, y: np.ndarray, theta: float) -> np.ndarray:
    return np.hstack([theta*x, (1-theta)*y])


def generate_vector_by_angle(
    x: np.ndarray,
    theta: float,
    *,
    random_state: np.random.RandomState | None = None,
) -> np.ndarray:
    """
    Generate a unit vector at a fixed angle from another unit vector.

    Args:
        x (np.ndarray): Unit vector with shape ``[D]``.
        theta (float): Angle between ``x`` and the new vector.
        random_state (np.random.RandomState | None): Optional seeded random state.

    Returns:
        np.ndarray: Generated unit vector with shape ``[D]``.
    """
    rng = random_state if random_state is not None else np.random
    random_vector = rng.randn(len(x))
    
    # create a vector perpendicular to x:
    x_orthogonal = random_vector - np.dot(random_vector, x)*x
    x_orthogonal = l2normalize(x_orthogonal)
    
    # y is the linear combination of x and x_orthogonal with coefficients cos(theta) and sin(theta), respectively:
    y = np.cos(theta)*x + np.sin(theta)*x_orthogonal
    
    return y


def generate_temporal_embeddings(
    gammas: list[float],
    dim: int,
    t0: np.ndarray | None = None,
    *,
    random_state: np.random.RandomState | None = None,
) -> np.ndarray:
    """
    Generate a temporal embedding trajectory.

    Args:
        gammas (list[float]): Consecutive angular steps.
        dim (int): Embedding dimensionality.
        t0 (np.ndarray | None): Optional initial temporal vector with shape ``[D]``.
        random_state (np.random.RandomState | None): Optional seeded random state.

    Returns:
        np.ndarray: Temporal embeddings with shape ``[T, D]``.
    """
    Ts = []
    rng = random_state if random_state is not None else np.random

    if t0 is not None:
        t = t0
    else:
        t = l2normalize(rng.randn(dim))
    Ts.append(t)
    for i in range(len(gammas)):
        t = generate_vector_by_angle(
            t,
            gammas[i],
            random_state=random_state,
        )
        Ts.append(t)

    return np.stack(Ts[1:])
