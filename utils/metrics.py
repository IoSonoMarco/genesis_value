import numpy as np


def score_win_prob_trajectory(p_win_left):
    """
    Compute two trajectory-level environment scores from p_win_left, reflecting environmental dynamics.

    Scores
    ------
    directional_predictability_index:
        Horizon-averaged ability of past reward advantage direction to predict
        the current better arm.

        Range is approximately [-1, 1]:
            +1 = past direction always predicts current direction
             0 = chance-level directional predictability
            -1 = past direction is systematically misleading

    mean_reward_gap:
        Average absolute difference between LEFT and RIGHT win probabilities.
        Higher values mean the better arm is more clearly better on average.

    Args:
        p_win_left (array-like):
            Sequence of LEFT win probabilities.

    Returns:
        dict:
            {
                "directional_predictability_index": float,
                "mean_reward_gap": float,
            }
    """
    p_win_left = np.asarray(p_win_left, dtype=float)

    if p_win_left.ndim != 1:
        raise ValueError("p_win_left must be a one-dimensional array.")
    if len(p_win_left) < 2:
        raise ValueError("p_win_left must contain at least two time points.")
    if np.any(~np.isfinite(p_win_left)):
        raise ValueError("p_win_left contains NaN or infinite values.")
    if np.any((p_win_left < 0) | (p_win_left > 1)):
        raise ValueError("p_win_left values must be probabilities in [0, 1].")

    # Signed advantage: positive means RIGHT is better, negative means LEFT is better.
    advantage = 1.0 - 2.0 * p_win_left

    # Score 1: mean reward gap / task easiness.
    mean_reward_gap = float(np.mean(np.abs(advantage)))

    # Score 2: horizon-averaged directional predictability.
    directional_scores = []

    for horizon in range(1, len(advantage)):
        current = advantage[horizon:]

        past_direction = np.array([
            np.sign(np.mean(advantage[t - horizon:t]))
            for t in range(horizon, len(advantage))
        ])

        current_direction = np.sign(current)

        # Remove exact ties, e.g. p_win_left == 0.5 or past mean exactly 0.
        valid = (past_direction != 0) & (current_direction != 0)

        if np.any(valid):
            accuracy = np.mean(past_direction[valid] == current_direction[valid])
            directional_scores.append(2.0 * accuracy - 1.0)

    directional_predictability_index = (
        float(np.mean(directional_scores))
        if directional_scores
        else np.nan
    )

    return {
        "directional_predictability_index": directional_predictability_index,
        "mean_reward_gap": mean_reward_gap,
    }
    

###


def early_window_ar1_predictability(p_win_left, N, R):
    """
    Fit an AR(1) model on the first N points of p_win_left and evaluate how well
    that early model predicts R later windows, each of length N.

    Parameters
    ----------
    p_win_left : array-like
        One-dimensional trajectory of LEFT win probability.

    N : int
        Length of the training window and each test window.

    R : int
        Number of later test windows.

    Returns
    -------
    dict
        {
            "score": mean predictive skill across test windows,
            "test_scores": list of per-window predictive skills,
            "test_starts": list of test-window start indices,
            "derived_lag": derived average lag between test windows,
            "ar_intercept": fitted AR(1) intercept,
            "ar_phi": fitted AR(1) coefficient,
            "test_mse": mean MSE across test windows,
        }

    Score definition
    ----------------
    For each test window:

        score = 1 - MSE_AR / MSE_baseline

    where the baseline predicts the mean of the first N points.

    Interpretation
    --------------
        score > 0  : early AR(1) model predicts future window better than early mean
        score = 0  : no better than carrying forward the early mean
        score < 0  : early dynamics are misleading for that future window
    """
    p = np.asarray(p_win_left, dtype=float)

    if p.ndim != 1:
        raise ValueError("p_win_left must be one-dimensional.")
    if np.any(~np.isfinite(p)):
        raise ValueError("p_win_left contains NaN or infinite values.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p_win_left values must be probabilities in [0, 1].")
    if not isinstance(N, int) or N < 2:
        raise ValueError("N must be an integer >= 2.")
    if not isinstance(R, int) or R < 1:
        raise ValueError("R must be an integer >= 1.")

    T = len(p)

    if 2 * N > T:
        raise ValueError("Need at least 2N trials: one training window and one test window.")

    max_unique_test_windows = T - 2 * N + 1
    if R > max_unique_test_windows:
        raise ValueError(
            f"R={R} is too large for T={T} and N={N}. "
            f"Maximum number of unique test starts is {max_unique_test_windows}."
        )

    # -----------------------------
    # 1. Fit AR(1) on first N points
    # -----------------------------
    train = p[:N]

    y = train[1:]
    x = train[:-1]

    X = np.column_stack([np.ones_like(x), x])
    intercept, phi = np.linalg.lstsq(X, y, rcond=None)[0]

    # ---------------------------------------------------
    # 2. Forecast recursively from the end of early window
    # ---------------------------------------------------
    forecast = np.full(T, np.nan)
    forecast[:N] = train

    last_pred = train[-1]

    for t in range(N, T):
        last_pred = intercept + phi * last_pred
        forecast[t] = np.clip(last_pred, 0.0, 1.0)

    # ------------------------------------------------------
    # 3. Define R test windows, optimally spaced over future
    # ------------------------------------------------------
    if R == 1:
        test_starts = np.array([T - N], dtype=int)
        derived_lag = T - 2 * N
    else:
        test_starts = np.round(
            np.linspace(N, T - N, R)
        ).astype(int)

        # Ensure strict uniqueness after rounding.
        if len(np.unique(test_starts)) != len(test_starts):
            raise ValueError(
                "R is too large relative to available trials; rounded test windows overlap "
                "with duplicate start indices."
            )

        derived_lag = float((T - 2 * N) / (R - 1))

    # ------------------------------------------------------
    # 4. Score each future test window
    # ------------------------------------------------------
    baseline_value = train.mean()

    test_scores = []
    test_mse = []

    for start in test_starts:
        stop = start + N

        y_true = p[start:stop]
        y_pred = forecast[start:stop]
        y_base = np.full(N, baseline_value)

        mse_model = np.mean((y_true - y_pred) ** 2)
        mse_baseline = np.mean((y_true - y_base) ** 2)

        if mse_baseline > 0:
            score = 1.0 - mse_model / mse_baseline
        else:
            # If the baseline is perfect, the test window is exactly equal
            # to the early mean. If AR is also perfect, score it as perfect;
            # otherwise the skill score is undefined, so return NaN.
            score = 1.0 if mse_model == 0 else np.nan

        test_scores.append(float(score))
        test_mse.append(float(mse_model))

    return {
        "score": float(np.nanmean(test_scores)),
        "test_scores": test_scores,
        "test_starts": test_starts.tolist(),
        "derived_lag": float(derived_lag),
        "ar_intercept": float(intercept),
        "ar_phi": float(phi),
        "test_mse": float(np.mean(test_mse)),
    }
    

###


def early_to_late_direction_score(p_win_left, N=20):

    p = np.asarray(p_win_left, dtype=float)
    left_adv = 2.0 * p - 1.0

    early_dir = np.sign(np.mean(left_adv[:N]))
    later_dir = np.sign(left_adv[N:])

    valid = later_dir != 0

    if early_dir == 0 or not np.any(valid):
        return np.nan

    return float(np.mean(later_dir[valid] == early_dir))