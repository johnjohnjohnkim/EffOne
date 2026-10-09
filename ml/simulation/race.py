"""Monte Carlo of one race: from expected finishing scores and DNF chances to outcome probabilities.

Each simulated race draws a noisy performance for every driver, `score + noise`, with noise
growing with the score (`sigma * score ** gamma`; a front-runner is more predictable than the
middle of the field). Drivers who retire in that draw are placed behind everyone who finished, in
random order. Because every simulated race is a complete finishing order, the outputs are
consistent by construction: win probabilities sum to 1, podium probabilities to 3 and top-10
probabilities to 10 (or to the field size when fewer than 10 are entered).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

RETIRED_OFFSET = 1e6  # sorts retirements behind every finisher


@dataclass(frozen=True)
class SimParams:
    sigma: float = 3.0
    gamma: float = 0.0
    n_sims: int = 20000
    seed: int = 0


def draw_noise(n_sims: int, n_drivers: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard normal performance noise, uniform DNF draws and uniform retirement-order draws.

    Returned separately so a parameter search can reuse the same random numbers for every setting
    (common random numbers make differences between settings far less noisy).
    """
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((n_sims, n_drivers)),
        rng.random((n_sims, n_drivers)),
        rng.random((n_sims, n_drivers)),
    )


def simulate_positions(
    scores: np.ndarray,
    p_dnf: np.ndarray,
    sigma: float,
    gamma: float,
    noise: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """(finishing positions 1..n, retired flags), each shaped (n_sims, n_drivers)."""
    scores = np.asarray(scores, dtype=float)
    p_dnf = np.asarray(p_dnf, dtype=float)
    if np.isnan(scores).any() or np.isnan(p_dnf).any():
        raise ValueError("scores and DNF probabilities must not contain NaN")
    if ((p_dnf < 0) | (p_dnf > 1)).any():
        raise ValueError("DNF probabilities must be between 0 and 1")
    eps, u_dnf, u_order = noise
    spread = sigma * np.maximum(scores, 1.0) ** gamma
    performance = scores + spread * eps
    retired = u_dnf < p_dnf
    key = np.where(retired, RETIRED_OFFSET + u_order, performance)
    order = np.argsort(key, axis=1)
    positions = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    positions[rows, order] = np.arange(1, order.shape[1] + 1)
    return positions, retired


def summarize(positions: np.ndarray, retired: np.ndarray) -> pd.DataFrame:
    """Per-driver probabilities (columns in driver order) from simulated finishing orders."""
    return pd.DataFrame(
        {
            "p_win": (positions == 1).mean(axis=0),
            "p_podium": (positions <= 3).mean(axis=0),
            "p_top10": (positions <= 10).mean(axis=0),
            "p_dnf": retired.mean(axis=0),
            "expected_position": positions.mean(axis=0),
        }
    )


def simulate_race(
    scores: np.ndarray, p_dnf: np.ndarray, params: SimParams | None = None
) -> pd.DataFrame:
    """Outcome probabilities for one race, one row per driver, in the order given."""
    params = params or SimParams()
    noise = draw_noise(params.n_sims, len(scores), params.seed)
    positions, retired = simulate_positions(scores, p_dnf, params.sigma, params.gamma, noise)
    return summarize(positions, retired)
