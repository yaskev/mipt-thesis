import pandas as pd
from numba import njit
from numpy.random import default_rng
import numpy as np
import matplotlib.pyplot as plt

from settings import NUMBER_OF_PATHS, NUMBER_OF_STEPS, USE_ANTITHETIC_VARIATES, PATH_RANDOM_SEED
from utils.plotting import plot_many_paths

rng = default_rng(PATH_RANDOM_SEED)

def generate_paths(ttm: float,
                   risk_free_rate: float,
                   volatility: float,
                   spot_strike_ratio: float,
                   num_of_values: int = NUMBER_OF_PATHS
                   ) -> np.ndarray:
    normal_samples = _get_standard_normal(num_of_values=num_of_values,
                                          num_of_steps=NUMBER_OF_STEPS + 1)
    if USE_ANTITHETIC_VARIATES and num_of_values > 1:
        normal_samples = np.append(normal_samples, -normal_samples, axis=0)
    return create_paths(ttm, normal_samples, risk_free_rate, volatility, spot_strike_ratio)


@njit(fastmath=True)
def create_paths(ttm: float,
                 paths: np.ndarray,
                 risk_free_rate: float,
                 volatility: float,
                 spot_strike_ratio: float):
    paths[:, 0] = spot_strike_ratio
    dt = ttm / NUMBER_OF_STEPS
    for i in range(NUMBER_OF_STEPS):
        paths[:, i + 1] = (paths[:, i] * np.exp((risk_free_rate
                                                 - 0.5 * volatility ** 2) * dt
                                                + volatility
                                                * np.sqrt(dt) * paths[:, i + 1]))
    return paths


def _get_standard_normal(num_of_values: int,
                         num_of_steps: int) -> np.ndarray:
    values = rng.standard_normal((num_of_values, num_of_steps))

    if num_of_values == 1:
        return values

    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return (values - mean) / std


def plot_paths(dataset: pd.DataFrame, folder: str = 'monte_carlo/charts'):
    for i, row in dataset.iterrows():
        paths = generate_paths(row['ttm'], row['risk_free_rate'], row['volatility'], row['spot_strike_ratio'])
        plot_many_paths(paths, folder)
