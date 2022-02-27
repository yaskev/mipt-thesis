from typing import Tuple

import numpy as np

from settings import BATCH_SIZE


def get_batches(dataset: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = dataset
    n_samples = X.shape[0]

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start in range(0, n_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], Y[batch_idx]
