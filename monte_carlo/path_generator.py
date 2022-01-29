import json

from numpy.random import default_rng
import numpy as np


class PathGenerator:
    def __init__(self):
        with open('settings.json') as f:
            self.params = json.loads(f.read())

    def generate_paths(self) -> np.ndarray:
        paths = self._get_standard_normal(num_of_values=self.params['numberOfPaths'],
                                          num_of_steps=self.params['numberOfSteps'] + 1)
        paths[:, 0] = self.params['initialSpotPrice']
        dt = self.params['timeToMaturity'] / self.params['numberOfSteps']
        for i in range(self.params['numberOfSteps']):
            paths[:, i + 1] = (paths[:, i] * np.exp((self.params['riskFreeRate']
                                                     - 0.5 * self.params['volatility'] ** 2) * dt
                                                    + self.params['volatility']
                                                    * np.sqrt(dt) * paths[:, i + 1]))
        return paths

    def _get_standard_normal(self, num_of_values: int, num_of_steps: int) -> np.ndarray:
        rng = default_rng()
        values = rng.standard_normal((num_of_values, num_of_steps))
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        return (values - mean) / std
