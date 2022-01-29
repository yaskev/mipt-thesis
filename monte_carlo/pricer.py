import json

import numpy as np


class Pricer:
    def __init__(self):
        with open('settings.json') as f:
            self.params = json.loads(f.read())

    def get_option_price(self, paths: np.ndarray):
        payoffs = (self.params['strike'] - paths.mean(axis=1))
        payoffs[payoffs < 0] = 0
        return np.exp(-self.params['riskFreeRate'] * self.params['timeToMaturity']) * payoffs.mean()
