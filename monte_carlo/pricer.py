from typing import Tuple

import numpy as np
from scipy.stats import norm

from settings import USE_ANTITHETIC_VARIATES, NUMBER_OF_PATHS, CONFIDENCE_INTERVAL
from utils.typing import OptionAvgType


def get_option_price_and_ci(paths: np.ndarray,
                            risk_free_rate: float,
                            avg_type: OptionAvgType,
                            ttm: float,
                            ) -> Tuple[float, float, float]:
    if avg_type == OptionAvgType.ARITHMETIC:
        mean = paths.mean(axis=1)
    elif avg_type == OptionAvgType.GEOMETRIC:
        mean = np.exp(np.log(paths).mean(axis=1))
    else:
        raise Exception(f'Unknown averaging type: {avg_type.value}')

    payoffs = mean - 1
    payoffs[payoffs < 0] = 0

    if USE_ANTITHETIC_VARIATES:
        payoffs = (payoffs[:NUMBER_OF_PATHS] + payoffs[NUMBER_OF_PATHS:]) / 2

    payoff_mean = np.exp(-risk_free_rate * ttm) * payoffs.mean()
    payoff_std = np.exp(-risk_free_rate * ttm) * payoffs.std()
    ci_multiplier = norm.ppf((1 + CONFIDENCE_INTERVAL) / 2)
    return (payoff_mean,
            max(payoff_mean - payoff_std * ci_multiplier / np.sqrt(NUMBER_OF_PATHS), 0),
            payoff_mean + payoff_std * ci_multiplier / np.sqrt(NUMBER_OF_PATHS))
