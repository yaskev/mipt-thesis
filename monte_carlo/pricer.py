from typing import Tuple

import numpy as np
from scipy.stats import norm

import settings
from utils.typing import OptionAvgType


def get_option_price_and_ci(paths: np.ndarray,
                            strike_spot_ratio: float,
                            risk_free_rate: float,
                            ttm: float,
                            avg_type: OptionAvgType
                            ) -> Tuple[float, float, float]:
    if avg_type == OptionAvgType.ARITHMETIC:
        mean = paths.mean(axis=1)
    elif avg_type == OptionAvgType.GEOMETRIC:
        mean = np.exp(np.log(paths).mean(axis=1))
    else:
        raise Exception(f'Unknown averaging type: {avg_type.value}')

    payoffs = mean - strike_spot_ratio
    payoffs[payoffs < 0] = 0

    payoff_mean = np.exp(-risk_free_rate * ttm) * payoffs.mean()
    payoff_std = np.exp(-risk_free_rate * ttm) * payoffs.std()
    ci_multiplier = norm.ppf((1 + settings.CONFIDENCE_INTERVAL) / 2)
    return (payoff_mean,
            max(payoff_mean - payoff_std * ci_multiplier / np.sqrt(settings.NUMBER_OF_PATHS), 0),
            payoff_mean + payoff_std * ci_multiplier / np.sqrt(settings.NUMBER_OF_PATHS))
