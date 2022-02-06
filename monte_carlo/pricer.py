import numpy as np

from utils.typing import OptionAvgType


def get_option_price(paths: np.ndarray,
                     strike_spot_ratio: float,
                     risk_free_rate: float,
                     ttm: float,
                     avg_type: OptionAvgType
                     ) -> float:
    if avg_type == OptionAvgType.ARITHMETIC:
        mean = paths.mean(axis=1)
    elif avg_type == OptionAvgType.GEOMETRIC:
        mean = np.exp(np.log(paths).mean(axis=1))
    else:
        raise Exception(f'Unknown averaging type: {avg_type.value}')

    payoffs = mean - strike_spot_ratio
    payoffs[payoffs < 0] = 0
    return np.exp(-risk_free_rate * ttm) * payoffs.mean()
