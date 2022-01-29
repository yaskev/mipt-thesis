import numpy as np

from utils.typing import OptionAvgType, OptionType


def get_option_price(paths: np.ndarray,
                     strike: float,
                     risk_free_rate: float,
                     ttm: float,
                     avg_type: OptionAvgType,
                     option_type: OptionType
                     ) -> float:
    if avg_type == OptionAvgType.ARITHMETIC:
        mean = paths.mean(axis=1)
    elif avg_type == OptionAvgType.GEOMETRIC:
        mean = np.exp(np.log(paths).mean(axis=1))
    else:
        raise Exception(f'Unknown averaging type: {avg_type.value}')

    if option_type == OptionType.CALL:
        payoffs = mean - strike
    elif option_type == OptionType.PUT:
        payoffs = strike - mean
    else:
        raise Exception(f'Unknown option type: {option_type.value}')
    payoffs[payoffs < 0] = 0
    return np.exp(-risk_free_rate * ttm) * payoffs.mean()
