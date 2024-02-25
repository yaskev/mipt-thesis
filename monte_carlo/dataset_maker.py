from typing import Tuple

import numpy as np
import pandas as pd
from numpy.random import default_rng

import settings
from settings import OPTIONS_PARAMS_RANDOM_SEED
from .path_generator import generate_paths
from .pricer import get_option_price_and_ci
from utils.typing import OptionAvgType
from scipy.stats import beta


def create_dataset(entries_cnt: int) -> pd.DataFrame:
    rng = default_rng(OPTIONS_PARAMS_RANDOM_SEED)
    # data = {
    #     'spot_strike_ratio': 0.5 + beta.rvs(0.5, 0.5, size=entries_cnt) * 1.5,
    #     'ttm': 0.5 + beta.rvs(0.5, 0.5, size=entries_cnt) * 1,
    #     'risk_free_rate': 0 + beta.rvs(0.5, 0.5, size=entries_cnt) * 0.2,
    #     'volatility': 0.05 + beta.rvs(0.5, 0.5, size=entries_cnt) * 0.5,
    #     'avg_type': [OptionAvgType.ARITHMETIC.value for _ in range(entries_cnt)],
    # }

    # data = {
    #     'spot_strike_ratio': 0.5 + rng.random(entries_cnt) * 0.45,
    #     'ttm': 0.5 + rng.random(entries_cnt) * 1,
    #     'risk_free_rate': 0 + rng.random(entries_cnt) * 0.05,
    #     'volatility': 0.05 + rng.random(entries_cnt) * 0.5,
    #     'avg_type': [OptionAvgType.ARITHMETIC.value for _ in range(entries_cnt)],
    # }

    data = {
        'spot_strike_ratio': 1 + rng.random(entries_cnt) * 0.5,
        'ttm': 0.5 + rng.random(entries_cnt) * 1,
        'risk_free_rate': 0 + rng.random(entries_cnt) * 0.2,
        'volatility': 0.05 + rng.random(entries_cnt) * 0.5,
        'avg_type': [OptionAvgType.ARITHMETIC.value for _ in range(entries_cnt)],
    }

    df = pd.DataFrame(data=data)

    if settings.START_SHIFT:
        price_and_ci_df = df.apply(_get_price_after_shift, axis=1, result_type='expand')
        price_and_ci_df.columns = ['price_strike_ratio', 'left_ci', 'right_ci', 'one_year_mean', 'new_spot_strike_ratio']
        df['spot_strike_ratio'] = price_and_ci_df['new_spot_strike_ratio']
        del price_and_ci_df['new_spot_strike_ratio']
    else:
        price_and_ci_df = df.apply(_get_price, axis=1, result_type='expand')
        price_and_ci_df.columns = ['price_strike_ratio', 'left_ci', 'right_ci']

    return pd.concat([df, price_and_ci_df], axis=1)


def _get_price(row) -> Tuple[float, float, float]:
    paths = generate_paths(row.ttm, row.risk_free_rate, row.volatility, row.spot_strike_ratio)
    return get_option_price_and_ci(paths, row.risk_free_rate,
                                   OptionAvgType(row.avg_type), row.ttm)

def _get_price_after_shift(row) -> Tuple[float, float, float, float, float]:
    # Experimental feature, do not use
    pre_path = generate_paths(settings.START_SHIFT, row.risk_free_rate, row.volatility, row.spot_strike_ratio, num_of_values=1)[0]
    new_spot_strike_ratio = pre_path[-1]

    if OptionAvgType(row.avg_type) == OptionAvgType.ARITHMETIC:
        pre_mean = pre_path.mean()
    elif OptionAvgType(row.avg_type) == OptionAvgType.GEOMETRIC:
        pre_mean = np.exp(np.log(pre_path).mean())
    else:
        raise Exception(f'Unknown averaging type: {row.avg_type}')

    paths = generate_paths(row.ttm, row.risk_free_rate, row.volatility, new_spot_strike_ratio)

    price, left_ci, right_ci = get_option_price_and_ci(paths, row.risk_free_rate,
                            OptionAvgType(row.avg_type), row.ttm, add_mean=pre_mean)

    return price, left_ci, right_ci, pre_mean, new_spot_strike_ratio
