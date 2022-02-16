from typing import Tuple

import pandas as pd
from numpy.random import default_rng

from settings import OPTIONS_PARAMS_RANDOM_SEED
from .path_generator import generate_paths
from .pricer import get_option_price_and_ci
from utils.typing import OptionAvgType


def create_dataset(entries_cnt: int) -> pd.DataFrame:
    rng = default_rng(OPTIONS_PARAMS_RANDOM_SEED)
    data = {
        'spot_strike_ratio': 0.5 + rng.random(entries_cnt) * 1.5,
        'risk_free_rate': 0 + rng.random(entries_cnt) * 0.2,
        'volatility': 0.05 + rng.random(entries_cnt) * 0.95,
        'avg_type': [OptionAvgType.ARITHMETIC.value if rng.random() >= 0.5 else OptionAvgType.GEOMETRIC.value
                     for _ in range(entries_cnt)],
    }
    df = pd.DataFrame(data=data)
    price_and_ci_df = df.apply(_get_price, axis=1, result_type='expand')
    price_and_ci_df.columns = ['price_strike_ratio', 'left_ci', 'right_ci']
    return pd.concat([df, price_and_ci_df], axis=1)


def _get_price(row) -> Tuple[float, float, float]:
    paths = generate_paths(row.risk_free_rate, row.volatility, row.spot_strike_ratio)
    return get_option_price_and_ci(paths, row.risk_free_rate,
                                   OptionAvgType(row.avg_type))
