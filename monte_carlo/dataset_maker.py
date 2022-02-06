import pandas as pd
from numpy.random import default_rng

from .path_generator import generate_paths
from .pricer import get_option_price
from utils.typing import OptionAvgType


def create_dataset(entries_cnt: int) -> pd.DataFrame:
    rng = default_rng()
    data = {
        'strike_spot_ratio': 0.5 + rng.random(entries_cnt) * 1.5,
        'ttm': 0.5 + rng.random(entries_cnt) * 1,
        'risk_free_rate': 0 + rng.random(entries_cnt) * 0.1,
        'volatility': 0.05 + rng.random(entries_cnt) * 0.5,
        'avg_type': [OptionAvgType.ARITHMETIC.value if rng.random() >= 0.5 else OptionAvgType.GEOMETRIC.value
                     for _ in range(entries_cnt)],
    }
    df = pd.DataFrame(data=data)
    df['price'] = df.apply(_get_price, axis=1)
    return df


def _get_price(row) -> float:
    paths = generate_paths(row.ttm, row.risk_free_rate, row.volatility)
    return get_option_price(paths, row.strike_spot_ratio, row.risk_free_rate, row.ttm,
                            OptionAvgType(row.avg_type))
