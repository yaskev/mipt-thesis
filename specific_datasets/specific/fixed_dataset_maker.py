import pandas as pd
from numpy.random import default_rng

from monte_carlo.dataset_maker import _get_price
from settings import OPTIONS_PARAMS_RANDOM_SEED
from utils.typing import OptionAvgType


VARIABLE_PARAMS_NUMBER = 4
ENTRIES_NUMBER = 10000


def create_fixed_datasets():
    rng = default_rng(OPTIONS_PARAMS_RANDOM_SEED)
    for i in range(VARIABLE_PARAMS_NUMBER):
        print(i)
        data = {
            'spot_strike_ratio': (0.5 + rng.random(ENTRIES_NUMBER)) if i == 0 else 1,
            'ttm': (0.5 + rng.random(ENTRIES_NUMBER)) if i == 1 else 1,
            'risk_free_rate': (rng.random(ENTRIES_NUMBER) * 0.2) if i == 2 else 0.1,
            'volatility': (0.05 + rng.random(ENTRIES_NUMBER) * 0.5) if i == 3 else 0.2,
            'avg_type': OptionAvgType.ARITHMETIC.value,
        }
        df = pd.DataFrame(data=data)
        price_and_ci_df = df.apply(_get_price, axis=1, result_type='expand')
        price_and_ci_df.columns = ['price_strike_ratio', 'left_ci', 'right_ci']
        df = pd.concat([df, price_and_ci_df], axis=1)

        idx_to_param = {
            0: 'spot',
            1: 'ttm',
            2: 'rate',
            3: 'vol'
        }
        df.to_csv(f'../fixed_{idx_to_param[i]}_dataset.csv', index=False, float_format='%.4f')


if __name__ == '__main__':
    create_fixed_datasets()
