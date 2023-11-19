import numpy as np
import pandas as pd

import settings
from utils.typing import OptionAvgType

N = settings.NUMBER_OF_STEPS
discount_factors = np.exp(-np.arange(0, N + 1) / N)

ADD_VALUE_TO_LOG = 10

def add_subtracted_intrinsic_value(df: pd.DataFrame) -> pd.DataFrame:
    df['int_val'] = df.apply(_get_price_without_int_val, axis=1)
    df['subtracted_int_val'] = df['price_strike_ratio'] - df['int_val']
    df = df.drop(columns='int_val')

    return df

def _get_price_without_int_val(row) -> float:
    if row.avg_type == OptionAvgType.ARITHMETIC.value:
        intrinsic_val = row.spot_strike_ratio * (discount_factors ** (row.risk_free_rate * row.ttm)).mean()\
                         - np.exp(-row.ttm * row.risk_free_rate)
    else:
        raise Exception('GEOM is not supported yet')
        # wrong formula
        intrinsic_val = np.exp(np.log(discount_factors).mean() * row.ttm * row.risk_free_rate)\
                        - np.exp(-row.ttm * row.risk_free_rate)

    return intrinsic_val if intrinsic_val > 0 else 0

def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = add_subtracted_intrinsic_value(df)
    df['price_strike_ratio'] = df['subtracted_int_val']
    df = df.drop(columns='subtracted_int_val')
    df.loc[df['price_strike_ratio'] <= 0, 'price_strike_ratio'] = 0.0001
    # df['price_strike_ratio'] = np.log(df['price_strike_ratio']) + ADD_VALUE_TO_LOG
    df['price_strike_ratio'] = np.log(df['price_strike_ratio'])

    return df

def decode(df: pd.DataFrame) -> pd.DataFrame:
    df['int_val'] = df.apply(_get_price_without_int_val, axis=1)

    # df['monte_carlo_price'] = np.exp(df['monte_carlo_price'] - ADD_VALUE_TO_LOG)
    df['monte_carlo_price'] = np.exp(df['monte_carlo_price'])
    df['monte_carlo_price'] += df['int_val']

    # df['net_price'] = np.exp(df['net_price'] - ADD_VALUE_TO_LOG)
    df['net_price'] = np.exp(df['net_price'])
    df['net_price'] += df['int_val']

    df = df.drop(columns='int_val')

    return df
