import numpy as np
import pandas as pd

import settings

N = settings.NUMBER_OF_STEPS
discount_factors = np.exp(-np.arange(0, N + 1) / N)


def add_subtracted_intrinsic_value(df: pd.DataFrame) -> pd.DataFrame:
    df['int_val'] = df.apply(_get_intrinsic_value, axis=1)
    df['subtracted_int_val'] = df['price_strike_ratio'] - df['int_val']
    df = df.drop(columns='int_val')

    return df


def _get_intrinsic_value(row) -> float:
    intrinsic_val = row.spot_strike_ratio * (discount_factors ** (row.risk_free_rate * row.ttm)).mean()\
                     - np.exp(-row.ttm * row.risk_free_rate)

    return intrinsic_val if intrinsic_val > 0 else 0

def encode_right(df: pd.DataFrame) -> pd.DataFrame:
    df = add_subtracted_intrinsic_value(df)
    df['price_strike_ratio'] = df['subtracted_int_val']
    df = df.drop(columns='subtracted_int_val')

    # Required to be able to use the network, increasing w.r.t. params
    df['spot_strike_ratio'] = -df['spot_strike_ratio']
    df['risk_free_rate'] = -df['risk_free_rate']

    return df


def decode_right(df: pd.DataFrame) -> pd.DataFrame:
    df['int_val'] = df.apply(_get_intrinsic_value, axis=1)
    df['monte_carlo_price'] = df['monte_carlo_price'] + df['int_val']
    df['net_price'] = df['net_price'] + df['int_val']
    df = df.drop(columns='int_val')

    df['spot_strike_ratio'] = -df['spot_strike_ratio']
    df['risk_free_rate'] = -df['risk_free_rate']

    return df


def only_special_decode(df: pd.DataFrame) -> pd.DataFrame:
    df['spot_strike_ratio'] = -df['spot_strike_ratio']
    df['risk_free_rate'] = -df['risk_free_rate']

    return df


def get_threshold(df: pd.DataFrame) -> pd.DataFrame:
    df['threshold'] = df.apply(_get_threshold_for_one_price, axis=1)

    return df


def _get_threshold_for_one_price(row) -> float:
    return np.exp(-row.ttm * row.risk_free_rate) / ((discount_factors ** (row.risk_free_rate * row.ttm)).mean())
