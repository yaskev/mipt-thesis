from typing import Tuple

import numpy as np
import pandas as pd

import settings

N = settings.NUMBER_OF_STEPS
ADD_VALUE_TO_LOG = 20
discount_factors = np.exp(-np.arange(0, N + 1) / N)


def add_subtracted_intrinsic_value(df: pd.DataFrame) -> pd.DataFrame:
    df['int_val'] = df.apply(_get_intrinsic_value, axis=1)
    df['subtracted_int_val'] = df['price_strike_ratio'] - df['int_val']
    df = df.drop(columns='int_val')

    return df

def get_intrinsic_values_greeks(df: pd.DataFrame) -> pd.DataFrame:
    # assumed only right part is passed here
    intr_greeks = df.apply(_get_one_intrinsic_value_greeks, axis=1, result_type='expand')
    intr_greeks.columns = ['delta', 'vega', 'theta', 'rho']

    return intr_greeks

def _get_one_intrinsic_value_greeks(row) -> Tuple[float, float, float, float]:
    # assumed only right part is passed here
    delta = (discount_factors ** (row.risk_free_rate * row.ttm)).mean()

    vega = 0

    # Minus is already in the formula
    theta = (
        row.spot_strike_ratio * row.risk_free_rate / (N ** 2) *
        (discount_factors ** (row.risk_free_rate * row.ttm) * np.arange(0, N + 1)).mean() -
         row.risk_free_rate * np.exp(-row.risk_free_rate * row.ttm)
    )

    rho = (
        -(row.spot_strike_ratio * row.ttm / (N ** 2)) *
        (discount_factors ** (row.risk_free_rate * row.ttm) * np.arange(0, N + 1)).mean() +
        row.ttm * np.exp(-row.risk_free_rate * row.ttm)
    )

    # delta, vega, theta, rho
    return delta, vega, theta, rho

def _get_intrinsic_value(row) -> float:
    intrinsic_val = row.spot_strike_ratio * (discount_factors ** (row.risk_free_rate * row.ttm)).mean()\
                     - np.exp(-row.ttm * row.risk_free_rate)

    return intrinsic_val if intrinsic_val > 0 else 0


def encode_left(df: pd.DataFrame) -> pd.DataFrame:
    return log_price_encode(df)

def decode_left(df: pd.DataFrame) -> pd.DataFrame:
    return log_price_decode(df)


def encode_right(df: pd.DataFrame) -> pd.DataFrame:
    df = add_subtracted_intrinsic_value(df)
    df['price_strike_ratio'] = df['subtracted_int_val']
    df = df.drop(columns='subtracted_int_val')

    df = log_price_encode(df)

    return right_special_encode_decode(df)


def decode_right(df: pd.DataFrame) -> pd.DataFrame:
    df = log_price_decode(df)

    df['int_val'] = df.apply(_get_intrinsic_value, axis=1)
    df['monte_carlo_price'] = df['monte_carlo_price'] + df['int_val']
    df['net_price'] = df['net_price'] + df['int_val']
    df = df.drop(columns='int_val')

    return right_special_encode_decode(df)


def right_special_encode_decode(df: pd.DataFrame) -> pd.DataFrame:
    # Required to be able to use network, increasing w.r.t. params
    df['spot_strike_ratio'] = -df['spot_strike_ratio']
    df['risk_free_rate'] = -df['risk_free_rate']

    return df


def log_price_encode(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df['price_strike_ratio'] <= 0, 'price_strike_ratio'] = 0.0001
    df['price_strike_ratio'] = np.log(df['price_strike_ratio']) + ADD_VALUE_TO_LOG

    return df


def log_price_decode(df: pd.DataFrame) -> pd.DataFrame:
    df['monte_carlo_price'] = np.exp(df['monte_carlo_price'] - ADD_VALUE_TO_LOG)
    df['net_price'] = np.exp(df['net_price'] - ADD_VALUE_TO_LOG)

    return df


def get_threshold(df: pd.DataFrame) -> pd.DataFrame:
    df['threshold'] = df.apply(_get_threshold_for_one_price, axis=1)

    return df


def _get_threshold_for_one_price(row) -> float:
    return np.exp(-row.ttm * row.risk_free_rate) / ((discount_factors ** (row.risk_free_rate * row.ttm)).mean())
