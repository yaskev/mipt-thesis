import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from positive_network.network import OptionsNet
from utils.typing import OptionAvgType


def get_trained_net_and_test_set(df: pd.DataFrame, test_size: float, fixed_avg_type: OptionAvgType = None):
    if fixed_avg_type == OptionAvgType.ARITHMETIC:
        df = df[df['avg_type'] == OptionAvgType.ARITHMETIC.value]
        df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility']].astype(
            np.float32).to_numpy()
    elif fixed_avg_type == OptionAvgType.GEOMETRIC:
        df = df[df['avg_type'] == OptionAvgType.GEOMETRIC.value]
        df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility']].astype(
            np.float32).to_numpy()
    else:
        df['numeric_avg_type'] = df.apply(lambda row: 1 if row.avg_type == OptionAvgType.ARITHMETIC.value else 0, axis=1)
        df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility', 'numeric_avg_type']].astype(
            np.float32).to_numpy()

    df_target = df['price_strike_ratio'].astype(np.float32).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(df_values, df_target, test_size=test_size)
    net = OptionsNet(x_train.shape[1])
    net.fit(x_train, y_train)

    return net, x_test, y_test
