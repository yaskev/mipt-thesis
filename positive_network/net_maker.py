import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from positive_network.network import OptionsNet
from utils.plotting import create_chart
from utils.typing import OptionAvgType


def get_trained_net_and_test_set(df: pd.DataFrame, test_size: float, fixed_avg_type: OptionAvgType = None,
                                 analytics_mode: bool = False, no_charts: bool = False):

    # Polynomial features
    df['t_sigma_2'] = df['ttm'] * df['volatility'] * df['volatility']

    if fixed_avg_type == OptionAvgType.ARITHMETIC:
        if not analytics_mode:
            df = df[df['avg_type'] == OptionAvgType.ARITHMETIC.value]
        df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility',
                        't_sigma_2',
                    ]].astype(
            np.float32).to_numpy()
    elif fixed_avg_type == OptionAvgType.GEOMETRIC:
        if not analytics_mode:
            df = df[df['avg_type'] == OptionAvgType.GEOMETRIC.value]
        df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility',
                        't_sigma_2',
                        ]].astype(
            np.float32).to_numpy()
    else:
        if not analytics_mode:
            df['numeric_avg_type'] = df.apply(lambda row: 1 if row.avg_type == OptionAvgType.ARITHMETIC.value else 0,
                                              axis=1)
        df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility', 'numeric_avg_type',
                        't_sigma_2',
                        ]].astype(
            np.float32).to_numpy()

    df_target = df['price_strike_ratio'].astype(np.float32).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(df_values, df_target, test_size=test_size, random_state=42)
    net = OptionsNet(x_train.shape[1])
    train_loss, val_loss = net.fit(x_train, y_train, analytics_mode)

    if not analytics_mode:
        if not no_charts:
            create_chart(train_loss, val_loss, 'positive_network')
        return net, x_test, y_test
    else:
        return net, x_test, y_test, train_loss, val_loss
