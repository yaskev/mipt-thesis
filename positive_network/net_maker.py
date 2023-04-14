import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from positive_network.network import OptionsNet
from utils.plotting import create_chart
from utils.typing import OptionAvgType


def get_trained_net_and_test_set(df: pd.DataFrame, test_size: float, fixed_avg_type: OptionAvgType = None,
                                 analytics_mode: bool = False, no_charts: bool = False):
    if fixed_avg_type == OptionAvgType.ARITHMETIC:
        if not analytics_mode:
            df = df[df['avg_type'] == OptionAvgType.ARITHMETIC.value]
        df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility']].astype(
            np.float32).to_numpy()
    elif fixed_avg_type == OptionAvgType.GEOMETRIC:
        if not analytics_mode:
            df = df[df['avg_type'] == OptionAvgType.GEOMETRIC.value]
        df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility']].astype(
            np.float32).to_numpy()
    else:
        if not analytics_mode:
            df['numeric_avg_type'] = df.apply(lambda row: 1 if row.avg_type == OptionAvgType.ARITHMETIC.value else 0,
                                              axis=1)
        df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility', 'numeric_avg_type']].astype(
            np.float32).to_numpy()

    df = add_time_value_column(df)

    df_target = df['time_value'].astype(np.float32).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(df_values, df_target, test_size=test_size, random_state=42)
    net = OptionsNet(x_train.shape[1])
    train_loss, val_loss = net.fit(x_train, y_train, analytics_mode)

    if not analytics_mode:
        if not no_charts:
            create_chart(train_loss, val_loss, 'positive_network')
        return net, x_test, y_test
    else:
        return net, x_test, y_test, train_loss, val_loss

def add_time_value_column(df: pd.DataFrame) -> pd.DataFrame:
    intrinsic_value = df['spot_strike_ratio'] - np.exp(-df['risk_free_rate'] * df['ttm'])
    intrinsic_value[intrinsic_value < 0] = 0
    df['time_value'] = df['price_strike_ratio'] - intrinsic_value

    print(df[df['time_value'] < 0].shape)

    return df


def remove_time_value_column(df: pd.DataFrame) -> pd.DataFrame:
    intrinsic_value = df['spot_strike_ratio'] - np.exp(-df['risk_free_rate'] * df['ttm'])
    intrinsic_value[intrinsic_value < 0] = 0
    df['net_price'] = df['time_value'] + intrinsic_value

    return df

