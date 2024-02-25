import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from semipositive_network.network import OptionsNet
from settings import USE_PRETRAINED_NET, SEMIPOSITIVE_MODEL_PATH
from utils.plotting import create_chart
from utils.typing import OptionAvgType


def get_trained_net_and_test_set(df: pd.DataFrame, val_df: pd.DataFrame, test_size: float, fixed_avg_type: OptionAvgType = None,
                                 analytics_mode: bool = False, no_charts: bool = False):
    df['numeric_avg_type'] = df.apply(lambda row: 1 if row.avg_type == OptionAvgType.ARITHMETIC.value else 0,
                                      axis=1)
    val_df['numeric_avg_type'] = val_df.apply(
        lambda row: 1 if row.avg_type == OptionAvgType.ARITHMETIC.value else 0,
        axis=1)
    df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility', 'numeric_avg_type']].astype(
        np.float32).to_numpy()
    val_df_values = val_df[
        ['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility', 'numeric_avg_type']].astype(
        np.float32).to_numpy()

    df_target = df['price_strike_ratio'].astype(np.float32).to_numpy()
    val_df_target = val_df['price_strike_ratio'].astype(np.float32).to_numpy()

    if test_size == 1 and USE_PRETRAINED_NET:
        x_test = df_values
        y_test = df_target
    else:
        x_train, x_test, y_train, y_test = train_test_split(df_values, df_target, test_size=test_size, random_state=42)

    if USE_PRETRAINED_NET:
        net = joblib.load(SEMIPOSITIVE_MODEL_PATH)
        return net, x_test, y_test, val_df_values, val_df_target

    net = OptionsNet(x_train.shape[1])
    train_loss, val_loss = net.fit(x_train, y_train, val_df_values, val_df_target, analytics_mode)

    if not analytics_mode:
        if not no_charts:
            create_chart(train_loss, val_loss, 'semipositive_network')
        return net, x_test, y_test, val_df_values, val_df_target
    else:
        return net, x_test, y_test, val_df_values, val_df_target, train_loss, val_loss
