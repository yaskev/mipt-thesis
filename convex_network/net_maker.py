import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from convex_network.network import ConvexNet
from settings import CONVEX_MODEL_PATH, USE_PRETRAINED_NET, WITH_CI_STATS
from utils.plotting import create_chart
from utils.typing import OptionAvgType


def get_trained_net_and_test_set(df: pd.DataFrame, val_df: pd.DataFrame, test_size: float, fixed_avg_type: OptionAvgType = None,
                                 analytics_mode: bool = False, no_charts: bool = False):
    if fixed_avg_type == OptionAvgType.ARITHMETIC:
        if not analytics_mode:
            df = df[df['avg_type'] == OptionAvgType.ARITHMETIC.value]
        df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility']].astype(
            np.float32).to_numpy()
        val_df_values = val_df[
            ['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility', 'numeric_avg_type', 'left_ci',
             'right_ci']].astype(
            np.float32).to_numpy()
    elif fixed_avg_type == OptionAvgType.GEOMETRIC:
        if not analytics_mode:
            df = df[df['avg_type'] == OptionAvgType.GEOMETRIC.value]
        df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility']].astype(
            np.float32).to_numpy()
        val_df_values = val_df[
            ['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility', 'numeric_avg_type', 'left_ci',
             'right_ci']].astype(
            np.float32).to_numpy()
    else:
        if not analytics_mode:
            df['numeric_avg_type'] = df.apply(lambda row: 1 if row.avg_type == OptionAvgType.ARITHMETIC.value else 0,
                                              axis=1)
            val_df['numeric_avg_type'] = val_df.apply(
                lambda row: 1 if row.avg_type == OptionAvgType.ARITHMETIC.value else 0,
                axis=1)
        if WITH_CI_STATS:
            df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility', 'numeric_avg_type', 'left_ci', 'right_ci']].astype(
                np.float32).to_numpy()
            val_df_values = val_df[
                ['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility', 'numeric_avg_type', 'left_ci',
                 'right_ci']].astype(
                np.float32).to_numpy()
        else:
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
        net = joblib.load(CONVEX_MODEL_PATH)
        return net, x_test, y_test, val_df_values, val_df_target

    net = ConvexNet(x_train.shape[1], torch.tensor([0]))
    train_loss, val_loss = net.fit(x_train, y_train, val_df_values, val_df_target, analytics_mode)
    if not analytics_mode:
        if not no_charts:
            create_chart(train_loss, val_loss, 'convex_network')
        return net, x_test, y_test, val_df_values, val_df_target
    else:
        return net, x_test, y_test, val_df_values, val_df_target, train_loss, val_loss
