import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from settings import USE_PRETRAINED_NET, SIGMA_MODEL_PATH
from sigma_network.network import SigmaNet, SigmaNetType
from utils.plotting import create_chart
from utils.typing import OptionAvgType


def get_trained_net_and_test_set(df: pd.DataFrame, test_df: pd.DataFrame, test_size: float, fixed_avg_type: OptionAvgType = None,
                                 analytics_mode: bool = False, no_charts: bool = False, net_type: SigmaNetType = SigmaNetType.MULTILAYER_FFNN):
    if fixed_avg_type == OptionAvgType.ARITHMETIC:
        if not analytics_mode:
            df = df[df['avg_type'] == OptionAvgType.ARITHMETIC.value]
            test_df = test_df[test_df['avg_type'] == OptionAvgType.ARITHMETIC.value]
        df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'price_strike_ratio']].astype(
            np.float64).to_numpy()
        test_df_values = test_df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'price_strike_ratio']].astype(
            np.float64).to_numpy()
    elif fixed_avg_type == OptionAvgType.GEOMETRIC:
        if not analytics_mode:
            df = df[df['avg_type'] == OptionAvgType.GEOMETRIC.value]
            test_df = test_df[test_df['avg_type'] == OptionAvgType.GEOMETRIC.value]
        df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'price_strike_ratio']].astype(
            np.float64).to_numpy()
        test_df_values = test_df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'price_strike_ratio']].astype(
            np.float64).to_numpy()
    else:
        # if not analytics_mode:
        #     df['numeric_avg_type'] = df.apply(lambda row: 1 if row.avg_type == OptionAvgType.ARITHMETIC.value else 0,
        #                                       axis=1)
        df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'price_strike_ratio']].astype(
            np.float64).to_numpy()
        test_df_values = test_df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'price_strike_ratio']].astype(
            np.float64).to_numpy()

    df_target = df['volatility'].astype(np.float64).to_numpy()
    test_df_target = test_df['volatility'].astype(np.float64).to_numpy()

    scaler = MinMaxScaler()
    scaler.fit(df_values)
    df_values = scaler.transform(df_values)
    test_df_values = scaler.transform(test_df_values)

    if USE_PRETRAINED_NET:
        x_test = df_values
        y_test = df_target
        net = joblib.load(SIGMA_MODEL_PATH)

        return net, x_test, y_test

    x_train, x_test, y_train, y_test = train_test_split(df_values, df_target, test_size=test_size, random_state=42)

    net = SigmaNet(x_train.shape[1], net_type)
    train_loss, val_loss = net.fit(x_train, y_train, test_df_values, test_df_target, analytics_mode)

    if not analytics_mode:
        if not no_charts:
            create_chart(train_loss, val_loss, 'sigma_network', str(net_type.value))
        return net, x_test, y_test
    else:
        return net, x_test, y_test, train_loss, val_loss
