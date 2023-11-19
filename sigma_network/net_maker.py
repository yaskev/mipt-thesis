import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import settings
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
            np.float32).to_numpy()
        test_df_values = test_df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'price_strike_ratio']].astype(
            np.float32).to_numpy()
    elif fixed_avg_type == OptionAvgType.GEOMETRIC:
        if not analytics_mode:
            df = df[df['avg_type'] == OptionAvgType.GEOMETRIC.value]
            test_df = test_df[test_df['avg_type'] == OptionAvgType.GEOMETRIC.value]
        df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'price_strike_ratio']].astype(
            np.float32).to_numpy()
        test_df_values = test_df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'price_strike_ratio']].astype(
            np.float32).to_numpy()
    else:
        # if not analytics_mode:
        #     df['numeric_avg_type'] = df.apply(lambda row: 1 if row.avg_type == OptionAvgType.ARITHMETIC.value else 0,
        #                                       axis=1)

        if settings.SUBTRACT_INTRINSIC_VALUE:
            df['intr_value'] = df['one_year_mean'] - np.exp(-df['risk_free_rate'] * df['ttm'])
            # df.loc[df['intr_value'] < 0, 'intr_value'] = 0
            test_df['intr_value'] = test_df['one_year_mean'] - np.exp(-test_df['risk_free_rate'] * test_df['ttm'])
            # test_df.loc[test_df['intr_value'] < 0, 'intr_value'] = 0

            df['price_strike_ratio'] -= df['intr_value']
            test_df['price_strike_ratio'] -= test_df['intr_value']

            # Apply log transform
            # df.loc[df['price_strike_ratio'] == 1, 'price_strike_ratio'] = 0.9999
            # test_df.loc[test_df['price_strike_ratio'] == 1, 'price_strike_ratio'] = 0.9999
            # df.loc[df['price_strike_ratio'] <= 0, 'price_strike_ratio'] = 0.0001
            # test_df.loc[test_df['price_strike_ratio'] <= 0, 'price_strike_ratio'] = 0.0001
            # df['price_strike_ratio'] = np.log(df['price_strike_ratio'])
            # test_df['price_strike_ratio'] = np.log(test_df['price_strike_ratio'])

        df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'price_strike_ratio']].astype(
            np.float32).to_numpy()
        test_df_values = test_df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'price_strike_ratio']].astype(
            np.float32).to_numpy()

    df_target = df['volatility'].astype(np.float32).to_numpy()
    test_df_target = test_df['volatility'].astype(np.float32).to_numpy()

    if settings.SIGMA_USE_SCALER:
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
    print(f'Train MSE: {train_loss[-1] ** 0.5}')
    print(f'Val MSE: {val_loss[-1] ** 0.5}')

    if not analytics_mode:
        if not no_charts:
            create_chart(train_loss, val_loss, 'sigma_network', str(net_type.value))
        return net, x_test, y_test
    else:
        return net, x_test, y_test, train_loss, val_loss
