import datetime
import os
import typing

import joblib
import numpy as np
import pandas as pd

from double_net_pricer.intrinsic import get_threshold, encode_right, decode_right, right_special_encode_decode, \
    log_price_decode, log_price_encode
from main import make_predicted_df
from positive_network.network import OptionsNet as PositiveNet
from semipositive_network.network import OptionsNet as SemiPositiveNet
from utils.plotting import create_chart
from utils.typing import OptionAvgType

IN_FEATURES_NUM = 4


# TODO: Use different number of averaging steps for different ttm (not always 365)

class DoubleNetPricer:
    def __init__(self, left_net, right_net):
        self.left_net = left_net
        self.right_net = right_net

        if left_net is None:
            self.left_net = PositiveNet(IN_FEATURES_NUM)

        if right_net is None:
            self.right_net = SemiPositiveNet(IN_FEATURES_NUM)

    # Expect the following fields: 'spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility', 'avg_type'
    # Target field: 'price_strike_ratio'
    def fit(self, df: pd.DataFrame, val_df: pd.DataFrame, which = 'both'):
        left_df, right_df = self._split_for_left_and_right_net_df(df)
        left_val_df, right_val_df = self._split_for_left_and_right_net_df(val_df)

        right_df = encode_right(right_df)
        right_val_df = encode_right(right_val_df)
        left_df = log_price_encode(left_df)
        left_val_df = log_price_encode(left_val_df)

        left_df_values = left_df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility']].astype(
            np.float32).to_numpy()
        left_df_target = left_df['price_strike_ratio'].astype(np.float32).to_numpy()
        left_val_df_values = left_val_df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility']].astype(
            np.float32).to_numpy()
        left_val_df_target = left_val_df['price_strike_ratio'].astype(np.float32).to_numpy()

        right_df_values = right_df[
            ['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility']].astype(
            np.float32).to_numpy()
        right_df_target = right_df['price_strike_ratio'].astype(np.float32).to_numpy()
        right_val_df_values = right_val_df[
            ['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility']].astype(
            np.float32).to_numpy()
        right_val_df_target = right_val_df['price_strike_ratio'].astype(np.float32).to_numpy()

        if which != 'right':
            left_train_loss, left_val_loss = self.left_net.fit(left_df_values, left_df_target, left_val_df_values, left_val_df_target)
            create_chart(left_train_loss, left_val_loss, '', 'left-')
            joblib.dump(self.left_net, os.path.join('models', f'left-{datetime.datetime.now()}.sav'))
        if which != 'left':
            right_train_loss, right_val_loss = self.right_net.fit(right_df_values, right_df_target, right_val_df_values, right_val_df_target)
            create_chart(right_train_loss, right_val_loss, '', 'right-')
            joblib.dump(self.right_net, os.path.join('models', f'right-{datetime.datetime.now()}.sav'))


    # Expect the following fields: 'spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility', 'avg_type'
    # Target field: 'price_strike_ratio'
    def predict_split(self, df: pd.DataFrame, no_decode=False):
        left_df, right_df = self._split_for_left_and_right_net_df(df)
        left_df = log_price_encode(left_df)
        right_df = encode_right(right_df)

        left_df_values = left_df[
            ['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility']].astype(
            np.float32).to_numpy()
        left_df_target = left_df['price_strike_ratio'].astype(np.float32).to_numpy()
        right_df_values = right_df[
            ['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility']].astype(
            np.float32).to_numpy()
        right_df_target = right_df['price_strike_ratio'].astype(np.float32).to_numpy()

        left_prices = self.left_net.predict(left_df_values).detach().numpy()
        right_prices = self.right_net.predict(right_df_values).detach().numpy()

        left_answer_df = make_predicted_df(left_df_values, left_df_target, left_prices, fixed_avg_type=OptionAvgType.ARITHMETIC)
        right_answer_df = make_predicted_df(right_df_values, right_df_target, right_prices, fixed_avg_type=OptionAvgType.ARITHMETIC)

        if not no_decode:
            right_answer_df = decode_right(right_answer_df)
        else:
            right_answer_df = log_price_decode(right_special_encode_decode(right_answer_df))

        left_answer_df = log_price_decode(left_answer_df)

        return left_answer_df, right_answer_df

    def predict(self, df: pd.DataFrame, no_decode=False):
        left, right = self.predict_split(df, no_decode)

        return pd.concat([left, right], ignore_index=True)


    # First value: Time value increases w.r.t. moneyness
    # Second value: Time value decreases w.r.t. moneyness
    def _split_for_left_and_right_net_df(self, df: pd.DataFrame) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        df = get_threshold(df)
        left = df[df['spot_strike_ratio'] <= df['threshold']]
        right = df[df['spot_strike_ratio'] > df['threshold']]

        del df['threshold']
        del left['threshold']
        del right['threshold']

        return left, right


if __name__ == '__main__':
    left_model = joblib.load('models/left-2024-03-04 15:40:51.111126.sav')
    right_model = joblib.load('models/right-2024-03-04 14:56:56.240905.sav')
    pricer = DoubleNetPricer(left_model, right_model)

    train_df = pd.read_csv('../prices_mc_20000_paths_5000.csv')
    test_df = pd.read_csv('../prices_mc_20k_test_paths_1000.csv')

    pricer.fit(train_df, test_df, which='left')

    df_test_left, df_test_right = pricer.predict_split(test_df)
    df_train_left, df_train_right = pricer.predict_split(train_df)

    print('Left RMSE: {:.2e}'.format(((df_train_left["monte_carlo_price"] - df_train_left["net_price"]) ** 2).mean() ** 0.5))
    print('Left Val RMSE: {:.2e}'.format(((df_test_left["monte_carlo_price"] - df_test_left["net_price"]) ** 2).mean() ** 0.5))

    print('Right RMSE: {:.2e}'.format(((df_train_right["monte_carlo_price"] - df_train_right["net_price"]) ** 2).mean() ** 0.5))
    print('Right Val RMSE: {:.2e}'.format(((df_test_right["monte_carlo_price"] - df_test_right["net_price"]) ** 2).mean() ** 0.5))
