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
SPLIT_VOL_INTO_PARTS = 128


# TODO: Use different number of averaging steps for different ttm (not always 365)

class DoubleNetPricer:
    def __init__(self, left_net, right_net):
        self.left_net = left_net
        self.right_net = right_net
        self.lower_vol = 0.05
        self.upper_vol = 0.55

        if left_net is None:
            self.left_net = PositiveNet(IN_FEATURES_NUM)

        if right_net is None:
            self.right_net = SemiPositiveNet(IN_FEATURES_NUM)

    # Expect the following fields: 'spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility', 'avg_type'
    # Target field: 'price_strike_ratio'
    def fit(self, df: pd.DataFrame, val_df: pd.DataFrame, which = 'both'):
        left_df, right_df = self.split_for_left_and_right_net_df(df)
        left_val_df, right_val_df = self.split_for_left_and_right_net_df(val_df)

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
        left_df, right_df = self.split_for_left_and_right_net_df(df)
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
        left_df, right_df = self.predict_split(df, no_decode)

        return pd.concat([left_df, right_df], ignore_index=True)


    # First value: Time value increases w.r.t. moneyness
    # Second value: Time value decreases w.r.t. moneyness
    def split_for_left_and_right_net_df(self, df: pd.DataFrame) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        df = get_threshold(df)
        left_df = df[df['spot_strike_ratio'] <= df['threshold']]
        right_df = df[df['spot_strike_ratio'] > df['threshold']]

        del df['threshold']
        del left_df['threshold']
        del right_df['threshold']

        return left_df, right_df

    def predict_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        predicted_vol_df = df.apply(lambda row: self._calc_vol_for_row(row), axis=1, result_type='expand')
        predicted_vol_df.columns = ['predicted_vol', 'vol_left_ci', 'vol_right_ci']

        return pd.concat([df, predicted_vol_df], axis=1)

    def _calc_vol_for_row(self, row) -> typing.Tuple[float, float, float]:
        fineness = (self.upper_vol - self.lower_vol) / SPLIT_VOL_INTO_PARTS

        dict_for_pred = {
            'spot_strike_ratio': [row.spot_strike_ratio] * SPLIT_VOL_INTO_PARTS * 3,
            'ttm': [row.ttm] * SPLIT_VOL_INTO_PARTS * 3,
            'risk_free_rate': [row.risk_free_rate] * SPLIT_VOL_INTO_PARTS * 3,
            'volatility': np.tile(np.arange(0.05, 0.55, fineness), 3),
            'price_strike_ratio': [row.price_strike_ratio] * SPLIT_VOL_INTO_PARTS
                                  + [row.left_ci] * SPLIT_VOL_INTO_PARTS
                                  + [row.right_ci] * SPLIT_VOL_INTO_PARTS
        }

        pred = self.predict(pd.DataFrame(dict_for_pred), no_decode=True)

        pred['diff'] = (pred['net_price'] - pred['monte_carlo_price']) ** 2

        pred_vol = pred.iloc[0:SPLIT_VOL_INTO_PARTS].sort_values('diff', inplace=False).head(5)['volatility'].mean()
        left_vol_ci = pred.iloc[SPLIT_VOL_INTO_PARTS:2 * SPLIT_VOL_INTO_PARTS].sort_values('diff', inplace=False).head(5)['volatility'].mean()
        right_vol_ci = pred.iloc[2 * SPLIT_VOL_INTO_PARTS:].sort_values('diff', inplace=False).head(5)['volatility'].mean()

        return pred_vol, left_vol_ci, right_vol_ci


if __name__ == '__main__':
    TASK = 'vol'
    # TASK = 'price'

    left_model = joblib.load('models/left-2024-03-05 19:49:14.236234.sav')
    right_model = joblib.load('models/right-2024-03-05 16:10:21.678640.sav')

    pricer = DoubleNetPricer(left_model, right_model)

    real_df = pd.read_csv('../datasets/barchart ulsd/31_07_24_fixed.csv')
    train_df = pd.read_csv('../datasets/double_pricer/train_1_20000_paths_5000.csv')
    test_df = pd.read_csv('../datasets/double_pricer/test_20k_paths_1000.csv')

    if TASK == 'vol':
        left, right = pricer.split_for_left_and_right_net_df(real_df)
        print(len(left))
        print(len(right))

        vol_df = pricer.predict_vol(left)
        # print('Left RMSE: {:.2e}'.format(
        #     ((vol_df["volatility"] - vol_df["predicted_vol"]) ** 2).mean() ** 0.5))
        # print('Left Vol CI: {:.2e}'.format(
        #     ((vol_df["vol_left_ci"] - vol_df["vol_right_ci"]) ** 2).mean() ** 0.5))
        # vol_df[['spot_strike_ratio', 'volatility', 'predicted_vol', 'vol_left_ci', 'vol_right_ci']].to_csv(f'left-vol-{datetime.datetime.now()}.csv', index=False, float_format='%.4f')
        vol_df[['spot_strike_ratio', 'predicted_vol']].to_csv(f'left-vol-{datetime.datetime.now()}.csv', index=False, float_format='%.4f')

        vol_df = pricer.predict_vol(right)
        # print('Right RMSE: {:.2e}'.format(
        #     ((vol_df["volatility"] - vol_df["predicted_vol"]) ** 2).mean() ** 0.5))
        # print('Right Vol CI: {:.2e}'.format(
        #     ((vol_df["vol_left_ci"] - vol_df["vol_right_ci"]) ** 2).mean() ** 0.5))
        # vol_df[['spot_strike_ratio', 'volatility', 'predicted_vol', 'vol_left_ci', 'vol_right_ci']].to_csv(f'right-vol-{datetime.datetime.now()}.csv', index=False, float_format='%.4f')
        vol_df[['spot_strike_ratio', 'predicted_vol']].to_csv(f'right-vol-{datetime.datetime.now()}.csv', index=False, float_format='%.4f')

    if TASK == 'price':
        pricer.fit(train_df, test_df, which='left')

        df_test_left, df_test_right = pricer.predict_split(test_df)
        df_train_left, df_train_right = pricer.predict_split(train_df)

        print('Left RMSE: {:.2e}'.format(((df_train_left["monte_carlo_price"] - df_train_left["net_price"]) ** 2).mean() ** 0.5))
        print('Left Val RMSE: {:.2e}'.format(((df_test_left["monte_carlo_price"] - df_test_left["net_price"]) ** 2).mean() ** 0.5))

        print('Right RMSE: {:.2e}'.format(((df_train_right["monte_carlo_price"] - df_train_right["net_price"]) ** 2).mean() ** 0.5))
        print('Right Val RMSE: {:.2e}'.format(((df_test_right["monte_carlo_price"] - df_test_right["net_price"]) ** 2).mean() ** 0.5))
