import datetime
import os
import typing

import joblib
import numpy as np
import pandas as pd
import torch

from double_net_pricer.intrinsic import get_threshold, encode_right, decode_right, right_special_encode_decode, \
    log_price_decode, log_price_encode
from main import make_predicted_df
from positive_network.network import OptionsNet as PositiveNet
from semipositive_network.network import OptionsNet as SemiPositiveNet
from convex_network.network import ConvexNet
from utils.plotting import create_chart
from utils.typing import OptionAvgType

IN_FEATURES_NUM = 4
SPLIT_VOL_INTO_PARTS = 256


# TODO: Use different number of averaging steps for different ttm (not always 365)

class DoubleNetPricer:
    def __init__(self, left_net, right_net, convex=False):
        self.left_net = left_net
        self.right_net = right_net
        self.lower_vol = 0.05
        self.upper_vol = 0.7
        self.convex = convex

        if left_net is None:
            if convex:
                self.left_net = ConvexNet(IN_FEATURES_NUM, torch.tensor([0]))
            else:
                self.left_net = PositiveNet(IN_FEATURES_NUM)

        if right_net is None:
            if convex:
                self.right_net = ConvexNet(IN_FEATURES_NUM, torch.tensor([0]))
            else:
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
            create_chart(left_train_loss, left_val_loss, 'convex' if self.convex else '', 'left-')
            joblib.dump(self.left_net, os.path.join(('convex/' if self.convex else '') + 'models', f'left-{datetime.datetime.now()}.sav'))
        if which != 'left':
            right_train_loss, right_val_loss = self.right_net.fit(right_df_values, right_df_target, right_val_df_values, right_val_df_target)
            create_chart(right_train_loss, right_val_loss, 'convex' if self.convex else '', 'right-')
            joblib.dump(self.right_net, os.path.join(('convex/' if self.convex else '') + 'models', f'right-{datetime.datetime.now()}.sav'))


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
            'volatility': np.tile(np.arange(self.lower_vol, self.upper_vol, fineness), 3),
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
    # TASK = 'vol'
    TASK = 'price'
    # TASK = 'real'
    # TASK = 'convex_price'
    # TASK = 'none'

    left_model = joblib.load('convex/models/left-2024-03-08 00:22:27.197058.sav')
    right_model = joblib.load('convex/models/right-2024-03-08 00:43:26.658967.sav')
    convex = True

    pricer = DoubleNetPricer(None, None, convex=convex)

    # real_df = pd.read_csv('../datasets/barchart ulsd/31_07_24_fixed.csv')
    train_df = pd.read_csv('../datasets/double_pricer/train_4_ttm_003_to_053_20000_paths_5000.csv')

    second_train_df = pd.read_csv('../datasets/double_pricer/train_5_ttm_003_to_053_20000_paths_5000.csv')
    # extra_df = pd.read_csv('../datasets/double_pricer/train_1_20000_paths_5000.csv')
    # extra_df = extra_df[extra_df['ttm'] < 1]
    train_df = pd.concat([train_df, second_train_df])
    test_df = pd.read_csv('../datasets/double_pricer/test_2_ttm_003_to_053_20000_paths_1000.csv')

    if TASK == 'real':
        # for date in ['28_03_24', '30_04_24', '31_05_24', '28_06_24', '31_07_24']:
        for date in ['28_06_24_v2', '30_04_24_v2', '28_03_24_v2', '31_05_24_v2', '31_07_24_v2']:  #['28_03_24', '30_04_24', '31_05_24', '28_06_24', '31_07_24']:
            real_df = pd.read_csv(f'../datasets/barchart ulsd/{date}_fixed.csv')
            # real_df = pd.read_csv('surface/cme_data.csv')
            # real_df = real_df[real_df['ttm'] == 0.458]
            # real_df.sort_values('spot_strike_ratio', inplace=True)
            left, right = pricer.split_for_left_and_right_net_df(real_df)

            vol_df = pricer.predict_vol(left)
            # print('Left RMSE: {:.2e}'.format(
            #     ((vol_df["volatility"] - vol_df["predicted_vol"]) ** 2).mean() ** 0.5))
            # print('Left Vol CI: {:.2e}'.format(
            #     ((vol_df["vol_left_ci"] - vol_df["vol_right_ci"]) ** 2).mean() ** 0.5))
            # vol_df[['spot_strike_ratio', 'volatility', 'predicted_vol', 'vol_left_ci', 'vol_right_ci']].to_csv(f'left-vol-{datetime.datetime.now()}.csv', index=False, float_format='%.4f')
            # vol_df[['spot_strike_ratio', 'predicted_vol']].to_csv(f'vol_res/left-vol-{datetime.datetime.now()}.csv', index=False, float_format='%.4f')

            vol_df_right = pricer.predict_vol(right)
            # print('Right RMSE: {:.2e}'.format(
            #     ((vol_df["volatility"] - vol_df["predicted_vol"]) ** 2).mean() ** 0.5))
            # print('Right Vol CI: {:.2e}'.format(
            #     ((vol_df["vol_left_ci"] - vol_df["vol_right_ci"]) ** 2).mean() ** 0.5))
            # vol_df[['spot_strike_ratio', 'volatility', 'predicted_vol', 'vol_left_ci', 'vol_right_ci']].to_csv(f'right-vol-{datetime.datetime.now()}.csv', index=False, float_format='%.4f')
            # vol_df[['spot_strike_ratio', 'predicted_vol']].to_csv(f'vol_res/right-vol-{datetime.datetime.now()}.csv', index=False, float_format='%.4f')

            whole = pd.concat([vol_df, vol_df_right], ignore_index=True)
            if convex:
                whole.to_csv(f'surface/convex_{date}.csv', index=False, float_format='%.4f')
            else:
                whole.to_csv(f'surface/{date}.csv', index=False, float_format='%.4f')
            # whole.to_csv(f'surface/cme_data_v4.csv', index=False, float_format='%.4f')

    if TASK == 'vol':
        left, right = pricer.split_for_left_and_right_net_df(train_df[:1000])

        vol_df = pricer.predict_vol(left)
        print('Left RMSE: {:.2e}'.format(
            ((vol_df["volatility"] - vol_df["predicted_vol"]) ** 2).mean() ** 0.5))
        print('Left Vol CI: {:.2e}'.format(
            ((vol_df["vol_left_ci"] - vol_df["vol_right_ci"]) ** 2).mean() ** 0.5))

        vol_df = pricer.predict_vol(right)
        print('Right RMSE: {:.2e}'.format(
            ((vol_df["volatility"] - vol_df["predicted_vol"]) ** 2).mean() ** 0.5))
        print('Right Vol CI: {:.2e}'.format(
            ((vol_df["vol_left_ci"] - vol_df["vol_right_ci"]) ** 2).mean() ** 0.5))

    if TASK == 'price':
        pricer.fit(train_df, test_df)

        df_test_left, df_test_right = pricer.predict_split(test_df)
        df_train_left, df_train_right = pricer.predict_split(train_df)

        print('Left RMSE: {:.2e}'.format(((df_train_left["monte_carlo_price"] - df_train_left["net_price"]) ** 2).mean() ** 0.5))
        print('Left Val RMSE: {:.2e}'.format(((df_test_left["monte_carlo_price"] - df_test_left["net_price"]) ** 2).mean() ** 0.5))

        print('Right RMSE: {:.2e}'.format(((df_train_right["monte_carlo_price"] - df_train_right["net_price"]) ** 2).mean() ** 0.5))
        print('Right Val RMSE: {:.2e}'.format(((df_test_right["monte_carlo_price"] - df_test_right["net_price"]) ** 2).mean() ** 0.5))

    if TASK == 'convex_price':
        # left_model = joblib.load('cmodels/left-2024-03-05 19:49:14.236234.sav')
        # right_model = joblib.load('models/right-2024-03-05 16:10:21.678640.sav')

        pricer = DoubleNetPricer(None, None, convex=True)

        train_df = pd.read_csv('../datasets/double_pricer/train_1_20000_paths_5000.csv')
        test_df = pd.read_csv('../datasets/double_pricer/test_1_20k_paths_1000.csv')

        pricer.fit(train_df, test_df)

        df_test_left, df_test_right = pricer.predict_split(test_df)
        df_train_left, df_train_right = pricer.predict_split(train_df)

        print('Left RMSE: {:.2e}'.format(
            ((df_train_left["monte_carlo_price"] - df_train_left["net_price"]) ** 2).mean() ** 0.5))
        print('Left Val RMSE: {:.2e}'.format(
            ((df_test_left["monte_carlo_price"] - df_test_left["net_price"]) ** 2).mean() ** 0.5))

        print('Right RMSE: {:.2e}'.format(
            ((df_train_right["monte_carlo_price"] - df_train_right["net_price"]) ** 2).mean() ** 0.5))
        print(
            'Right Val RMSE: {:.2e}'.format(((df_test_right["monte_carlo_price"] - df_test_right["net_price"]) ** 2).mean() ** 0.5))
