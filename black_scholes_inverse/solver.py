import os

import joblib
import numpy as np
import pandas as pd

from preprocessing.intrinsic_val import encode
from settings import POSITIVE_MODEL_PATH
from utils.typing import OptionAvgType

PATH_PREFIX = '..'
VOL_TOL_EXITS = 0
PRICE_TOL_EXITS = 0
ITER_NUM = 0


class Solver:
    def __init__(self, model_path: str, vol_tol = 1e-3, price_tol = 1e-4, lower_vol = 0.0, upper_vol = 0.8, use_bin_search = True):
        self.model = joblib.load(os.path.join(PATH_PREFIX, model_path))
        self.vol_tol = vol_tol
        self.price_tol = price_tol
        self.lower_vol = lower_vol
        self.upper_vol = upper_vol
        self.use_bin_search = use_bin_search

    def predict_vol(self, df: pd.DataFrame, apply_preproc = False) -> pd.DataFrame:
        df['numeric_avg_type'] = df.apply(lambda row: 1 if row.avg_type == OptionAvgType.ARITHMETIC.value else 0,
                                          axis=1)
        if apply_preproc:
            df = encode(df)

        df['predicted_vol'] = df.apply(lambda row: self._calc_vol_for_row(row, False), axis=1)

        return df

    def _calc_vol_for_row(self, row, apply_preproc) -> float:
        global PRICE_TOL_EXITS, VOL_TOL_EXITS, ITER_NUM
        if self.use_bin_search:
            left = self.lower_vol
            right = self.upper_vol
            mid = (left + right) / 2

            np_row = row[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility', 'numeric_avg_type']].astype(
                np.float32).to_numpy()

            while abs(right - left) > self.vol_tol:
                ITER_NUM += 1
                mid = (left + right) / 2
                np_row[3] = mid
                net_price = self.model.predict(np_row).detach().numpy()

                true_price = row.price_strike_ratio

                if apply_preproc:
                    net_price = np.exp(net_price)
                    true_price = np.exp(true_price)

                if abs(net_price - true_price) < self.price_tol:
                    PRICE_TOL_EXITS += 1
                    break
                if net_price > row.price_strike_ratio:
                    right = mid
                    continue
                left = mid
            if abs(right - left) <= self.vol_tol and abs(net_price - row.price_strike_ratio) >= self.price_tol:
                VOL_TOL_EXITS += 1

            return mid
        else:
            curr_vol = self.lower_vol
            np_row = row[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility', 'numeric_avg_type']].astype(
                np.float32).to_numpy()

            vols_giving_good_tol = []

            while curr_vol < self.upper_vol:
                ITER_NUM += 1
                np_row[3] = curr_vol
                net_price = self.model.predict(np_row).detach().numpy()

                true_price = row.price_strike_ratio

                if apply_preproc:
                    net_price = np.exp(net_price)
                    true_price = np.exp(true_price)

                if abs(net_price - true_price) < self.price_tol:
                    if not vols_giving_good_tol:
                        PRICE_TOL_EXITS += 1
                    vols_giving_good_tol.append(curr_vol)

                curr_vol += 0.01

            if curr_vol >= self.upper_vol and not vols_giving_good_tol:
                VOL_TOL_EXITS += 1
                return (self.lower_vol + self.upper_vol) / 2

            return sum(vols_giving_good_tol) / len(vols_giving_good_tol)


if __name__ == '__main__':
    # Return first vol which satisfies price constraint
    #
    # FF net without intr values
    # price tol 5e-4: MSE 0.117 (step 0.001)
    # price tol 0: MSE 0.144
    #
    # Pos net, without intr values (no preprocessing)
    # price tol 1e-2: MSE 0.23
    # closer to predicting mean => better quality
    #
    # FF net with intr value subtraction and log
    # price tol (log): 0.9, MSE: 0.136
    #
    # Trained pos net using accurate data with subtracted intrinsic value, without log
    # price tol: 2e-2, MSE: 0.259
    #
    # Trained (and overfitted) net using accurate data from part 2 and preprocessing with intrinsic value subtraction and log
    # price tol: 1.6, MSE: 0.16
    #
    #
    #
    # Return mean of the vols which satisfy price constraint
    #
    # FF net without intr values
    # price tol: 2.6e-3, MSE: 9.15e-2
    #
    # Pos net, without intr values (no preprocessing)
    # price tol: 1.6e-2, MSE: 0.126
    #
    # FF net with intr value subtraction and log
    # price tol (log): 3.0, MSE 9.62e-2
    #
    # Trained pos net using accurate data with subtracted intrinsic value, without log
    # price tol: any, MSE: 1.44e-1 (mean vol)
    #
    # Trained (and overfitted) net using accurate data from part 2 and preprocessing with intrinsic value subtraction and log
    # price tol: 1.5 (log), MSE 9.47e-2

    price_tols = [2.6e-3]

    for tol in price_tols:
        print(tol)

        VOL_TOL_EXITS = 0
        PRICE_TOL_EXITS = 0
        ITER_NUM = 0

        df = pd.read_csv(os.path.join(PATH_PREFIX, 'prices_mc_20000_paths_5000.csv'))
        df = df[df['avg_type'] == OptionAvgType.ARITHMETIC.value]
        solver = Solver(POSITIVE_MODEL_PATH, vol_tol=1e-5, price_tol=tol, lower_vol=0.05, upper_vol=0.55, use_bin_search=False)
        df = solver.predict_vol(df, apply_preproc=False)

        print('MSE: {:.2e}'.format(((df["volatility"] - df["predicted_vol"]) ** 2).mean() ** 0.5))
        df[['volatility', 'predicted_vol']].to_csv('predict_vol.csv', index=False, float_format='%.4f')

        print(f'vol_tol_exit: {VOL_TOL_EXITS}')
        print(f'price_tol_exit: {PRICE_TOL_EXITS}')
        print(f'avg iter_num: {ITER_NUM / (VOL_TOL_EXITS + PRICE_TOL_EXITS)}')
