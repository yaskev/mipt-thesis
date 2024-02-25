import time

import numpy as np
import pandas as pd
import joblib
import torch

import preprocessing.intrinsic_val
from monte_carlo import create_dataset
from monte_carlo.greeks import get_greeks
from monte_carlo.path_generator import plot_paths
from positive_network.net_maker import get_trained_net_and_test_set as get_positive_net_and_test_set
from convex_network.net_maker import get_trained_net_and_test_set as get_convex_net_and_test_set
from sigma_network.net_maker import get_trained_net_and_test_set as get_sigma_positive_net_and_test_set
from semipositive_network.net_maker import get_trained_net_and_test_set as get_semipositive_net_and_test_set
from settings import USE_DATA_FROM_FILE, DATASET_SIZE, FIXED_AVG_TYPE, PLOT_SOME_PATHS, CALC_GREEKS, \
    SAVE_TRAINED_NET, NETWORK_TYPE, WITH_CI_STATS
from sigma_network.network import SigmaNetType
from utils.mapping import idx_to_col_name
from utils.typing import OptionAvgType, ComplexNetworkType


def make_predicted_df(x_test: list, y_test: list, predict_price: np.ndarray, fixed_avg_type: OptionAvgType = None):
    if fixed_avg_type is None:
        if WITH_CI_STATS:
            df_predicted = pd.DataFrame(x_test, columns=['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility',
                                                         'numeric_avg_type', 'left_ci', 'right_ci'])
        else:
            df_predicted = pd.DataFrame(x_test, columns=['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility',
                                                         'numeric_avg_type'])
        df_predicted['avg_type'] = df_predicted.apply(
            lambda row: 'ARITHMETIC' if row.numeric_avg_type == 1 else 'GEOMETRIC', axis=1)
        del df_predicted['numeric_avg_type']
    else:
        df_predicted = pd.DataFrame(x_test, columns=['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility'])

    df_predicted['monte_carlo_price'] = y_test
    df_predicted['net_price'] = predict_price

    return df_predicted


def make_predicted_vol_df(x_test: list, y_test: list, predict_vol: np.ndarray, fixed_avg_type: OptionAvgType = None):
    if fixed_avg_type is None:
        df_predicted = pd.DataFrame(x_test, columns=['spot_strike_ratio', 'ttm', 'risk_free_rate', 'price_strike_ratio'])
        # df_predicted = pd.DataFrame(x_test, columns=['spot_strike_ratio', 'ttm', 'risk_free_rate', 'price_strike_ratio',
        #                                              'numeric_avg_type'])
        # df_predicted['avg_type'] = df_predicted.apply(
        #     lambda row: 'ARITHMETIC' if row.numeric_avg_type == 1 else 'GEOMETRIC', axis=1)
        # del df_predicted['numeric_avg_type']
    else:
        df_predicted = pd.DataFrame(x_test, columns=['spot_strike_ratio', 'ttm', 'risk_free_rate', 'price_strike_ratio'])

    df_predicted['monte_carlo_vol'] = y_test
    df_predicted['net_vol'] = predict_vol

    return df_predicted


def main():
    if USE_DATA_FROM_FILE:
        # df = pd.read_csv('datasets/train/prices_mc_with_ci.csv')
        # df = pd.read_csv('datasets/train/prices_mc_with_ci.csv')
        # test_df = pd.read_csv('datasets/test/prices_mc_with_ci.csv')

        # df = pd.read_csv('prices_mc_20000_paths_5000.csv')
        #
        # df = df[df['avg_type'] == OptionAvgType.ARITHMETIC.value]

        df = pd.read_csv('prices_mc_mn_up_to_0.95.csv')

        # df = pd.read_csv('prices_mc_mn_from_1.csv')
        df = df[df['avg_type'] == OptionAvgType.ARITHMETIC.value]
        # test_df = pd.read_csv('prices_mc_20k_test_paths_1000.csv')
        # test_df = test_df[test_df['avg_type'] == OptionAvgType.ARITHMETIC.value]

        df = preprocessing.intrinsic_val.encode(df)
        test_df = df
        # test_df = preprocessing.intrinsic_val.encode(test_df)
        # test_df.to_csv('subtr.csv', index=False, float_format='%.4f')
        #
        # df.to_csv('tmp.csv', index=False, float_format='%.4f')
        # df = preprocessing.intrinsic_val.add_subtracted_intrinsic_value(
        #     df
        # )
        # df['spot_strike_ratio'] = df['subtracted_int_val']
        # df = df.drop(columns='subtracted_int_val')
        # df.loc[df['spot_strike_ratio'] == 0, 'spot_strike_ratio'] = 0.0001
        # df['spot_strike_ratio'] = np.log(df['spot_strike_ratio']) + 10
        # df.to_csv('with_sub_int_val.csv', index=False, float_format='%.4f')
        # return
        # test_df = pd.read_csv('datasets/article/test.csv')
        # df = pd.read_csv('datasets/sigma_with_shift/prices_mc_with_shift.csv')
        # test_df = pd.read_csv('datasets/sigma_with_shift/prices_mc_with_shift_test.csv')
        # df = pd.read_csv('datasets/train64/prices_mc_with_ci_train_greeks_50.csv')
        # test_df = pd.read_csv('datasets/test/prices_mc_with_ci.csv')
        # test_df = df
        # df = pd.read_csv('datasets/test/prices_mc_with_ci.csv')
        # df = pd.read_csv('cme_data.csv')
        # df = pd.read_csv('datasets/mc_fixed/prices_mc_fixed_train.csv')
        # test_df = pd.read_csv('datasets/mc_fixed/prices_mc_fixed_test.csv')
    else:
        df = create_dataset(DATASET_SIZE)
        # test_df = df
        df.to_csv('prices_mc_mn_from_1.csv', index=False, float_format='%.4f')
        return

    if PLOT_SOME_PATHS:
        plot_paths(df.iloc[:5, :])

    if NETWORK_TYPE == ComplexNetworkType.CONVEX_NETWORK:
        net, x_test, y_test, x_val, y_val = get_convex_net_and_test_set(df, test_df, test_size=1, fixed_avg_type=FIXED_AVG_TYPE)
    elif NETWORK_TYPE == ComplexNetworkType.POSITIVE_NETWORK:
        net, x_test, y_test, x_val, y_val = get_positive_net_and_test_set(df, test_df, test_size=1, fixed_avg_type=FIXED_AVG_TYPE)
    elif NETWORK_TYPE == ComplexNetworkType.SEMIPOSITIVE_NETWORK:
        net, x_test, y_test, x_val, y_val = get_semipositive_net_and_test_set(df, test_df, test_size=0.1,
                                                                          fixed_avg_type=FIXED_AVG_TYPE)
    else:
        net, x_test, y_test = get_sigma_positive_net_and_test_set(df, test_df, test_size=0.1,
                                                                  fixed_avg_type=FIXED_AVG_TYPE,
                                                                  net_type=SigmaNetType.SMALL_FFNN)
        predict_vol = net.predict(x_test).detach().numpy()
        df_test = make_predicted_vol_df(x_test, y_test, predict_vol, fixed_avg_type=FIXED_AVG_TYPE)
        df_test.to_csv('pos_net_sigma_inf.csv', index=False, float_format='%.4f')
        print('MSE: {:.2e}'.format(((df_test["monte_carlo_vol"] - df_test["net_vol"]) ** 2).mean() ** 0.5))

        if SAVE_TRAINED_NET:
            joblib.dump(net, 'trained_sigma_fixed_mc_small_5000.sav')

        return

    t = []
    predict_price = None
    predict_val_price = None
    for _ in range(1):
        start = time.process_time()
        if WITH_CI_STATS:
            predict_price = net.predict(x_test[:,:-2]).detach().numpy()
            predict_val_price = net.predict(x_val[:,:-2]).detach().numpy()
        else:
            predict_price = net.predict(x_test).detach().numpy()
            predict_val_price = net.predict(x_val).detach().numpy()
        t.append(time.process_time() - start)
    print(sum(t) / len(t))

    df_test = make_predicted_df(x_test, y_test, predict_price, fixed_avg_type=FIXED_AVG_TYPE)
    df_val = make_predicted_df(x_val, y_val, predict_val_price, fixed_avg_type=FIXED_AVG_TYPE)

    df_test = preprocessing.intrinsic_val.decode(df_test)
    df_val = preprocessing.intrinsic_val.decode(df_val)

    df_test.to_csv(f'{get_file_name()}.csv', index=False, float_format='%.4f')
    print('RMSE: {:.2e}'.format(((df_test["monte_carlo_price"] - df_test["net_price"]) ** 2).mean() ** 0.5))
    print('Val RMSE: {:.2e}'.format(((df_val["monte_carlo_price"] - df_val["net_price"]) ** 2).mean() ** 0.5))

    if WITH_CI_STATS:
        in_ci = (df_test['net_price'] < df_test['right_ci']) * (df_test['net_price'] > df_test['left_ci'])
        print(in_ci.mean())

    if SAVE_TRAINED_NET:
        joblib.dump(net, 'neg_net_1+.sav')

    if CALC_GREEKS:
        t = []
        greeks = None
        for _ in range(1):
            start = time.process_time()
            greeks = net.get_greeks(x_test)
            t.append(time.process_time() - start)
        print(sum(t) / len(t))

        # df_from_numpy = pd.DataFrame(x_test[:,:-2], columns=[*list(idx_to_col_name.values()), 'numeric_avg_type'])
        # df_from_numpy['avg_type'] = df_from_numpy.apply(
        #     lambda row: 'ARITHMETIC' if row.numeric_avg_type == 1 else 'GEOMETRIC', axis=1)
        # greeks_mc = get_greeks(df_from_numpy)
        # full_df = pd.concat([df_test, greeks, greeks_mc], axis=1)
        # # full_df.to_csv('convex_net_greeks.csv' if USE_CONVEX_NETWORK else 'pos_net_greeks.csv', index=False,
        # #                float_format='%.4f')
        # full_df.to_csv('greeks_test_con_50.csv', index=False, float_format='%.4f')

def get_file_name() -> str:
    if NETWORK_TYPE == ComplexNetworkType.CONVEX_NETWORK:
        return 'convex_net_prices'
    if NETWORK_TYPE == ComplexNetworkType.POSITIVE_NETWORK:
        return 'pos_net_prices'
    if NETWORK_TYPE == ComplexNetworkType.SEMIPOSITIVE_NETWORK:
        return 'semipos_net_prices'

    return 'unknown_net_prices'

if __name__ == '__main__':
    main()
