import pandas as pd

from monte_carlo import create_dataset
from monte_carlo.path_generator import plot_paths
from positive_network.net_maker import get_trained_net_and_test_set as get_positive_net_and_test_set
from convex_network.net_maker import get_trained_net_and_test_set as get_convex_net_and_test_set
from settings import USE_DATA_FROM_FILE, DATASET_SIZE, USE_CONVEX_NETWORK, FIXED_AVG_TYPE, PLOT_SOME_PATHS
from utils.typing import OptionAvgType


def make_predicted_df(fixed_avg_type: OptionAvgType = None):
    if fixed_avg_type is None:
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


if __name__ == '__main__':
    if USE_DATA_FROM_FILE:
        df = pd.read_csv('monte_carlo_prices.csv')
    else:
        df = create_dataset(DATASET_SIZE)
        df.to_csv('options_prices.csv', index=False, float_format='%.4f')

    if PLOT_SOME_PATHS:
        plot_paths(df.iloc[:5, :])

    if USE_CONVEX_NETWORK:
        net, x_test, y_test = get_convex_net_and_test_set(df, test_size=0.1, fixed_avg_type=FIXED_AVG_TYPE)
    else:
        net, x_test, y_test = get_positive_net_and_test_set(df, test_size=0.1, fixed_avg_type=FIXED_AVG_TYPE)

    predict_price = net.predict(x_test).detach().numpy()

    df_test = make_predicted_df(fixed_avg_type=FIXED_AVG_TYPE)
    df_test.to_csv('convex_net_prices.csv' if USE_CONVEX_NETWORK else 'pos_net_prices.csv', index=False,
                   float_format='%.4f')

    print(f'MSE: {((y_test - predict_price) ** 2).mean()}')
