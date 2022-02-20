import pandas as pd

from monte_carlo import create_dataset
from neural_network.net_maker import get_trained_net_and_test_set
from settings import USE_DATA_FROM_FILE, DATASET_SIZE


def make_predicted_df():
    df_predicted = pd.DataFrame(x_test, columns=['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility',
                                                 'numeric_avg_type'])
    df_predicted['avg_type'] = df_predicted.apply(
        lambda row: 'ARITHMETIC' if row.numeric_avg_type == 1 else 'GEOMETRIC', axis=1)
    df_predicted['monte_carlo_price'] = y_test
    df_predicted['net_price'] = predict_price

    del df_predicted['numeric_avg_type']

    return df_predicted


if __name__ == '__main__':
    if USE_DATA_FROM_FILE:
        df = pd.read_csv('options_prices.csv')
    else:
        df = create_dataset(DATASET_SIZE)
        df.to_csv('options_prices.csv', index=False, float_format='%.4f')

    net, x_test, y_test = get_trained_net_and_test_set(df, test_size=0.1)

    predict_price = net.predict(x_test).detach().numpy()

    df_test = make_predicted_df()
    df_test.to_csv('net_prices.csv', index=False, float_format='%.4f')

    print(f'MSE: {((y_test - predict_price) ** 2).mean()}')
