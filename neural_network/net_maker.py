import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from neural_network.network import OptionsNet


def get_trained_net_and_test_set(df: pd.DataFrame, test_size: float):
    df['numeric_avg_type'] = df.apply(lambda row: 1 if row.avg_type == 'ARITHMETIC' else 0, axis=1)
    df_values = df[['spot_strike_ratio', 'ttm', 'risk_free_rate', 'volatility', 'numeric_avg_type']].astype(
        np.float32).to_numpy()
    df_target = df['price_strike_ratio'].astype(np.float32).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(df_values, df_target, test_size=test_size)
    net = OptionsNet(x_train.shape[1])
    net.fit(x_train, y_train)

    return net, x_test, y_test
