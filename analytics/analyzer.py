import os
from typing import Tuple, List

import numpy as np
import pandas as pd

from utils.plotting import create_chart, plot_different_metrics
from utils.typing import OptionAvgType, NetType
from positive_network.net_maker import get_trained_net_and_test_set as get_positive_net_and_test_set
from convex_network.net_maker import get_trained_net_and_test_set as get_convex_net_and_test_set


def get_mse_and_loss(dataset: pd.DataFrame, dataset_size: int, net_type: NetType) -> Tuple[float,
                                                                                           List[float],
                                                                                           List[float]]:
    mse = []
    train_loss, val_loss = [], []
    if net_type == NetType.CONVEX:
        for i in range(5 if dataset_size <= 1000 else 1):
            net, x_test, y_test, curr_t_loss, curr_v_loss = get_convex_net_and_test_set(
                dataset[i * dataset_size: (i + 1) * dataset_size],
                test_size=0.1, fixed_avg_type=None, analytics_mode=True)
            predict_price = net.predict(x_test).detach().numpy()
            mse.append(((y_test - predict_price) ** 2).mean())
            train_loss.append(curr_t_loss)
            val_loss.append(curr_v_loss)
    else:
        for i in range(5 if dataset_size <= 1000 else 1):
            net, x_test, y_test, curr_t_loss, curr_v_loss = get_positive_net_and_test_set(
                dataset[i * dataset_size: (i + 1) * dataset_size],
                test_size=0.1, fixed_avg_type=None, analytics_mode=True)
            predict_price = net.predict(x_test).detach().numpy()
            mse.append(((y_test - predict_price) ** 2).mean())
            train_loss.append(curr_t_loss)
            val_loss.append(curr_v_loss)

    return sum(mse) / len(mse), np.array(train_loss).mean(axis=0), np.array(val_loss).mean(axis=0)


def create_analytic_charts():
    df_all = pd.read_csv('dataset_analytics.csv')
    df_all['numeric_avg_type'] = df_all.apply(lambda row: 1 if row.avg_type == OptionAvgType.ARITHMETIC.value else 0, axis=1)
    df_arithmetic = df_all[df_all['avg_type'] == OptionAvgType.ARITHMETIC.value]
    df_geometric = df_all[df_all['avg_type'] == OptionAvgType.GEOMETRIC.value]
    dataset_sizes = [30, 100, 500, 1000, 10000]

    # Convex net
    convex_arithm_mse = []
    convex_geom_mse = []
    convex_all_mse = []
    for dataset_size in dataset_sizes:
        print(f'Convex, arithm, df size: {dataset_size}')
        mean_mse, mean_train_loss, mean_val_loss = get_mse_and_loss(df_arithmetic, dataset_size, NetType.CONVEX)
        convex_all_mse.append(mean_mse)
        create_chart(mean_train_loss, mean_val_loss, '',
                     strict_path=os.path.join('charts', 'convex_net', 'arithm', f'{dataset_size}.png'))

        print(f'Convex, geom, df size: {dataset_size}')
        mean_mse, mean_train_loss, mean_val_loss = get_mse_and_loss(df_geometric, dataset_size, NetType.CONVEX)
        convex_geom_mse.append(mean_mse)
        create_chart(mean_train_loss, mean_val_loss, '',
                     strict_path=os.path.join('charts', 'convex_net', 'geom', f'{dataset_size}.png'))

        print(f'Convex, all, df size: {dataset_size}')
        mean_mse, mean_train_loss, mean_val_loss = get_mse_and_loss(df_all, dataset_size, NetType.CONVEX)
        convex_arithm_mse.append(mean_mse)
        create_chart(mean_train_loss, mean_val_loss, '',
                     strict_path=os.path.join('charts', 'convex_net', 'all', f'{dataset_size}.png'))

    # Positive net
    positive_arithm_mse = []
    positive_geom_mse = []
    positive_all_mse = []
    for dataset_size in dataset_sizes:
        print(f'Positive, arithm, df size: {dataset_size}')
        mean_mse, mean_train_loss, mean_val_loss = get_mse_and_loss(df_arithmetic, dataset_size, NetType.POSITIVE)
        positive_arithm_mse.append(mean_mse)
        create_chart(mean_train_loss, mean_val_loss, '',
                     strict_path=os.path.join('charts', 'positive_net', 'arithm', f'{dataset_size}.png'))

        print(f'Positive, geom, df size: {dataset_size}')
        mean_mse, mean_train_loss, mean_val_loss = get_mse_and_loss(df_geometric, dataset_size, NetType.POSITIVE)
        positive_geom_mse.append(mean_mse)
        create_chart(mean_train_loss, mean_val_loss, '',
                     strict_path=os.path.join('charts', 'positive_net', 'geom', f'{dataset_size}.png'))

        print(f'Positive, all, df size: {dataset_size}')
        mean_mse, mean_train_loss, mean_val_loss = get_mse_and_loss(df_all, dataset_size, NetType.POSITIVE)
        positive_all_mse.append(mean_mse)
        create_chart(mean_train_loss, mean_val_loss, '',
                     strict_path=os.path.join('charts', 'positive_net', 'all', f'{dataset_size}.png'))

    plot_different_metrics({
        'Convex all': convex_all_mse,
        'Convex arithm': convex_arithm_mse,
        'Convex geom': convex_geom_mse,
        'Positive all': positive_all_mse,
        'Positive arithm': positive_arithm_mse,
        'Positive geom': positive_geom_mse
    },
        dataset_sizes=dataset_sizes,
        strict_path=os.path.join('charts', 'mse_comp.png'),
        metric_name='MSE'
    )


if __name__ == '__main__':
    create_analytic_charts()
