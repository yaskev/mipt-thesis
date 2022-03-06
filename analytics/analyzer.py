import os
from typing import Tuple, List

import numpy as np
import pandas as pd

from utils.plotting import create_chart, plot_different_metrics, plot_loss_comparison
from utils.typing import OptionAvgType, NetType
from positive_network.net_maker import get_trained_net_and_test_set as get_positive_net_and_test_set
from convex_network.net_maker import get_trained_net_and_test_set as get_convex_net_and_test_set


class Analyzer:
    def __init__(self, plot_metrics: bool, plot_cmp: bool, plot_loss: bool, dataset_sizes: List[int], print_steps: bool):
        self.plot_metrics = plot_metrics
        self.print_steps = print_steps
        self.plot_cmp = plot_cmp
        self.plot_loss = plot_loss
        self.dataset_sizes = dataset_sizes
        self.convex_all_mse, self.convex_arithm_mse, self.convex_geom_mse = [], [], []
        self.positive_all_mse, self.positive_arithm_mse, self.positive_geom_mse = [], [], []
        self.loss_cmp = {}

    def get_mse_and_loss(self, dataset: pd.DataFrame, dataset_size: int, net_type: NetType) -> Tuple[float,
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

        return sum(mse) / len(mse), \
               np.array(train_loss).mean(axis=0), \
               np.array(val_loss).mean(axis=0)

    def create_analytic_charts(self):
        df_all = pd.read_csv('dataset_analytics.csv')
        df_all['numeric_avg_type'] = df_all.apply(
            lambda row: 1 if row.avg_type == OptionAvgType.ARITHMETIC.value else 0, axis=1)
        df_arithmetic = df_all[df_all['avg_type'] == OptionAvgType.ARITHMETIC.value]
        df_geometric = df_all[df_all['avg_type'] == OptionAvgType.GEOMETRIC.value]

        self.calc_convex_metrics(self.dataset_sizes,
                                 df_all,
                                 df_arithmetic,
                                 df_geometric)

        self.calc_positive_metrics(self.dataset_sizes,
                                   df_all,
                                   df_arithmetic,
                                   df_geometric)

        if self.plot_metrics:
            plot_different_metrics({
                'Convex all': self.convex_all_mse,
                'Convex arithm': self.convex_arithm_mse,
                'Convex geom': self.convex_geom_mse,
                'Positive all': self.positive_all_mse,
                'Positive arithm': self.positive_arithm_mse,
                'Positive geom': self.positive_geom_mse
            },
                dataset_sizes=self.dataset_sizes,
                strict_path=os.path.join('charts', 'mse_comp.png'),
                metric_name='MSE'
            )

        if self.plot_cmp:
            plot_loss_comparison(self.loss_cmp, os.path.join('charts', 'comparison'))

    def calc_positive_metrics(self, dataset_sizes, df_all, df_arithmetic, df_geometric):
        for dataset_size in dataset_sizes:
            if dataset_size not in self.loss_cmp:
                self.loss_cmp[dataset_size] = {}

            # Arithmetic options
            if self.print_steps:
                print(f'Positive, arithm, df size: {dataset_size}')
            mean_mse, mean_train_loss, mean_val_loss = self.get_mse_and_loss(df_arithmetic, dataset_size, NetType.POSITIVE)
            self.positive_arithm_mse.append(mean_mse)
            if self.plot_loss:
                create_chart(mean_train_loss, mean_val_loss, '',
                             strict_path=os.path.join('charts', 'positive_net', 'arithm', f'{dataset_size}.png'))
            self.loss_cmp[dataset_size][f'Positive arithm {dataset_size}'] = mean_train_loss

            # Geometric options
            if self.print_steps:
                print(f'Positive, geom, df size: {dataset_size}')
            mean_mse, mean_train_loss, mean_val_loss = self.get_mse_and_loss(df_geometric, dataset_size, NetType.POSITIVE)
            self.positive_geom_mse.append(mean_mse)
            if self.plot_loss:
                create_chart(mean_train_loss, mean_val_loss, '',
                             strict_path=os.path.join('charts', 'positive_net', 'geom', f'{dataset_size}.png'))
            self.loss_cmp[dataset_size][f'Positive geom {dataset_size}'] = mean_train_loss

            # All options
            if self.print_steps:
                print(f'Positive, all, df size: {dataset_size}')
            mean_mse, mean_train_loss, mean_val_loss = self.get_mse_and_loss(df_all, dataset_size, NetType.POSITIVE)
            self.positive_all_mse.append(mean_mse)
            if self.plot_loss:
                create_chart(mean_train_loss, mean_val_loss, '',
                             strict_path=os.path.join('charts', 'positive_net', 'all', f'{dataset_size}.png'))
            self.loss_cmp[dataset_size][f'Positive all {dataset_size}'] = mean_train_loss

    def calc_convex_metrics(self, dataset_sizes, df_all, df_arithmetic, df_geometric):
        for dataset_size in dataset_sizes:
            if dataset_size not in self.loss_cmp:
                self.loss_cmp[dataset_size] = {}

            # Arithmetic options
            if self.print_steps:
                print(f'Convex, arithm, df size: {dataset_size}')
            mean_mse, mean_train_loss, mean_val_loss = self.get_mse_and_loss(df_arithmetic, dataset_size, NetType.CONVEX)
            self.convex_all_mse.append(mean_mse)
            if self.plot_loss:
                create_chart(mean_train_loss, mean_val_loss, '',
                             strict_path=os.path.join('charts', 'convex_net', 'arithm', f'{dataset_size}.png'))
            self.loss_cmp[dataset_size][f'Convex arithm {dataset_size}'] = mean_train_loss

            # Geometric options
            if self.print_steps:
                print(f'Convex, geom, df size: {dataset_size}')
            mean_mse, mean_train_loss, mean_val_loss = self.get_mse_and_loss(df_geometric, dataset_size, NetType.CONVEX)
            self.convex_geom_mse.append(mean_mse)
            if self.plot_loss:
                create_chart(mean_train_loss, mean_val_loss, '',
                             strict_path=os.path.join('charts', 'convex_net', 'geom', f'{dataset_size}.png'))
            self.loss_cmp[dataset_size][f'Convex geom {dataset_size}'] = mean_train_loss

            # All options
            if self.print_steps:
                print(f'Convex, all, df size: {dataset_size}')
            mean_mse, mean_train_loss, mean_val_loss = self.get_mse_and_loss(df_all, dataset_size, NetType.CONVEX)
            self.convex_arithm_mse.append(mean_mse)
            if self.plot_loss:
                create_chart(mean_train_loss, mean_val_loss, '',
                             strict_path=os.path.join('charts', 'convex_net', 'all', f'{dataset_size}.png'))
            self.loss_cmp[dataset_size][f'Convex all {dataset_size}'] = mean_train_loss


if __name__ == '__main__':
    Analyzer(plot_metrics=False,
             plot_cmp=True,
             plot_loss=False,
             dataset_sizes=[30, 100, 500],
             print_steps=True).create_analytic_charts()
