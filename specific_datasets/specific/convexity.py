import matplotlib.pyplot as plt
import pandas as pd

from utils.mapping import *


def create_convexity_plots():
    for key, value in idx_to_param.items():
        df_pos = pd.read_csv(f'../positive/fixed_{value}.csv')
        df_con = pd.read_csv(f'../convex/fixed_{value}.csv')

        df_pos.sort_values(idx_to_col_name[key], inplace=True)
        df_con.sort_values(idx_to_col_name[key], inplace=True)

        plt.figure(figsize=(12, 8))
        plt.grid()
        plt.plot(df_pos[idx_to_col_name[key]], df_pos['net_price'], linewidth=2, label='Positive net')
        plt.plot(df_con[idx_to_col_name[key]], df_con['net_price'], linewidth=2, label='Convex net')
        plt.title(f'Price by {idx_to_chart_name[key].lower()}', fontsize=20)
        plt.xlabel(f'{idx_to_chart_name[key]}', fontsize=18)
        plt.ylabel('Price/strike ratio', fontsize=18)
        plt.legend(fontsize=18)
        plt.savefig(f'../charts/fixed_{value}.jpg')


if __name__ == '__main__':
    create_convexity_plots()
