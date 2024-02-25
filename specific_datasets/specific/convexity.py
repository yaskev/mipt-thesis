import matplotlib.pyplot as plt
import pandas as pd

from utils.mapping import *


def create_convexity_plots(idxs):
    for key, value in idx_to_param.items():
        # if value != 'ttm':
        #     continue

        for idx in idxs:
            # df_semipos = pd.read_csv(f'../semipositive/fixed_{value}_with_sub_{idx}.csv')
            df_pos = pd.read_csv(f'../positive/fixed_{value}_with_sub_{idx}.csv')
            # df_con = pd.read_csv(f'../convex/fixed_{value}_5.csv')
            df_mc = pd.read_csv(f'../fixed_{value}_dataset_with_sub_{idx}(0_5-0_95).csv')

            # df_semipos.sort_values(idx_to_col_name[key], inplace=True)
            df_pos.sort_values(idx_to_col_name[key], inplace=True)
            # df_con.sort_values(idx_to_col_name[key], inplace=True)
            df_mc.sort_values(idx_to_col_name[key], inplace=True)

            plt.figure(figsize=(12, 8))
            plt.grid()
            plt.plot(df_mc[idx_to_col_name[key]], df_mc['price_strike_ratio'], linewidth=2, label='MC')
            # plt.plot(df_semipos[idx_to_col_name[key]], df_semipos['net_price'], linewidth=2, label='Semipositive net')
            plt.plot(df_pos[idx_to_col_name[key]], df_pos['net_price'], linewidth=2, label='Positive net')
            # plt.plot(df_con[idx_to_col_name[key]], df_con['net_price'], linewidth=2, label='Convex net')
            plt.title(f'Price by {idx_to_chart_name[key].lower()}', fontsize=20)
            plt.xlabel(f'{idx_to_chart_name[key]}', fontsize=18)
            plt.ylabel('Price/strike ratio', fontsize=18)
            plt.legend(fontsize=18)
            plt.savefig(f'../charts/fixed_{value}_with_sub_{idx}.svg')


if __name__ == '__main__':
    indices = [3]
    create_convexity_plots(indices)
