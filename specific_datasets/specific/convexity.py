import matplotlib.pyplot as plt
import pandas as pd

idx_to_param = {
    0: 'spot',
    1: 'ttm',
    2: 'rate',
    3: 'vol'
}

idx_to_col_name = {
    0: 'spot_strike_ratio',
    1: 'ttm',
    2: 'risk_free_rate',
    3: 'volatility'
}

fixed_params_values = {
    0: 'Spot/strike = 1',
    1: 'TTM = 1',
    2: 'Rate = 0.1',
    3: 'Volatility = 0.2'
}

idx_to_chart_name = {
    0: 'Spot/strike ratio',
    1: 'TTM',
    2: 'Risk-free rate',
    3: 'Volatility'
}


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
