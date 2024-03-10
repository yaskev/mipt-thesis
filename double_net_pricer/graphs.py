import joblib
import matplotlib.pyplot as plt
import pandas as pd

from double_net_pricer.pricer import DoubleNetPricer

file_idx_to_col = {
    1: 'spot_strike_ratio',
    2: 'ttm',
    3: 'risk_free_rate',
    4: 'volatility',
    5: 'risk_free_rate',
    6: 'ttm',
    7: 'volatility'
}

file_idx_to_label = {
    1: 'Spot/Strike ratio',
    2: 'TTM',
    3: 'Risk-free rate',
    4: 'Volatility',
    5: 'Risk-free rate',
    6: 'TTM',
    7: 'Volatility'
}

param_to_col = {
    'rate': 'risk_free_rate',
    'spot': 'spot_strike_ratio',
    'ttm': 'ttm',
    'vol': 'volatility',
}

def create_convexity_plots(pricer: DoubleNetPricer, task: str):
    postfix = '' if task == 'normal' else '_low'
    for i in range(1, 8):
        df = pd.read_csv(f'datasets_for_graphs/{i}{postfix}.csv')
        predicted_df = pricer.predict(df, no_decode=True)

        predicted_df.sort_values(file_idx_to_col[i], inplace=True)

        plt.figure(figsize=(12, 8))
        plt.grid()
        plt.plot(predicted_df[file_idx_to_col[i]], predicted_df['monte_carlo_price'], linewidth=2, label='Monte Carlo')
        plt.plot(predicted_df[file_idx_to_col[i]], predicted_df['net_price'], linewidth=2, label='Positive net')
        plt.title(f'Price by {file_idx_to_label[i]}. Moneyness={1.1 if i < 5 else 0.9}', fontsize=20)
        # plt.title(f'Price by {file_idx_to_label[i]}', fontsize=20)
        plt.xlabel(f'{file_idx_to_label[i]}', fontsize=18)
        plt.ylabel('Price/strike ratio', fontsize=18)
        plt.legend(fontsize=18)
        plt.savefig(f'graphs/{i}{postfix}.svg')

    # if task == 'low':
    #     for param in ['rate', 'spot', 'ttm', 'vol']:
    #         df = pd.read_csv(f'datasets_for_graphs/low_ttm_fixed_{param}.csv')
    #         predicted_df = pricer.predict(df, no_decode=True)
    #
    #         predicted_df.sort_values(param_to_col[param], inplace=True)
    #
    #         plt.figure(figsize=(12, 8))
    #         plt.grid()
    #         plt.plot(predicted_df[param_to_col[param]], predicted_df['monte_carlo_price'], linewidth=2, label='MC')
    #         plt.plot(predicted_df[param_to_col[param]], predicted_df['net_price'], linewidth=2, label='Positive net')
    #         plt.title(f'Price by {param_to_col[param].lower()}. Moneyness={1.1}', fontsize=20)
    #         plt.xlabel(f'{param_to_col[param]}', fontsize=18)
    #         plt.ylabel('Price/strike ratio', fontsize=18)
    #         plt.legend(fontsize=18)
    #         plt.savefig(f'graphs/{param}.jpg')


if __name__ == '__main__':
    # TASK = 'normal'
    TASK = 'low'

    # not convex, normal
    # left_model = joblib.load('models/left-2024-03-05 16:16:25.028609.sav')
    # right_model = joblib.load('models/right-2024-03-05 19:40:03.951467.sav')

    # not convex, low
    left_model = joblib.load('models/left-2024-03-07 17:04:46.667532.sav')
    right_model = joblib.load('models/right-2024-03-07 16:34:26.889750.sav')

    # Convex
    # left_model = joblib.load('convex/models/left-2024-03-08 16:36:45.264273.sav')
    # right_model = joblib.load('convex/models/right-2024-03-08 17:18:26.183705.sav')
    pricer = DoubleNetPricer(left_model, right_model)

    create_convexity_plots(pricer, TASK)