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

def create_convexity_plots(pricer: DoubleNetPricer):
    for i in range(1, 8):
        df = pd.read_csv(f'datasets_for_graphs/{i}.csv')
        predicted_df = pricer.predict(df, no_decode=True)

        predicted_df.sort_values(file_idx_to_col[i], inplace=True)

        plt.figure(figsize=(12, 8))
        plt.grid()
        plt.plot(predicted_df[file_idx_to_col[i]], predicted_df['monte_carlo_price'], linewidth=2, label='MC')
        plt.plot(predicted_df[file_idx_to_col[i]], predicted_df['net_price'], linewidth=2, label='Positive net')
        plt.title(f'Price by {file_idx_to_col[i].lower()}. Moneyness={1.1 if i < 5 else 0.9}', fontsize=20)
        plt.xlabel(f'{file_idx_to_col[i]}', fontsize=18)
        plt.ylabel('Price/strike ratio', fontsize=18)
        plt.legend(fontsize=18)
        plt.savefig(f'graphs/{i}.jpg')


if __name__ == '__main__':
    left_model = joblib.load('models/left-2024-03-05 16:16:25.028609.sav')
    right_model = joblib.load('models/right-2024-03-05 16:10:21.678640.sav')
    pricer = DoubleNetPricer(left_model, right_model)

    create_convexity_plots(pricer)