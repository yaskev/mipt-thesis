import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import default_rng

from monte_carlo.dataset_maker import _get_price
from settings import OPTIONS_PARAMS_RANDOM_SEED
from utils.typing import OptionAvgType

if __name__ == '__main__':
    # rng = default_rng(OPTIONS_PARAMS_RANDOM_SEED)
    # data = {
    #     'spot_strike_ratio': 0.5 + rng.random(10000),
    #     'ttm': 1,
    #     'risk_free_rate': 0.1,
    #     'volatility': 0.2,
    #     'avg_type': OptionAvgType.ARITHMETIC.value,
    # }
    # df = pd.DataFrame(data=data)
    # price_and_ci_df = df.apply(_get_price, axis=1, result_type='expand')
    # price_and_ci_df.columns = ['price_strike_ratio', 'left_ci', 'right_ci']
    # df = pd.concat([df, price_and_ci_df], axis=1)
    #
    # df.to_csv('../fixed_params_dataset.csv', index=False, float_format='%.4f')

    df = pd.read_csv('../specific_datasets/fixed_convex_net_prices.csv')

    df.sort_values('spot_strike_ratio', inplace=True)

    plt.figure(figsize=(12, 8))
    plt.text(0.61, 0.45, 'Fixed params:', fontsize=20)
    plt.text(0.61, 0.39, 'TTM = 1', fontsize=20)
    plt.text(0.61, 0.33, 'Rate = 0.1', fontsize=20)
    plt.text(0.61, 0.27, 'Volatility = 0.2', fontsize=20)
    plt.text(0.61, 0.21, 'Avg = Arithmetic', fontsize=20)
    plt.grid()
    plt.plot(df['spot_strike_ratio'], df['net_price'], linewidth=2)
    plt.title('Price by spot, convex net', fontsize=20)
    plt.xlabel('Spot to strike ratio', fontsize=18)
    plt.ylabel('Price to strike ratio', fontsize=18)
    plt.savefig('convexity_convex_net.jpg')
