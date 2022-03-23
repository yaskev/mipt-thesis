import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monte_carlo import create_dataset

if __name__ == '__main__':
    df = create_dataset(100)
    df.to_csv('av.csv', index=False, float_format='%.4f')

    av = pd.read_csv('sandbox/av.csv')
    noav = pd.read_csv('sandbox/no_av.csv')

    av = pd.DataFrame(av, columns=['price_strike_ratio', 'left_ci', 'right_ci'])
    noav = pd.DataFrame(noav, columns=['price_strike_ratio', 'left_ci', 'right_ci'])

    av.sort_values('price_strike_ratio', inplace=True)
    noav.sort_values('price_strike_ratio', inplace=True)

    plt.figure(figsize=(12, 8))
    plt.grid()
    plt.plot(np.arange(100), av['price_strike_ratio'], linewidth=2)
    plt.fill_between(np.arange(100), av['left_ci'], av['right_ci'], color='b', alpha=.2, label='with AV')
    plt.fill_between(np.arange(100), noav['left_ci'], noav['right_ci'], color='r', alpha=.1, label='no AV')
    plt.title('Price with confidence intervals')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('sandbox/ci.jpg')
