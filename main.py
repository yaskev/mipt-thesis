from monte_carlo import create_dataset

if __name__ == '__main__':
    df = create_dataset(100)
    df.to_csv('options_prices.csv', index=False, float_format='%.3f')
