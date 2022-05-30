import pandas as pd

from monte_carlo.dataset_maker import _get_price
from utils.mapping import idx_to_col_name, idx_to_greek

EPS = 0.01


def get_greeks(data: pd.DataFrame) -> pd.DataFrame:
    print('original')
    original_price_ci_df = data.apply(_get_price, axis=1, result_type='expand')
    original_price_ci_df.columns = ['price_strike_ratio', 'left_ci', 'right_ci']

    greek_dict = dict()

    for key, value in idx_to_col_name.items():
        print(f'{value}')
        shifted_data = data.copy(deep=True)
        shifted_data[idx_to_col_name[key]] = shifted_data[idx_to_col_name[key]] * (1 + EPS)
        shifted_price_ci_df = shifted_data.apply(_get_price, axis=1, result_type='expand')
        shifted_price_ci_df.columns = ['price_strike_ratio', 'left_ci', 'right_ci']

        greeks = (shifted_price_ci_df['price_strike_ratio'] - original_price_ci_df['price_strike_ratio']) / (data[idx_to_col_name[key]] * EPS)
        greek_dict[idx_to_greek[key]] = greeks

        right_ci_greek = (shifted_price_ci_df['right_ci'] - original_price_ci_df['left_ci']) / (data[idx_to_col_name[key]] * EPS)
        left_ci_greek = (shifted_price_ci_df['left_ci'] - original_price_ci_df['right_ci']) / (data[idx_to_col_name[key]] * EPS)
        greek_dict[f'{idx_to_greek[key]}_l_ci'] = left_ci_greek
        greek_dict[f'{idx_to_greek[key]}_r_ci'] = right_ci_greek

    res_df = pd.DataFrame(greek_dict)
    res_df['theta_mc'] = -res_df['theta_mc']
    return res_df
