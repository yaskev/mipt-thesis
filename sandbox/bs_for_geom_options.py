import numpy as np
import pandas as pd
from scipy.stats import norm


def get_mean_greeks():
    pos_greeks = np.array([[0.8641, 0.5139, 0.1477, 0.4491, 0.2724, -0.0119, 0.0520, 0.0177, 1.6773],
                           [1.4098, 1.3742, 0.1202, 0.1618, 1.0805, -0.0474, 0.2062, 0.0704, -0.3073],
                           [1.9397, 0.6757, 0.0068, 0.1439, 0.9818, -0.0255, 0.1110, 0.0379, -1.6399],
                           [0.5255, 0.9587, 0.1296, 0.2390, 0.0727, -0.0032, 0.0139, 0.0047, 0.2154],
                           [1.3682, 0.5417, 0.1711, 0.5094, 1.0562, -0.0463, 0.2015, 0.0688, -0.3951]])

    con_greeks = np.array([[0.8641, 0.5139, 0.1477, 0.4491, 0.3113, -0.0053, 0.0199, 0.0078, 1.2427],
                           [1.4098, 1.3742, 0.1202, 0.1618, 0.8979, -0.0253, 0.1079, 0.0444, 0.4634],
                           [1.9397, 0.6757, 0.0068, 0.1439, 1.0145, -0.0582, 0.2547, 0.0961, 0.0266],
                           [0.5255, 0.9587, 0.1296, 0.2390, 0.0834, -0.0013, 0.0049, 0.0020, 0.3333],
                           [1.3682, 0.5417, 0.1711, 0.5094, 0.8853, -0.0252, 0.0963, 0.0382, 0.5805]])

    mean_greeks = (pos_greeks + con_greeks) / 2
    pd.DataFrame(mean_greeks).to_csv('mean_greeks.csv', index=False, float_format='%.4f')


def get_precise_price():
    params = np.array([[1.7006, 0.7712, 0.0503, 0.3867],
                       [1.1763, 0.7434, 0.1320, 0.1232],
                       [0.9228, 1.4121, 0.1239, 0.4093],
                       [1.2529, 0.6457, 0.1994, 0.3754],
                       [1.1791, 0.7208, 0.1686, 0.4109]])

    params = pd.DataFrame(params, columns=['spot_strike_ratio', 'ttm', 'risk_free_rate', 'vol'])
    params['vol_g'] = params['vol'] / np.sqrt(3)
    params['b'] = 1/2 * (-1/2 * params['vol_g']**2 + params['risk_free_rate'])
    params['d_1'] = (np.log(params['spot_strike_ratio']) + (params['b'] + 1/2 * params['vol_g']**2)*params['ttm']) / (params['vol_g'] * np.sqrt(params['ttm']))
    params['d_2'] = params['d_1'] - params['vol_g'] * np.sqrt(params['ttm'])

    params['price'] = params['spot_strike_ratio'] * np.exp((params['b'] - params['risk_free_rate'])*params['ttm']) * norm.cdf(params['d_1']) - np.exp(-params['risk_free_rate'] * params['ttm'])*norm.cdf(params['d_2'])

    print(params['price'])


if __name__ == '__main__':
    get_precise_price()
