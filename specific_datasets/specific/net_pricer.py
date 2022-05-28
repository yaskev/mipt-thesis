import pandas as pd

from main import make_predicted_df
from positive_network.net_maker import get_trained_net_and_test_set as get_positive_net_and_test_set
from convex_network.net_maker import get_trained_net_and_test_set as get_convex_net_and_test_set
from settings import FIXED_AVG_TYPE

from specific_datasets.specific.convexity import idx_to_param


def get_net_prices():
    for key, value in idx_to_param.items():
        df = pd.read_csv(f'../fixed_{value}_dataset.csv')

        con_net, con_x_test, con_y_test = get_convex_net_and_test_set(df, test_size=0.1,
                                                                      fixed_avg_type=FIXED_AVG_TYPE,
                                                                      no_charts=True)
        # pos_net, pos_x_test, pos_y_test = get_positive_net_and_test_set(df, test_size=0.1,
        #                                                                 fixed_avg_type=FIXED_AVG_TYPE,
        #                                                                 no_charts=True)

        predict_price = con_net.predict(con_x_test).detach().numpy()
        df_test = make_predicted_df(con_x_test, con_y_test, predict_price, fixed_avg_type=FIXED_AVG_TYPE)
        df_test.to_csv(f'../convex/fixed_{value}.csv', index=False, float_format='%.4f')

        # predict_price = pos_net.predict(pos_x_test).detach().numpy()
        # df_test = make_predicted_df(pos_x_test, pos_y_test, predict_price, fixed_avg_type=FIXED_AVG_TYPE)
        # df_test.to_csv(f'../positive/fixed_{value}.csv', index=False, float_format='%.4f')


if __name__ == '__main__':
    get_net_prices()
