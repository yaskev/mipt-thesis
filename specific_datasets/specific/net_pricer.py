import pandas as pd

from main import make_predicted_df
from positive_network.net_maker import get_trained_net_and_test_set as get_positive_net_and_test_set
from convex_network.net_maker import get_trained_net_and_test_set as get_convex_net_and_test_set
from semipositive_network.net_maker import get_trained_net_and_test_set as get_semipositive_net_and_test_set
from settings import FIXED_AVG_TYPE
from utils.mapping import idx_to_param
from utils.typing import OptionAvgType


def get_net_prices(idxs):
    for key, value in idx_to_param.items():
        # if value == 'vol':
        #     continue

        for idx in idxs:
            df = pd.read_csv(f'../fixed_{value}_dataset_with_sub_{idx}(0_5-0_95).csv')

            # con_net, con_x_test, con_y_test, _, _ = get_convex_net_and_test_set(df, df, test_size=1,
            #                                                               fixed_avg_type=FIXED_AVG_TYPE,
            #                                                               no_charts=True)
            pos_net, pos_x_test, pos_y_test, _, _ = get_positive_net_and_test_set(df, df, test_size=1,
                                                                            fixed_avg_type=FIXED_AVG_TYPE,
                                                                            no_charts=True)
            # semipos_net, semipos_x_test, semipos_y_test, _, _ = get_semipositive_net_and_test_set(df, df, test_size=1,
            #                                                                       fixed_avg_type=FIXED_AVG_TYPE,
            #                                                                       no_charts=True)

            # predict_price = con_net.predict(con_x_test).detach().numpy()
            # df_test = make_predicted_df(con_x_test, con_y_test, predict_price, fixed_avg_type=FIXED_AVG_TYPE)
            # df_test.to_csv(f'../convex/fixed_{value}_5.csv', index=False, float_format='%.4f')

            predict_price = pos_net.predict(pos_x_test).detach().numpy()
            df_test = make_predicted_df(pos_x_test, pos_y_test, predict_price, fixed_avg_type=FIXED_AVG_TYPE)
            df_test.to_csv(f'../positive/fixed_{value}_with_sub_{idx}.csv', index=False, float_format='%.4f')

            # predict_price = semipos_net.predict(semipos_x_test).detach().numpy()
            # df_test = make_predicted_df(semipos_x_test, semipos_y_test, predict_price, fixed_avg_type=FIXED_AVG_TYPE)
            # df_test.to_csv(f'../semipositive/fixed_{value}_with_sub_{idx}.csv', index=False, float_format='%.4f')


if __name__ == '__main__':
    indices = [3]
    get_net_prices(indices)
