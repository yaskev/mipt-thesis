from monte_carlo import PathGenerator
from monte_carlo import Pricer
from utils.typing import OptionAvgType, OptionType

if __name__ == '__main__':
    paths = PathGenerator().generate_paths(100, 1, 0.05, 0.2)
    price = Pricer().get_option_price(paths, 90, 0.05, 1, OptionAvgType.GEOMETRIC, OptionType.PUT)
    print(price)
