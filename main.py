from utils.typing import OptionAvgType, OptionType
from monte_carlo import generate_paths, get_option_price

if __name__ == '__main__':
    paths = generate_paths(100, 1, 0.05, 0.2)
    price = get_option_price(paths, 110, 0.05, 1, OptionAvgType.GEOMETRIC, OptionType.CALL)
    print(price)
