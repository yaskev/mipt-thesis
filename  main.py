from monte_carlo import PathGenerator
from monte_carlo import Pricer


if __name__ == '__main__':
    paths = PathGenerator().generate_paths()
    price = Pricer().get_option_price(paths)
    print(price)
