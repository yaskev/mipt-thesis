from enum import Enum


class OptionAvgType(Enum):
    ARITHMETIC = 'ARITHMETIC'
    GEOMETRIC = 'GEOMETRIC'


class NetType(Enum):
    CONVEX = 'CONVEX'
    POSITIVE = 'POSITIVE'


class ComplexNetworkType(Enum):
    POSITIVE_NETWORK = 'positive_network'
    CONVEX_NETWORK = 'convex_network'
    SIGMA_NETWORK = 'sigma_network'
    SEMIPOSITIVE_NETWORK = 'semipositive_network'
