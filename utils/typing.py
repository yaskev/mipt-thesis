from enum import Enum


class OptionAvgType(Enum):
    ARITHMETIC = 'ARITHMETIC'
    GEOMETRIC = 'GEOMETRIC'


class NetType(Enum):
    CONVEX = 'CONVEX'
    POSITIVE = 'POSITIVE'
