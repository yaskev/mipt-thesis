from enum import Enum


class OptionAvgType(Enum):
    ARITHMETIC = 'ARITHMETIC'
    GEOMETRIC = 'GEOMETRIC'


class OptionType(Enum):
    PUT = 'PUT'
    CALL = 'CALL'
