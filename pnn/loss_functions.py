import numpy as np


def logloss(a, y):
    return -(y * np.log(a) + (1 - y) * np.log(1 - a))


def d_logloss(a, y):
    return (a - y) / (a * (1 - a))


def squared_error(a, y):
    return (a - y)**2


def d_squared_error(a, y):
    return 2 * (a - y)