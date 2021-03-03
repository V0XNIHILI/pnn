import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    s = sigmoid(x)

    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1 - np.tanh(x)**2


def relu(x):
    return x * (x > 0)


def d_relu(x):
    return (x > 0) * 1


def binary(x):
    return d_relu(x)


def d_binary_step(x):
    return 0