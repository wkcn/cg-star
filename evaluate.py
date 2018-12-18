import numpy as np


def evaluate(f, x, y):
    pred = f(x)
    diff = pred - y
    return np.sqrt(np.square(diff).mean())
