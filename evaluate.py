import numpy as np


def evaluate(f, t, xy):
    pred = f(t)
    diff = pred - xy
    return np.sqrt(np.square(diff).mean())
