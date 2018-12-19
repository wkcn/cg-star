import numpy as np


def evaluate(f, t, xy):
    pred = f(t)
    diff = pred - xy
    return np.sqrt(np.square(diff).sum(1)).mean()


def diff(a, b, order=2):
    assert a.ndim == 2
    assert b.ndim == 2
    assert a.shape == b.shape
    return np.linalg.norm(a-b, ord=order, axis=1).mean()
