import numpy as np
import scipy.interpolate as interpolate


def get_bspline(x, y, k):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    if x[0] > x[1]:
        x = x[::-1]
        y = y[::-1]
    return interpolate.make_interp_spline(x, y, k)
