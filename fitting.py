import numpy as np
import scipy.interpolate as interpolate


def get_bspline(x, y, k, bc_type=None):
    return interpolate.make_interp_spline(x, y, k, bc_type=bc_type)
