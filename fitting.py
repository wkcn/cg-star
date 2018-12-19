import numpy as np
import scipy.interpolate as interpolate
from scipy.misc import comb


def get_bspline(T, P, k, bc_type=None):
    return interpolate.make_interp_spline(T, P, k, bc_type=bc_type)


def _bernsteain_poly(i, n, t):
    return comb(n, i) * (t ** i) * (1 - t) ** (n - i)


def get_bezier(T, P, k, bc_type=None):
    assert bc_type is None, "Doesn't support bc_type now :("
    Phi = np.matrix([[_bernsteain_poly(i, k, t) for i in range(k+1)] for t in T])
    # Ax = b
    # A^TAx = A^Tb
    At = Phi.T
    AtA = At * Phi
    Atb = At * P
    control_p = AtA.I * Atb
    class BezierCurve:
        def __init__(self, k, control_p):
            self.k = k
            self.control_p = control_p
        def __call__(self, T):
            k = self.k
            Phi = np.matrix([[_bernsteain_poly(i, k, t) for i in range(k+1)] for t in T])
            return Phi * self.control_p
    return BezierCurve(k, control_p)
