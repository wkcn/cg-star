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
            return np.array(Phi * self.control_p)
    return BezierCurve(k, control_p)


def _get_gradient_coef_power(k, order):
    assert int(order) == order
    assert 0 <= order <= k
    power = np.arange(k + 1)
    coef = np.ones_like(power)
    for _ in range(order):
        coef *= power
        power -= 1
    invalid = power < 0
    coef[invalid] = 0
    power[invalid] = 0
    return coef, power


def get_poly(T, P, k, bc_type=None):
    # assert bc_type is None, "Doesn't support bc_type now :("
    power = np.arange(k + 1)
    A = np.matrix(np.power(T.reshape((-1, 1)), power))

    if bc_type is not None:
        # gradient
        lefts, rights = bc_type
        for gs, t in [(lefts, T[0]), (rights, T[-1])]:
            for order, value in gs:
                # extra row for A and b
                c, p = _get_gradient_coef_power(k, order)
                eA = np.matrix(np.power(t, p) * c)
                A = np.concatenate([A, eA], 0)
                P = np.concatenate([P, value.reshape((-1, P.shape[1]))], 0)
    At = A.T
    AtA = At * A
    Atb = At * P
    coef = AtA.I * Atb

    class PolyCurve:
        def __init__(self, k, coef):
            self.k = k
            self.coef = coef

        def __call__(self, T):
            power = np.arange(k + 1)
            A = np.matrix(np.power(T.reshape((-1, 1)), power))
            return np.array(A * coef)
    return PolyCurve(k, coef)
