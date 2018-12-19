import numpy as np


def _get_dis(xy):
    diff_xy = xy - np.concatenate([xy[1:, :], xy[0:1, :]])
    dis = np.sqrt(np.square(diff_xy).sum(axis=1))
    return dis


def get_uniform(sample):
    return np.arange(len(sample)) / (len(sample) - 1)


def get_cum_chord(sample):
    delta_p = _get_dis(sample)
    delta_p = np.hstack([0, delta_p[:-1]])
    cum = np.cumsum(delta_p)
    return cum / cum[-1]


def get_entad(sample):
    delta_p = np.sqrt(_get_dis(sample))
    delta_p = np.hstack([0, delta_p[:-1]])
    cum = np.cumsum(delta_p)
    return cum / cum[-1]
