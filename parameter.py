import numpy as np


def get_uniform(sample):
    return np.arange(len(sample)) / (len(sample) - 1)


def get_cum_chord(sample):
    delta_p = get_dis(sample)
    delta_p = np.hstack([0, delta_p[:-1]])
    cum = np.cumsum(delta_p)
    return cum / cum[-1]


def get_entad(sample):
    delta_p = np.sqrt(get_dis(sample))
    delta_p = np.hstack([0, delta_p[:-1]])
    cum = np.cumsum(delta_p)
    return cum / cum[-1]
