import numpy as np


def direc_delta(x):
    if np.isscalar(x):
        return 0
    else:
        return np.zeros(x.shape)
