import numpy as np


def cholesky_inv(a):
    """
    Calculate the inverse of matrix a based on Cholesky matrix decomposition.

    :param a: numpy.ndarray
        The positive definite matrix a to be inverted.
    :return: numpy.ndarray
        The inverse of a.
    """
    l_inv = np.linalg.inv(np.linalg.cholesky(a))
    return l_inv.T.dot(l_inv)
