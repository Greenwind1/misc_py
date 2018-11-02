# -*- coding: utf-8 -*-

import numpy as np


def artProb_init(M, q):
    """
    :array q: (n x 1) matrix
    :array M: (n x n) matrix
    """
    assert isinstance(M, np.ndarray)
    assert isinstance(q, np.ndarray)

    n, n = M.shape
    x0 = np.ones((n, 1))
    mu0 = np.dot(q.T, x0) / (n + 1) + 1
    z0 = mu0 / x0
    r = z0 - np.dot(M, x0) - q
    qn1 = (n + 1) * mu0 - np.dot(q.T, x0)

    MM = np.hstack((M, r))
    MM = np.vstack([MM, np.append(-r.T, 0)])
    qq = np.vstack([q, qn1])
    xx0 = np.vstack([x0, np.array([[1]])])
    zz0 = np.vstack([z0, mu0])

    return MM, qq, xx0, zz0


if __name__ == '__main__':
    M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    q = np.array([[0], [0], [0]])
    MM, qq, xx0, zz0 = artProb_init(M, q)
    print(MM, qq, xx0, zz0)
