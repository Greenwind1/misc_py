# -*- coding: utf-8 -*-

import numpy as np
from primal_dual_path_fm.Mq_cAb import Mq_cAb


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

    # MM : ( n + 1 ) x ( n + 1 )
    # qq : ( n + 1 ) x 1
    # xx0 : ( n + 1 ) x 1
    # zz0 : ( n + 1 ) x 1
    return MM, qq, xx0, zz0


if __name__ == '__main__':
    c = np.array([[150], [200], [300]])
    A = np.array([[3, 1, 2], [1, 3, 0], [0, 2, 4]])
    b = np.array([[60], [36], [48]])
    M, q = Mq_cAb(c, A, b)
    MM, qq, xx0, zz0 = artProb_init(M, q)
    print(MM, qq, xx0, zz0)
