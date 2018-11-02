# -*- coding: utf-8 -*-

import numpy as np


def Mq_cAb(c, A, b):
    """
    :array c: (n x 1) matrix
    :array A: (m x n) matrix
    :array b: (m x 1) matrix
    primal problem
        c^T x --> max
        Ax <= b, x >= 0
    """
    assert isinstance(c, np.ndarray)
    assert isinstance(A, np.ndarray)
    assert isinstance(b, np.ndarray)
    m, n = A.shape
    m1 = np.hstack([np.zeros((m, m)), -A, b])
    m2 = np.hstack([A.T, np.zeros((n, n)), -c])
    m3 = np.append(np.append(-b.T, c.T), 0)
    M = np.vstack((m1, m2, m3))
    q = np.zeros((m + n + 1, 1))
    return M, q


if __name__ == '__main__':
    c = np.array([[150], [200], [300]])
    A = np.array([[3, 1, 2], [1, 3, 0], [0, 2, 4]])
    b = np.array([[60], [36], [48]])
    M, q = Mq_cAb(c, A, b)
    print(M, q)
