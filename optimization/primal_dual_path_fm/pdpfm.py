# -*- coding: utf-8 -*-

import numpy as np
from primal_dual_path_fm.Mq_cAb import Mq_cAb
from primal_dual_path_fm.artProb_init import artProb_init
from primal_dual_path_fm.bin_search import bin_search


def pdpfm(c, A, b, beta=0.5, precision=0.001, MEPS=1e-10):
    M0, q0 = Mq_cAb(c, A, b)
    M, q, x, z = artProb_init(M0, q0)

    m, k = A.shape
    n, n = M.shape

    # optimization process...
    print('\noptimization process...')
    count = 0
    mu = np.dot(x.T, z) / n
    while mu > MEPS:
        count += 1
        print('\n{} th step :'.format(count))
        # predict...
        delta = 0
        dx = np.dot(np.linalg.inv(M + np.diag((z / x).flatten())),
                    delta * mu * (1 / x) - z)
        dz = delta * mu * (1 / x) - z - (1 / x) * z * dx
        th = bin_search(x, z, dx, dz, beta, precision)
        print('theta = {:.5f}'.format(th), end=' , ')
        x += th * dx
        z += th * dz
        mu = np.dot(x.T, z) / n
        # adjust...
        delta = 1
        dx = np.dot(np.linalg.inv(M + np.diag((z / x).flatten())),
                    delta * mu * (1 / x) - z)
        dz = delta * mu * (1 / x) - z - (1 / x) * z * dx
        x += dx
        z += dz
        mu = np.dot(x.T, z) / n
        print('objective mu = {:.10f}'.format(mu.item()))

    # output process...
    x = x.flatten()
    if x[n - 2] > MEPS:
        pos = x[m:m + k] / x[n - 2]
        pov = np.dot(c.flatten(), x[m:m + k] / x[n - 2])
        print('Primal Optimal solution : ', pos)
        print('Primal Optimal value = ', pov)
        print('Dual Optimal solution : ', x[:m] / x[n - 2])
        print('Dual Optimal value = ',
              np.dot(b.flatten(), x[:m] / x[n - 2]))
        return pos, pov
    else:
        return print('Infeasible', end='\n')


if __name__ == '__main__':
    c = np.array([[150], [200], [300]])
    A = np.array([[3, 1, 2], [1, 3, 0], [0, 2, 4]])
    b = np.array([[60], [36], [48]])
    pdpfm(c, A, b)
