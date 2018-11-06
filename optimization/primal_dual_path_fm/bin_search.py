# -*- coding: utf-8 -*-

import numpy as np


def bin_search(x, z, dx, dz, beta=0.5, precision=0.001):
    """
    :array x: (n x 1) matrix
    :array z: (n x 1) matrix
    :array dx: (n x 1) matrix
    :array dz: (n x 1) matrix
    :float beta: N_2(beta)
    :float precision: threshold
    """
    n = x.shape[0]
    th_low = 0
    th_high = 1

    if (dx < 0).sum() > 0 or (dz < 0).sum() > 0:
        th_high = min(th_high,
                      np.min(-x[dx < 0] / dx[dx < 0]),
                      np.min(-z[dz < 0] / dz[dz < 0]))
    x_low = x + th_low * dx
    z_low = z + th_low * dz
    x_high = x + th_high * dx
    z_high = z + th_high * dz
    mu_high = np.dot(x_high.T, z_high).item() / n
    # from path of centers definition
    if beta * mu_high >= np.linalg.norm(x_high * z_high - mu_high):
        return th_high
    while th_high - th_low > precision:
        th_mid = (th_high + th_low) / 2
        x_mid = x + th_mid * dx
        z_mid = z + th_mid * dz
        mu_mid = np.dot(x_mid.T, z_mid).item() / n
        if beta * mu_mid >= np.linalg.norm(x_mid * z_mid - mu_mid):
            th_low = th_mid
        else:
            th_high = th_mid
    return th_low


if __name__ == '__main__':
    x = np.array([[1 / 2], [1]])
    dx = np.array([[-1 / 6], [-4 / 3]])
    z = np.array([[1], [1]])
    dz = np.array([[0.1], [-0.1]])
    th = bin_search(x, z, dx, dz)
    print(th)
