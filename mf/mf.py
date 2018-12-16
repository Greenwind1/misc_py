import numpy as np


def user_item_error(r, p, q):
    if r == 0:
        return 0
    else:
        return r - np.dot(p, q)


def error(R, P, Q, beta):
    e = 0.0
    beta2 = beta / 2.
    for i in range(len(R)):
        for j in range(len(R[i])):
            e += pow(user_item_error(R[i][j], P[:, i], Q[:, j]), 2)
            e += beta2 * (np.linalg.norm(P) + np.linalg.norm(Q))
    return e


def matrix_factorization(R, k,
                         seed=0, steps=5000, alpha=0.0002,
                         beta=0.02, th=0.001):
    """
    :integer k: hyper paramter of MF
    :matrix P, Q: decompsed matrices
    """
    np.random.seed(seed=seed)
    P = np.random.rand(k, len(R))
    Q = np.random.rand(k, len(R[0]))
    for step in range(steps):

        # update of P and Q matrices
        for i in range(len(R)):
            for j in range(len(R[i])):
                err = user_item_error(R[i][j], P[:, i], Q[:, j])
                for kk in range(k):
                    P[kk][i] += alpha * (2 * err * Q[kk][j])
                    Q[kk][j] += alpha * (2 * err * P[kk][i])

        e = error(R, P, Q, beta)
        if e < th:
            break
    return P, Q


# set Rating matrix
R = np.array([[5, 3, 0, 1],
              [4, 0, 0, 1],
              [1, 1, 0, 5],
              [1, 0, 0, 4],
              [0, 1, 5, 4]])

nP, nQ = matrix_factorization(R, 2)
nR = np.dot(nP.T, nQ)
print(np.round(nR, 3))
