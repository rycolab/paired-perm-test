import numpy as np
from numba import jit

from pairedpermtest.utils import exact_acc


@jit(nopython=True, nogil=True, cache=True)
def pmf(xs, ys, max_diff):
    """
    Compute the exact PMF for the paired permutation test in difference of accuracy
    This method uses an array to support fast loops/access.
    Runs in O(LN^2) time
    :param xs: accuracy scores for system A
    :param ys: accuracy scores for system B
    :return: PMF distribution for paired permutation test
    """
    N = len(xs)
    W_old = np.zeros(max_diff * N * 2 + 1)
    W_old[max_diff * N] = 1.0
    W_new = np.zeros(max_diff * N * 2 + 1)
    for n in range(0, N):
        for diff, val in enumerate(W_old):
            if not val:
                continue
            for (x, y) in [(xs[n], ys[n]), (ys[n], xs[n])]:
                d = diff + x - y
                W_new[d] += 0.5 * val
        W_old = np.copy(W_new)
        W_new = np.zeros(max_diff * N * 2 + 1)
    return W_old


@jit(nopython=True, nogil=True, cache=True)
def perm_test(xs, ys):
    """
    Run exact paired permutation test on difference in accuracy.
    Uses array to store PMF (see function pmf_array).
    Runs in O(LN^2) time
    :param xs: accuracy scores for system A
    :param ys: accuracy scores for system B
    :return: exact p-value  for paired permutation test
    """
    N = len(xs)
    observed = np.abs(np.sum(xs - ys))
    max_diff = np.max(np.abs(xs - ys))
    P = pmf(xs, ys, max_diff)
    if observed == 0:
        return np.sum(P)
    positive_edge = max_diff * N + observed
    negative_edge = max_diff * N - observed + 1
    p = np.sum(P[:negative_edge]) + np.sum(P[positive_edge:])
    return p


def tests():
    N = 10
    C = 5
    for _ in range(20):
        xs = np.random.randint(0, C, N)
        ys = np.random.randint(0, C, N)
        x = perm_test(xs, ys)
        y = exact_acc(xs, ys)
        assert np.isclose(x, y), f"{y} =!= {x}"
    print('ok')


if __name__ == '__main__':
    tests()
