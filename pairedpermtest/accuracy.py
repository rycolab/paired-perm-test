import numpy as np
from numba import jit
from typing import Dict

from pairedpermtest.utils import exact_acc


@jit(nopython=True, nogil=True, cache=True)
def pmf(xs, ys):
    T = len(xs)
    W_old: Dict[int, float] = {0: 1.0}
    W_new: Dict[int, float] = {0: 0.0}
    for t in range(0, T):
        for diff in W_old:
            val = W_old[diff]
            for (x, y) in [(xs[t], ys[t]), (ys[t], xs[t])]:
                d = diff + x - y
                if d not in W_new:
                    W_new[d] = 0
                W_new[d] += 0.5 * val
        if not W_new[0]:
            W_new.pop(0)
        W_old = W_new
        W_new = {0: 0.0}
    return W_old


@jit(nopython=True, nogil=True, cache=True)
def perm_test(xs, ys):
    observed = np.abs(np.sum(xs - ys))
    P: Dict[int, float] = pmf(xs, ys)
    p = 0.0
    for diff, val in P.items():
        if abs(diff) >= observed:
            p += val
    return p


@jit(nopython=True, nogil=True, cache=True)
def pmf_array(xs, ys, max_diff):
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
def perm_test_array(xs, ys):
    N = len(xs)
    observed = np.abs(np.sum(xs - ys))
    max_diff = np.max(np.abs(xs - ys))
    P = pmf_array(xs, ys, max_diff)
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
        x_array = perm_test_array(xs, ys)
        y = exact_acc(xs, ys)
        assert np.isclose(x, y), f"{y} =!= {x}"
        assert np.isclose(x_array, y), f"{y} =!= {x_array}"
    print('ok')


if __name__ == '__main__':
    tests()
