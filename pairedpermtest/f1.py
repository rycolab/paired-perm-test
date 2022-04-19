import numpy as np
from numba import jit
from typing import Dict, Tuple

from pairedpermtest.utils import f1, exact_f1


@jit(nopython=True, nogil=True, cache=True)
def pmf(xs, ys):
    T = len(xs)
    W_old: Dict[Tuple[int, int, int, int], float] = {(0, 0, 0, 0): 1.0}
    W_new: Dict[Tuple[int, int, int, int], float] = {(0, 0, 0, 0): 0.0}
    for t in range(0, T):
        for tp_x, tp_y, i_x, i_y in W_old:
            val = W_old[(tp_x, tp_y, i_x, i_y)]
            for (tp_x_, i_x_), (tp_y_, i_y_) in [(xs[t], ys[t]), (ys[t], xs[t])]:
                tp_x_n = tp_x + tp_x_
                tp_y_n = tp_y + tp_y_
                i_x_n = i_x + i_x_
                i_y_n = i_y + i_y_
                if (tp_x_n, tp_y_n, i_x_n, i_y_n) not in W_new:
                    W_new[(tp_x_n, tp_y_n, i_x_n, i_y_n)] = 0
                W_new[(tp_x_n, tp_y_n, i_x_n, i_y_n)] += 0.5 * val
        if not W_new[(0, 0, 0, 0)]:
            W_new.pop((0, 0, 0, 0))
        W_old = W_new
        W_new = {(0, 0, 0, 0): 0.0}
    return W_old


def perm_test(xs, ys):
    def f1_delta(tp_x, tp_y, i_x, i_y):
        x = tp_x / (tp_x + 0.5 * i_x) if tp_x + i_x != 0 else 0
        y = tp_y / (tp_y + 0.5 * i_y) if tp_y + i_y != 0 else 0
        return x - y

    def effect(xs, ys): return np.abs(f1(xs) - f1(ys))

    observed = effect(xs, ys)
    P = pmf(xs, ys)
    p = sum(p for (tp_x, tp_y, i_x, i_y), p in P.items()
            if abs(f1_delta(tp_x, tp_y, i_x, i_y)) >= observed)
    return p


def _test_f1(N):
    xs = np.random.randint(0, 5, (N, 2))
    ys = np.random.randint(0, 5, (N, 2))
    x = perm_test(xs, ys)
    y = exact_f1(xs, ys)
    assert x == y


def tests():
    for N in range(4, 10):
        for _ in range(20):
            _test_f1(N)
    print('ok')


if __name__ == '__main__':
    tests()
