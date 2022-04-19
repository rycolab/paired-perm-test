import numpy as np
from pairedpermtest.utils import exact_acc, exact_f1, f1
from numba import jit


@jit(nopython=True, nogil=True, cache=True)
def random_swap(xs, ys):
    n = len(xs)
    swaps = np.random.random(n) < 0.5
    xs_ = np.select([swaps, ~swaps], [xs, ys])
    ys_ = np.select([~swaps, swaps], [xs, ys])
    return xs_, ys_


@jit(nopython=True, nogil=True, cache=True)
def monte_carlo(xs, ys, K, effect):
    p_val = 0
    obs = effect(xs, ys)
    for _ in range(K):
        xs_, ys_ = random_swap(xs, ys)
        if effect(xs_, ys_) >= obs:
            p_val += 1
    return p_val / K


@jit(nopython=True, nogil=True, cache=True)
def acc(xs, ys):
    return np.abs(np.mean(xs) - np.mean(ys))


def perm_test_acc(xs, ys, K):
    return monte_carlo(xs, ys, K, acc)


@jit(nopython=True, nogil=True, cache=True)
def random_swap_pairs(xs, ys):
    n = len(xs)
    swaps = (np.random.random(n) < 0.5).repeat(2).reshape(n, -1)
    xs_ = np.select([swaps, ~swaps], [xs, ys])
    ys_ = np.select([~swaps, swaps], [xs, ys])
    return xs_, ys_


@jit(nopython=True, nogil=True, cache=True)
def monte_carlo_pairs(xs, ys, K, effect):
    p_val = 0
    obs = effect(xs, ys)
    for _ in range(K):
        xs_, ys_ = random_swap_pairs(xs, ys)
        if effect(xs_, ys_) >= obs:
            p_val += 1
    return p_val / K


@jit(nopython=True, nogil=True, cache=True)
def _f1(xs, ys):
    return np.abs(f1(xs) - f1(ys))


def perm_test_f1(xs, ys, K):
    return monte_carlo_pairs(xs, ys, K, _f1)


def test_acc():
    N = 10
    C = 5
    K = 10000
    xs = np.random.randint(0, C, N)
    ys = np.random.randint(0, C, N)
    x = perm_test_acc(xs, ys, K)
    y = exact_acc(xs, ys)
    assert np.isclose(x, y, rtol=5e-1), f"{y} =!= {x}"


def test_f1():
    N = 10
    C = 5
    K = 10000
    xs = np.random.randint(0, C, (N, 2))
    ys = np.random.randint(0, C, (N, 2))
    x = perm_test_f1(xs, ys, K)
    y = exact_f1(xs, ys)
    assert np.isclose(x, y, rtol=5e-1), f"{y} =!= {x}"


if __name__ == '__main__':
    print("Testing accuracy...")
    for _ in range(20):
        test_acc()
    print("ok")
    print("Testing f1...")
    for _ in range(20):
        test_f1()
    print("ok")