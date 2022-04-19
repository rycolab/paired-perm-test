import numpy as np
from pairedpermtest.utils import exact_acc, exact_f1, f1
from numba import jit


@jit(nopython=True, nogil=True, cache=True)
def random_swap(xs, ys):
    """
    Create a random pair of samples, (xs_, ys_), from (xs, ys).
    For element i:
        - xs_[i] = xs[i] and ys_[i] = ys[i] with probability 0.5
        - xs_[i] = ys[i] and ys_[i] = ss[i] with probability 0.5
    :param xs: scores of system A
    :param ys: scores of system B
    :param k:
    :return: paired samples (xs_, ys_) as in description
    """
    n = len(xs)
    k = xs.shape[-1] if len(xs.shape) > 1 else 1
    swaps = (np.random.random(n) < 0.5).repeat(k).reshape(n, k)
    xs_ = np.select([swaps, ~swaps], [xs.reshape(n, k), ys.reshape(n, k)])
    ys_ = np.select([~swaps, swaps], [xs.reshape(n, k), ys.reshape(n, k)])
    return xs_, ys_


@jit(nopython=True, nogil=True, cache=True)
def monte_carlo(xs, ys, K, effect):
    """
    Runs monte carlo sampling approximation of the paired permutation test
    based on score function effect with K samples
    :param xs: scores of system A
    :param ys: scores of system B
    :param K: number of samples
    :param effect: Scoring function between xs and ys
    :return: Approximate p-value of the paired permutation test
    """
    p_val = 0
    obs = effect(xs, ys)
    for _ in range(K):
        xs_, ys_ = random_swap(xs, ys)
        if effect(xs_, ys_) >= obs:
            p_val += 1
    return p_val / K


@jit(nopython=True, nogil=True, cache=True)
def acc_diff(xs, ys):
    """
    Returns absolute difference in accuracy between two systems
    :param xs: accuracy scores of system A
    :param ys: accuracy scores of system B
    :return: absolute difference in mean scores
    """
    return np.abs(np.mean(xs) - np.mean(ys))


def perm_test_acc(xs, ys, K):
    """
    Runs monte carlo sampling approximation of the paired permutation test
    based on difference of accuracy with K samples
    :param xs: scores of system A
    :param ys: scores of system B
    :param K: number of samples
    :return: Approximate p-value of the paired permutation test
    """
    return monte_carlo(xs, ys, K, acc_diff)


@jit(nopython=True, nogil=True, cache=True)
def f1_diff(xs, ys):
    """
    Returns absolute difference in F1 scores between two systems
    :param xs: accuracy scores of system A
    :param ys: accuracy scores of system B
    :return: absolute difference in F1 scores
    """
    return np.abs(f1(xs) - f1(ys))


def perm_test_f1(xs, ys, K):
    """
    Runs monte carlo sampling approximation of the paired permutation test
    based on difference of F1 scores with K samples
    :param xs: scores of system A
    :param ys: scores of system B
    :param K: number of samples
    :return: Approximate p-value of the paired permutation test
    """
    return monte_carlo(xs, ys, K, f1_diff)


def _test_acc():
    N = 10
    C = 5
    K = 10000
    xs = np.random.randint(0, C, N)
    ys = np.random.randint(0, C, N)
    x = perm_test_acc(xs, ys, K)
    y = exact_acc(xs, ys)
    assert np.isclose(x, y, rtol=5e-1), f"{y} =!= {x}"


def _test_f1():
    N = 10
    C = 5
    K = 10000
    xs = np.random.randint(0, C, (N, 2))
    ys = np.random.randint(0, C, (N, 2))
    x = perm_test_f1(xs, ys, K)
    y = exact_f1(xs, ys)
    assert np.isclose(x, y, rtol=5e-1), f"{y} =!= {x}"


def tests():
    print("Testing accuracy...")
    for _ in range(20):
        _test_acc()
    print("ok")
    print("Testing f1...")
    for _ in range(20):
        _test_f1()
    print("ok")

if __name__ == '__main__':
    tests()