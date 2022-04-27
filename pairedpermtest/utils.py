from itertools import product
import numpy as np
from numba import jit


def fast_multi_product(ps):
    """
    Perform fast recursive multiplication on a list of items.
    :param ps: list of objects
    :return: product of objects
    """
    N = len(ps)
    if N == 1:
        return ps[0]
    else:
        return fast_multi_product(ps[:N//2]) * fast_multi_product(ps[N//2:])


def exact_acc(xs, ys, statistic=np.mean):
    """
    Compute the exact p-value for the paired permutation test for accuracy.
    This uses a brute force method.
    WARNING: this is slow and should only be used for testing purposes

    :param xs: accuracy scores for system A
    :param ys: accuracy scores for system B
    :param statistic: accumulation statistic for accuracy (default mean)
    :return: exact p-value on the paired permutation test between xs and ys
    """
    def effect(xs,ys): return np.abs(statistic(xs) - statistic(ys))
    observed = effect(xs, ys)
    p = 0.0
    n = len(xs)
    pe = 2 ** -n
    for swaps in product(*([0,1] for _ in range(n))):
        swaps = np.array(swaps, dtype=bool)
        E = effect(np.select([swaps,~swaps],[xs,ys]),  # swap elements accordingly
                   np.select([~swaps,swaps],[xs,ys]))
        p += pe * (E >= observed)
    return p


@jit(nopython=True, nogil=True, cache=True)
def f1(xs):
    """
    Compute the F1 score of a system.
    :param xs: [N, 2] array containing the number of true positives of each
    prediction in xs[:, 0] and number of incorrect predictions in xs[:, 1]
    :return: F1 score of xs
    """
    tp = np.sum(xs[:, 0])
    incorrect = np.sum(xs[:, 1])
    if tp + incorrect == 0:
        return 0.
    return tp / (tp + 0.5 * incorrect)


def exact_f1(xs, ys):
    """
    Compute the exact p-value for the paired permutation test for F!.
    This uses a brute force method.
    WARNING: this is slow and should only be used for testing purposes

    Both xs and ys are [N, 2] arrays containing the number of true positives of each
    prediction in xs[:, 0] (or ys[:, 0] and number of incorrect predictions in xs[:, 1] (or ys[:, 1]

    :param xs: accuracy scores for system A
    :param ys: accuracy scores for system B
    :return: exact p-value on the paired permutation test between xs and ys
    """
    def effect(xs, ys): return np.abs(f1(xs) - f1(ys))
    observed = effect(xs, ys)
    p = 0.0
    n = len(xs)
    pe = 2 ** -n
    for swaps in product(*([0, 1] for _ in range(n))):
        swaps = np.array(swaps, dtype=bool).repeat(2).reshape(n, 2)
        E = effect(
            np.where(swaps, xs, ys),  # swap elements accordingly
            np.where(~swaps, xs, ys),
        )
        p += pe * (E >= observed)
    return p
