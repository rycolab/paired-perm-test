from itertools import product
import numpy as np
from numba import jit


def exact_acc(xs, ys, statistic=np.mean):
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
    tp = np.sum(xs[:, 0])
    incorrect = np.sum(xs[:, 1])
    if tp + incorrect == 0:
        return 0.
    return tp / (tp + 0.5 * incorrect)


def exact_f1(xs, ys):
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
