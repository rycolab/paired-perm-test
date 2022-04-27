import numpy as np
from scipy.signal import convolve

from pairedpermtest.utils import fast_multi_product

class shifted_array:
    "Dense array with bookkeeping for shift"
    def __init__(self, x, s):
        self.x = x               # dense array
        self.s = s               # shift (+/- int)

    def to_sparse(self):
        # unshift
        return {
            (k - self.s): v for k,v in enumerate(self.x) if v != 0
        }

    def __repr__(self):
        return repr(self.to_sparse())

    def __mul__(self, q):
        """
        Multiply polynomial by another polynomial.
        This does not replace self with resulting polynomial.
        :param q: other polynomial
        :return: self * q
        """
        x = convolve(self.x, q.x)
        n = len(x)
        x = np.trim_zeros(x, 'f')
        s = n - len(x)
        return shifted_array(np.trim_zeros(x, 'f'), self.s + q.s - s)

    @classmethod
    def from_sparse(cls, a):
        s = -min(a)   # shift to make smallest element position zero in the array
        d = 1 + (max(a) - min(a))
        x = np.zeros(d)
        for k, v in a.items():
            x[k + s] += v
        return cls(x, s)


def from_sparse(a):
    return shifted_array.from_sparse(a)



def pmf_polymul(xs, ys):
    """
    Compute the exact PMF for the paired permutation test in difference of accuracy
    This method uses fast polynomial multiplication
    Runs in O(LN (log LN) (log N)) time
    :param xs: accuracy scores for system A
    :param ys: accuracy scores for system B
    :return: PMF distribution for paired permutation test
    """
    [N] = xs.shape
    return fast_multi_product([
        from_sparse({
            (xs[n] - ys[n]): 0.5,
            (ys[n] - xs[n]): 0.5,
        })
        for n in range(N)
        if xs[n] - ys[n] != 0
    ])


def perm_test_polymul(xs, ys):
    """
    Run exact paired permutation test on difference in accuracy.
    This method uses fast polynomial multiplication
    Runs in O(LN (log LN) (log N)) time
    :param xs: accuracy scores for system A
    :param ys: accuracy scores for system B
    :return: exact p-value  for paired permutation test
    """
    observed = np.abs(np.sum(xs - ys))
    P = pmf_polymul(xs, ys).to_sparse()
    p = 0.0
    for diff, val in P.items():
        if abs(diff) >= observed:
            p += val
    return p


def tests():
    N = 100
    C = 10

    for i in range(20):
        xs = np.random.randint(0, C, N)
        ys = np.random.randint(0, C, N)

        p1 = perm_test(xs, ys)
        p2 = perm_test_polymul(xs, ys)
        assert np.isclose(p1, p2)
    print("ok")


if __name__ == '__main__':
    from pairedpermtest.accuracy import perm_test
    tests()