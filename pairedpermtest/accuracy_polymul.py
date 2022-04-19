import numpy as np
from numba import jit
from numba import int64, float64 
from numba.experimental import jitclass
from numba.types import DictType
from numba.typed import Dict

from pairedpermtest.accuracy import perm_test


spec = [
    ('p', DictType(int64, float64))
]


@jitclass(spec)
class SparsePolynomial:
    """
    Polynomial in one variable with possibly negative exponents
    """

    def __init__(self, p):
        self.p = p

    def __repr__(self):
        return repr(dict(self.p))

    def __iter__(self):
        return iter(self.p)

    def __getitem__(self, k):
        return self.p[k]

    def dense(self, d):
        """
        Represent sparse polynomial as a dense array.
        Expected that all exponents are non-negative
        :param d: size of array (largest exponent)
        :return: dense array of polynomial
        """
        p = np.zeros(d)
        for k,v in self.p.items():
            p[k] += v
        return p

    def shift(self, k_shift):
        """
        Shift all exponents of a polynomial by k_shift
        :param k_shift: integral value to shift by
        :return: shifted polynomial
        """
        d = Dict.empty(
            key_type=int64,
            value_type=float64,
        )
        for k, v in self.p.items():
            d[k + k_shift] = v
        return SparsePolynomial(d)

    def mul(self, q):
        """
        Multiply polynomial by another polynomial.
        This does not replace self with resulting polynomial.
        :param q: other polynomial
        :return: self * q
        """
        s = min(min(self.p), min(q.p))
        d = 1 + max(max(self.p), max(q.p)) - s
        p1 = self.shift(-s).dense(d)
        p2 = q.shift(-s).dense(d)
        return sparse_from_dense(np.convolve(p1, p2)).shift(2 * s)


@jit(nopython=True, nogil=True)
def sparse_from_dense(a):
    """
    Convert a dense array representation of a polynomial into a SparsePolynomial.
    It is assumed a[i] is the coefficient for exponent i.
    :param a: dense array
    :return: SparsePolynomial of array a
    """
    d = Dict.empty(
        key_type=int64,
        value_type=float64,
    )
    for k, v in enumerate(a):
        if v != 0:
            d[k] = v
    return SparsePolynomial(d)


@jit(nopython=True, nogil=True)
def fast_multi_product(ps) -> SparsePolynomial:
    """
    Perform fast recursive multiplication on a list of polynomials.
    If M is the maximum degree of a polynomial, then runs in: O(M (log M) (log N)) time.
    :param ps: list of polynomials
    :return: product of polynomials
    """
    N = len(ps)
    if  N == 0:
        return SparsePolynomial({0: 1})
    elif N == 1:
        return SparsePolynomial(ps[0])
    else:
        return fast_multi_product(ps[:N//2]).mul(fast_multi_product(ps[N//2:]))


@jit(nopython=True, nogil=True)
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
    assert xs.shape == ys.shape
    polys = []
    for n in range(N):
        if xs[n] - ys[n] != 0:
            x = {
                xs[n] - ys[n]: 0.5,
                ys[n] - xs[n]: 0.5
            }
            polys.append(x)
    return fast_multi_product(polys)


@jit(nopython=True, nogil=True)
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
    P: Dict[int, float] = pmf_polymul(xs, ys).p
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
    tests()