import numpy as np
from numba import jit
from numba import int64, float64    # import the types
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
        p = np.zeros(d)
        for k,v in self.p.items():
            p[k] += v
        return p


@jit(nopython=True, nogil=True)
def poly_shift(poly, shift):
    d = Dict.empty(
        key_type=int64,
        value_type=float64,
    )
    for k, v in poly.p.items():
        d[k + shift] = v
    return SparsePolynomial(d)


@jit(nopython=True, nogil=True)
def polymul(p, q):
    # carefully shift and unshift to work around negative exponents
    s = min(min(p.p), min(q.p))
    return poly_shift(_polymul(poly_shift(p, -s), poly_shift(q, -s)), (2 * s))


@jit(nopython=True, nogil=True)
def sparse_from_dense(a):
    d = Dict.empty(
        key_type=int64,
        value_type=float64,
    )
    for k, v in enumerate(a):
        if v != 0:
            d[k] = v
    return SparsePolynomial(d)


@jit(nopython=True, nogil=True)
def _polymul(p, q):
    assert min(p.p) >= 0 and min(q.p) >= 0
    d = 1 + max(max(p.p), max(q.p))
    return sparse_from_dense(np.convolve(p.dense(d), q.dense(d)))


@jit(nopython=True, nogil=True)
def fast_multi_product(ps) -> SparsePolynomial:
    N = len(ps)
    if  N == 0:
        return SparsePolynomial({0: 1})
    elif N == 1:
        return SparsePolynomial(ps[0])
    else:
        return polymul(fast_multi_product(ps[:N//2]), fast_multi_product(ps[N//2:]))


@jit(nopython=True, nogil=True)
def pmf_polymul(xs, ys):
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