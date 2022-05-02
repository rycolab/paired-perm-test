import numpy as np
from scipy.signal import convolve
from collections import Counter, namedtuple

from pairedpermtest.utils import fast_multi_product, exact_f1


class state(namedtuple('state', 'x,y')):
    def __add__(self, other):
        return state(self.x+other.x, self.y+other.y)

    def f1(self):     # difference in F1 scores
        return self.x.f1() - self.y.f1()


class f1state(namedtuple('state', 'tp,i')):
    def __add__(self, other):
        return f1state(self.tp+other.tp, self.i+other.i)

    def f1(self):
        if self.tp + self.i == 0: return 0.0
        return self.tp / (self.tp + 0.5 * self.i)


def pmf(xs, ys):
    N = len(xs)
    ps = [
        load_dense([
            (state(xs[n], ys[n]), 0.5),   # Unswapped
            (state(ys[n], xs[n]), 0.5),   # Swapped
        ])
        for n in range(N)
    ]
    return fast_multi_product(ps)


def f1(xs):
    N = len(xs)
    tp = sum(xs[n].tp for n in range(N))
    i = sum(xs[n].i for n in range(N))
    if tp + i == 0: return 0.0
    return tp / (tp + 0.5 * i)


def test_statistic(xs, ys):
    return np.abs(f1(xs) - f1(ys))


def h(s):
    (i,j),(k,l) = s
    return state(f1state(i,j), f1state(k,l)).f1()


def perm_test(xs, ys):
    observed = test_statistic(xs, ys)
    P = pmf(xs, ys).pmf()

    p = sum(p for z, p in P.items() if abs(z) >= observed)
    return p


TOL = 1e-6

class M:
    def __init__(self, x):
        self.x = x
    def __repr__(self): return repr(self.x)
    def __iter__(self): return iter(self.x.items())
    def items(self):    return self.x.items()
    def __mul__(self, other):
        zs = Counter()
        for x,xv in self:
            for y,yv in other:
                zs[x + y] += xv * yv
        return M(zs)

    def to_dense(self):
        ds = list(zip(*[tuple(flatten(x)) for x,v in self.x.items()
                        if v > TOL]))
        dims = tuple(max(d) for d in ds)

        x = np.zeros(tuple(np.array(list(dims))+1))
        for (k,v) in self.x.items():
            k = tuple(flatten(k))
            x[k] = v

        return D(x, np.array([0,0,0,0]))

    def pmf(self):
        pmf = Counter()
        for s, v in self.items():
            z = s.f1()
            pmf[z] += v
        return pmf


class D:
    def __init__(self, x, lo):
        assert isinstance(x, np.ndarray), type(x)
        self.x = x
        self.lo = lo

    def __mul__(self, other):
        if self.x.ndim == 0: return other
        if other.x.ndim == 0: return self
        return D(convolve(self.x, other.x), self.lo+other.lo).trim()


    def pmf(self):
        pmf = Counter()
        I,J,K,L = self.lo
        for [i,j,k,l] in zip(*np.where(self.x > TOL)):
            v = self.x[i,j,k,l]
            z = F1(i+I, j+J, k+K,l+L)
            pmf[z] += v
        return pmf

    def trim(self):

        ds = np.where(self.x > TOL)
        if len(ds[0]) == 0: return D(np.zeros((0,0,0,0)), self.lo)

        I,J,K,L = tuple(d.min() for d in ds)
        i,j,k,l = tuple(d.max() for d in ds)

        return D(self.x[I:i+1,J:j+1,K:k+1,L:l+1], self.lo + np.array([I,J,K,L]))


def f(tp, i):
    if tp + i == 0: return 0.0
    return tp / (tp + 0.5 * i)


def F1(i,j,k,l):
    return f(i,j) - f(k,l)


def flatten(xs):
    if isinstance(xs, (tuple, list)):
        for x in xs:
            yield from flatten(x)
    else:
        yield xs


def load_dense(x):
    y = Counter()
    for s,v in x:
        y[s] += v
    return M(y).to_dense()


def _test_f1(N):
    xs = np.random.randint(0, 5, (N, 2))
    ys = np.random.randint(0, 5, (N, 2))
    xs_ = [f1state(xs[n, 0], xs[n, 1]) for n in range(N)]
    ys_ = [f1state(ys[n, 0], ys[n, 1]) for n in range(N)]
    x = perm_test(xs_, ys_)
    y = exact_f1(xs, ys)
    assert np.isclose(x, y, rtol=0.001), f"{x} != {y}"


def tests():
    for N in range(4, 10):
        for _ in range(20):
            _test_f1(N)
    print('ok')


if __name__ == '__main__':
    tests()
