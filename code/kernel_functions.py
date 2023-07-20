from numpy import real
from util import *
import cmath


@cuda.jit
def fwht_step_kernel(a, h):
    i = cuda.grid(1)
    x, y = i >> h, i & ((1 << h) - 1)
    x = x << (h + 1) | y
    y = x + (1 << h)
    a[x], a[y] = a[x] + a[y], a[x] - a[y]


@cuda.jit
def multiply_by_constant_kernel(s, k):
    i = cuda.grid(1)
    s[i] *= k


@cuda.jit
def assign_constant_kernel(s, k):
    i = cuda.grid(1)
    s[i] = k


@cuda.jit
def phase_shift_kernel(s, k, p):
    i = cuda.grid(1)
    s[i] = cmath.exp(k * p[i]) * s[i]


@cuda.jit
def diff_phase_shift_kernel(s, k, p):
    i = cuda.grid(1)
    s[i] = -1j * p[i] * cmath.exp(k * p[i]) * s[i]


@cuda.jit
def compute_expectation_kernel(s, h):
    i = cuda.grid(1)
    s[i] = s[i].conjugate() * h[i] * s[i]


@cuda.jit
def compute_expectation_2_kernel(s0, h, s1):
    i = cuda.grid(1)
    s0[i] = s0[i].conjugate() * h[i] * s1[i]


@cuda.jit
def binary_add_kernel(s, w):
    i = cuda.grid(1)
    s[i << (w + 1)] += s[((i << 1) + 1) << w]


@cuda.jit
def element_access_kernel(s, v):
    i = cuda.grid(1)
    v[i] = s[i]


PTHREAD_MAX = 1024


def get_grid(dim):
    PTHREAD = min(PTHREAD_MAX, dim)
    NBLOCK = dim // PTHREAD
    return (NBLOCK, PTHREAD)


def prepare_H_C(G):
    n = G.number_of_nodes()

    cut_all = cuda.device_array(1 << n, dtype=np.int16)

    def compute_cut_all(s, edges):
        @ cuda.jit
        def kernel(a):
            s = cuda.grid(1)
            a[s] = 0
            for x, y in edges:
                a[s] += 1 if s >> x & 1 != s >> y & 1 else -1
                # critical! not ZZ, it's I-ZZ
        edges = np.array(edges, dtype=np.int32)
        kernel[get_grid(1 << n)](s)

    compute_cut_all(cut_all, np.array(G.edges))

    return cut_all


@cuda.jit
def get_probability_kernel(s):
    i = cuda.grid(1)
    s[i] = s[i].conjugate() * s[i]


@cuda.reduce
def max_reduce(a, b):
    return max(a, b)
