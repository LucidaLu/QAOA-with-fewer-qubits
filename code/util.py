from math import ceil
from tqdm import trange, tqdm
# from matplotlib import pyplot as plt
from loguru import logger
import networkx as nx
import numpy as np
import cvxpy as cvx
import random as rnd
from numba import cuda
# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool
from scipy.optimize import fmin_bfgs as BFGS
import numba
from numba import cuda
from scipy import optimize as spo

from numba.core.errors import NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

import time

FIXED_RANDOM_SEED = 233


def time_it(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger.info('%r %2.2f sec' % (method.__name__, te - ts))
        return (result, te - ts)
    return timed


def set_random_seed(seed):
    rnd.seed(seed)
    np.random.seed(seed)


def brute_force(G):
    n = G.number_of_nodes()

    mx = 0
    for s in tqdm(range(1 << (n - 1))):
        cnt = 0
        for x, y in G.edges:
            if (s >> x & 1) != (s >> y & 1):
                cnt += 1
        mx = max(mx, cnt)

    return mx


class goemans_williamson:
    def __init__(self, graph: nx.Graph):
        """
        The Goemans-Williamson algorithm for solving the maxcut problem.
        Ref:
            Goemans, M.X. and Williamson, D.P., 1995. Improved approximation
            algorithms for maximum cut and satisfiability problems using
            semidefinite programming. Journal of the ACM (JACM), 42(6), 1115-1145
        Returns:
            np.ndarray: Graph coloring (+/-1 for each node)
            float:      The GW score for this cut.
            float:      The GW bound from the SDP relaxation
        """
        # Kudos: Originally implementation by Nick Rubin, with refactoring and
        # cleanup by Jonathon Ward and Gavin E. Crooks
        set_random_seed(FIXED_RANDOM_SEED)
        self.laplacian = np.array(0.25 * nx.laplacian_matrix(graph).todense())

        # Setup and solve the GW semidefinite programming problem
        psd_mat = cvx.Variable(self.laplacian.shape, PSD=True)
        obj = cvx.Maximize(cvx.trace(self.laplacian @ psd_mat))
        constraints = [cvx.diag(psd_mat) == 1]  # unit norm

        prob = cvx.Problem(obj, constraints)
        prob.solve(solver=cvx.MOSEK)

        evals, evects = np.linalg.eigh(psd_mat.value)
        self.sdp_vectors = evects.T[evals > float(1.0E-6)].T

        # logger.info(self.sdp_vectors)
        # Bound from the SDP relaxation
        self.bound = np.trace(self.laplacian @ psd_mat.value)

    def rand(self):
        random_vector = np.random.randn(self.sdp_vectors.shape[1])
        random_vector /= np.linalg.norm(random_vector)
        colors = np.sign([vec @ random_vector for vec in self.sdp_vectors])
        score = colors @ self.laplacian @ colors.T
        colors = ''.join('0' if c == -1 else '1' for c in colors)
        return colors, score

    def expectation(self, shots=1 << 18):
        set_random_seed(FIXED_RANDOM_SEED)
        return np.average([self.rand()[1] for _ in range(shots)])


def block1D(dim):
    PTHREAD_MAX = 1024
    PTHREAD = min(PTHREAD_MAX, dim)
    NBLOCK = dim // PTHREAD
    return (NBLOCK, PTHREAD)


import copy


def refactor_label(c):
    return nx.relabel_nodes(c, lambda x: sorted(c).index(x))


def convert_to_adj(G):
    adj = [[] for _ in range(G.number_of_nodes())]

    for x, y in G.edges:
        adj[x].append(y)
        adj[y].append(x)

    return adj


def split(G: nx.Graph, ratio):
    n = G.number_of_nodes()
    n0 = ceil(ratio * n)

    def expand_policy():
        def expand(x):
            adj = convert_to_adj(G)

            free = list(range(n))

            cluster = [x]
            free.remove(x)

            cnt = [-1] * n
            for y in range(n):
                if y in free:
                    if y in adj[x]:
                        cnt[y] = 1
                    else:
                        cnt[y] = 0

            while len(cluster) < n * ratio and len(free):
                e = cnt.index(max(cnt))
                cluster.append(e)

                free.remove(e)
                cnt[e] = -1

                for t in adj[e]:
                    if t in free:
                        cnt[t] += 1
            return cluster

        mxd, c0 = 0, None

        for i in range(n):
            c = expand(i)
            d = len(G.subgraph(c).edges)
            if d > mxd:
                mxd, c0 = d, c
        c0.sort()
        return c0

    def expand_edge_policy():
        def expand(x0, x1):
            adj = convert_to_adj(G)

            free = list(range(n))

            cluster = [x0, x1]
            free.remove(x0)
            free.remove(x1)
            cnt = [0] * n
            for y in adj[x0]:
                cnt[y] += 1
            for y in adj[x1]:
                cnt[y] += 1
            cnt[x0] = cnt[x1] = -1

            while len(cluster) < n0 and len(free):
                e = cnt.index(max(cnt))
                cluster.append(e)

                free.remove(e)
                cnt[e] = -1

                for t in adj[e]:
                    if t in free:
                        cnt[t] += 1
            return cluster

        mxd, c0 = 0, None

        for x, y in G.edges:
            c = expand(x, y)
            d = len(G.subgraph(c).edges)
            if d > mxd:
                mxd, c0 = d, c
        c0.sort()
        return c0

    def remove_policy():
        c0 = set(range(n))
        adj = convert_to_adj(G)
        cnt = [0] * n
        for x, y in G.edges:
            cnt[x] += 1
            cnt[y] += 1

        while len(c0) > n0:
            x = cnt.index(min(cnt))
            c0.remove(x)
            cnt[x] = n + 1
            for y in adj[x]:
                if y in c0:
                    cnt[y] -= 1

        return list(c0)

    # e, r = expand_edge_policy(), remove_policy()
    # x = len(G.subgraph(r).edges) - len(G.subgraph(e).edges)
    # if x != 0:
    #     print(x)
    c0 = expand_edge_policy()
    assert len(G.subgraph(c0).edges) - (n0 * (n0 - 1)) / (n * (n - 1)) * len(G.edges) >= 0
    return c0, [i for i in range(n) if i not in c0]


def calc_refactored_graph_info(G, c0, c1):
    refactored_crossing_edges = []
    for x, y in G.edges:
        if x in c0 and y in c1:
            refactored_crossing_edges.append((c0.index(x), c1.index(y)))
        elif x in c1 and y in c0:
            refactored_crossing_edges.append((c0.index(y), c1.index(x)))
    C0, C1 = refactor_label(G.subgraph(c0)), refactor_label(G.subgraph(c1))

    assert len(C0.edges) + len(C1.edges) + len(refactored_crossing_edges) == len(G.edges)
    for x, y in refactored_crossing_edges:
        assert ((c0[x], c1[y]) in G.edges)  # ((c1[y], c0[x]) in G.edges)
    return C0, C1, refactored_crossing_edges


def split_check(G: nx.Graph, ratio, heuristic=False):
    n = G.number_of_nodes()
    n0 = ceil(ratio * n)

    if not heuristic:
        E = np.array(G.edges)

        @cuda.jit()
        def find_dense_kernel(mx):
            s = cuda.grid(1)
            c = 0
            for i in range(n):
                if s >> i & 1 == 1:
                    c += 1
            if c == n0:
                dens = 0
                for x, y in E:
                    if s >> x & 1 and s >> y & 1:
                        dens += 1

                cuda.atomic.max(mx, 0, dens)

        mx = np.array([0])
        find_dense_kernel[block1D(1 << n)](mx)
        return mx[0]
    else:
        def expand(x):
            adj = [[] for _ in range(n)]

            for x, y in G.edges:
                adj[x].append(y)
                adj[y].append(x)

            free = list(range(n))

            cluster = [x]
            free.remove(x)

            cnt = [-1] * n
            for y in range(n):
                if y in free:
                    if y in adj[x]:
                        cnt[y] = 1
                    else:
                        cnt[y] = 0

            while len(cluster) < n * ratio and len(free):
                e = cnt.index(max(cnt))
                cluster.append(e)

                free.remove(e)
                cnt[e] = -1

                for t in adj[e]:
                    if t in free:
                        cnt[t] += 1
            return cluster

        density, c0 = 0, None

        for i in range(n):
            c = expand(i)
            d = len(G.subgraph(c).edges)
            if d > density:
                density, c0 = d, c

    c1 = [i for i in range(n) if i not in c0]
    return c0, c1


def hamming_dist(a, b):
    ans = 0
    while a != 0 or b != 0:
        ans += 1 if a & 1 != b & 1 else 0
        a //= 2
        b //= 2
    return ans


PTHREAD_MAX = 1024


def block1D(dim):
    PTHREAD = min(PTHREAD_MAX, dim)
    NBLOCK = dim // PTHREAD
    return NBLOCK, PTHREAD


def block2D(dimx, dimy):
    TX = min(PTHREAD_MAX, dimx)
    BX = dimx // TX
    TY = min(PTHREAD_MAX // TX, dimy)
    BY = dimy // TY
    return (BX, BY), (TX, TY)


def get_maxcut(G):
    n = G.number_of_nodes()
    edges = np.array(G.edges)

    @cuda.jit()
    def maxcut_kernel(mx):
        s = cuda.grid(1)
        cut = 0
        for x, y in edges:
            if s >> x & 1 != s >> y & 1:
                cut += 1
        cuda.atomic.max(mx, 0, cut)

    mx = np.array([0])
    maxcut_kernel[block1D(1 << n)](mx)
    return mx[0]


def sample_on_distribution(p):
    x = np.random.random()
    for i in range(len(p)):
        if p[i].real >= x:
            return i
        else:
            x -= p[i].real
    print('err', x)
