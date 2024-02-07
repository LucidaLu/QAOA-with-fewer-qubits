from util import *
from qaoa import qaoa_maximize
from labrat import take_snapshot
from loguru import logger

# logger.critical(take_snapshot())

x = logger.add(
    "./logs/coups/{time}.log",
)

def coup_qaoa_maxcut(G, ratio, assume_alpha=None):
    set_random_seed(FIXED_RANDOM_SEED)

    n = G.number_of_nodes()

    c0, c1 = split(G, ratio)
    C0, C1, cross_edges = calc_refactored_graph_info(G, c0, c1)
    n0, n1 = len(c0), len(c1)
    logger.info("n0 = {}, n1 = {}", n0, n1)
    cross_edges = np.array(cross_edges, dtype=np.int16)
    assert len(cross_edges) + len(C0.edges) + len(C1.edges) == len(G.edges)

    inner0_gpu = cuda.device_array(1 << len(c0), dtype=np.int32)
    inner1_gpu = cuda.device_array(1 << len(c1), dtype=np.int32)
    H_C = cuda.device_array(1 << len(c0), dtype=np.int32)

    def compute_inner(a, edges):
        @cuda.jit
        def kernel(a):
            s = cuda.grid(1)
            a[s] = 0
            for x, y in edges:
                a[s] += 1 if s >> x & 1 != s >> y & 1 else -1

        edges = np.array(edges, dtype=np.int32)
        if len(edges) != 0:
            kernel[block1D(len(a))](a)
        else:
            from kernel_functions import assign_constant_kernel

            assign_constant_kernel[block1D(len(a))](a, 0)

    @cuda.jit(device=True)
    def get_cross_gpu(s0, s1):
        if len(cross_edges) == 0:
            return 0
        else:
            ans = numba.int16(0)
            for x, y in cross_edges:
                ans += 1 if s0 >> x & 1 != s1 >> y & 1 else -1
            return ans

    @cuda.jit
    def compute_H_C_kernel(H_C, c0_inner, in_s1, s1):
        s0 = cuda.grid(1)
        H_C[s0] = c0_inner[s0] + in_s1 + get_cross_gpu(s0, s1)

    @cuda.jit
    def compute_H_C_gm_kernel(H_C, c0_inner, in_s1, gm):
        s0 = cuda.grid(1)
        H_C[s0] = c0_inner[s0] + in_s1
        for i in range(n0):
            H_C[s0] += gm[i, s0 >> i & 1]

    from kernel_functions import max_reduce

    logger.info(f"into kernel")
    compute_inner(inner0_gpu, C0.edges)
    logger.info(f"into kernel")
    compute_inner(inner1_gpu, C1.edges)

    inner1 = inner1_gpu.copy_to_host()

    def work_s1(s1):
        gm = np.zeros((len(c0), 2), dtype=np.int32)
        for x, y in cross_edges:
            gm[x, 1 - (s1 >> y & 1)] += 1
            gm[x, s1 >> y & 1] -= 1

        compute_H_C_gm_kernel[block1D(1 << len(c0))](H_C, inner0_gpu, inner1[s1], gm)

        if assume_alpha is not None:
            qe, dstr = qaoa_maximize(len(c0), H_C, assume_alpha=assume_alpha)
        else:
            qe, dstr = qaoa_maximize(len(c0), H_C, callback=lambda e: -e / mxe)
        return qe, (qe + len(G.edges)) / 2

    s1 = 0
    _, pcut = work_s1(s1)
    nrounds = 0
    mem = {s1: pcut}
    while True:
        logger.info(f"round {nrounds}")
        nbs = []
        nrounds += 1
        for i in range(len(c1)):
            ns1 = s1 ^ (1 << i)
            if ns1 not in mem:
                mem[ns1] = work_s1(ns1)[1]
            ncut = mem[ns1]
            nbs.append((ncut, ns1))
        if max(nbs)[0] > pcut:
            pcut, s1 = max(nbs)
        else:
            logger.info(f"local max is found after {nrounds} rounds")
            return pcut


def qaoa_in_qaoa(G, ratio):
    set_random_seed(FIXED_RANDOM_SEED)

    n = G.number_of_nodes()

    l = list(range(n))
    rnd.shuffle(l)
    logger.info(l)

    import math

    c0 = l[: math.ceil(n * ratio)]
    c1 = [x for x in l if x not in c0]
    c0.sort()
    c1.sort()

    C0, C1, cross_edges = calc_refactored_graph_info(G, c0, c1)

    def compute_inner(a, edges):
        @cuda.jit
        def kernel(a):
            s = cuda.grid(1)
            for x, y in edges:
                a[s] += 1 if s >> x & 1 != s >> y & 1 else -1

        edges = np.array(edges, dtype=np.int32)
        if len(edges) != 0:
            kernel[block1D(len(a))](a)

    inner0_gpu = cuda.to_device(np.zeros(1 << len(c0), dtype=np.int32))
    inner1_gpu = cuda.to_device(np.zeros(1 << len(c1), dtype=np.int32))

    compute_inner(inner0_gpu, C0.edges)
    compute_inner(inner1_gpu, C1.edges)

    inner0, inner1 = inner0_gpu.copy_to_host(), inner1_gpu.copy_to_host()

    def merge(m0, m1):
        ans0 = (inner0[m0] + inner1[m1] + len(C0.edges) + len(C1.edges)) / 2
        ans1 = ans0
        for x, y in cross_edges:
            if m0 >> x & 1 != m1 >> y & 1:
                ans0 += 1
            else:
                ans1 += 1
        return max(ans0, ans1)

    mx0, mx1 = np.max(inner0), np.max(inner1)

    v0, d0 = qaoa_maximize(
        len(c0), inner0_gpu, level=1, callback=lambda e: -e / mx0, return_state=True
    )
    v1, d1 = qaoa_maximize(
        len(c1), inner1_gpu, level=1, callback=lambda e: -e / mx1, return_state=True
    )

    logger.info(f"qiq 2 ratios = {v0/mx0}, {v1/mx1}")
    m0, m1 = d0[0][1], d1[0][1]

    ans = 0
    for p0, m0 in d0:
        for p1, m1 in d1:
            ans += p0 * p1 * merge(m0, m1)
    return ans


if __name__ == "__main__":
    n, ratio = 24, 0.75
    logger.info(f"n = {n}, ratio = {ratio}]")

    data = []

    from labrat import data_saver

    ds = data_saver("sim")
    ds.save([])

    cuda.select_device(2)

    for seed in trange(100):
        logger.info(f"seed = {seed}")

        set_random_seed(seed)
        G = nx.erdos_renyi_graph(n=n, p=0.8)
        mxct = get_maxcut(G)

        gw = goemans_williamson(G)
        gw_ex = gw.expectation()

        ans_qiq = qaoa_in_qaoa(G, ratio)

        ans_ls = coup_qaoa_maxcut(G, ratio)

        data.append((gw_ex / mxct, ans_qiq / mxct, ans_qiq / mxct, ans_ls / mxct))
        ds.save(data)
