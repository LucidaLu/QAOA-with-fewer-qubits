from util import *
from labrat import *

logger.add("logs/T_greedy/{time}.log")
logger.critical(take_snapshot())


def partial_T_greedy(beta, G, c0, c1, s0_shots, s0_samples_gpu):
    n0, n1 = len(c0), len(c1)

    C0, C1, cross_edges = calc_refactored_graph_info(G, c0, c1)
    cross_edges = np.array(cross_edges, dtype=np.int16)

    f_gpu = cuda.to_device(np.zeros(s0_shots, dtype=np.int32))

    @cuda.jit(device=True)
    def get_cross_gpu(s0, s1):
        if len(cross_edges) == 0:
            return 0
        else:
            ans = numba.int16(0)
            for x, y in cross_edges:
                ans += 1 if s0 >> x & 1 != s1 >> y & 1 else 0
            return ans

    low_n1 = (1 << n1) - 1

    @cuda.jit
    def compute_f_kernel(f, s0_samples, in1):
        # s0_i, s1 = cuda.grid(2)
        s0_i = cuda.grid(1)
        s1 = s0_i & low_n1
        s0_i >>= n1

        s0 = s0_samples[s0_i]
        cuda.atomic.max(f, s0_i, in1[s1] + get_cross_gpu(s0, s1))

    b1d = block1D(s0_shots)
    b2d = block2D(s0_shots, 1 << n1)
    b2d_1 = block1D(s0_shots << n1)

    in1 = np.zeros(1 << n1, dtype=np.int16)

    for s1 in range(1 << n1):
        in1[s1] = sum(1 if s1 >> x & 1 != s1 >> y & 1 else 0 for x, y in C1.edges)

    in1_gpu = cuda.to_device(in1)

    compute_f_kernel[b2d_1](f_gpu, s0_samples_gpu, in1_gpu)

    covered_gpu = cuda.to_device(np.zeros(s0_shots, dtype=np.int8))  # a bool array

    @cuda.jit
    def compute_coversize_kernel(f, s0_samples, covered, in1, coversize):
        # s0_i, s1 = cuda.grid(2)
        s0_i = cuda.grid(1)
        s1 = s0_i & low_n1
        s0_i >>= n1

        if not covered[s0_i]:
            s0 = s0_samples[s0_i]
            if in1[s1] + get_cross_gpu(s0, s1) >= beta * f[s0_i]:
                cuda.atomic.add(coversize, s1, 1)

    @cuda.jit
    def do_cover_kernel(f, s0_samples, covered, in1_s1, s1):
        s0_i = cuda.grid(1)

        if not covered[s0_i]:
            s0 = s0_samples[s0_i]
            if in1_s1 + get_cross_gpu(s0, s1) >= beta * f[s0_i]:
                covered[s0_i] = True

    T = []

    while True:
        coversize_gpu = cuda.to_device(np.zeros(1 << n1, dtype=np.int32))
        compute_coversize_kernel[b2d_1](
            f_gpu, s0_samples_gpu, covered_gpu, in1_gpu, coversize_gpu
        )
        cuda.synchronize()
        coversize = coversize_gpu.copy_to_host()
        cuda.synchronize()
        mxcover = np.max(coversize)
        best_s1 = np.where(coversize == mxcover)[0][0]
        if mxcover == 0:
            break
        else:
            do_cover_kernel[b1d](
                f_gpu, s0_samples_gpu, covered_gpu, in1[best_s1], best_s1
            )
            T.append(best_s1)

    logger.info(f"found T = {T}, size = {len(T)}")
    return T


def T_greedy_increment(gpu_id, n, split_ratio, beta):
    cuda.select_device(gpu_id)

    import math

    n0 = math.ceil(split_ratio * n)
    logger.info(f"n0 = {n0}, n = {n}, ratio = {split_ratio}, beta = {beta}")

    fname = f"T_greedy-{n}-{n0}-{beta}"
    logger.add("logs/T_greedy/{time}" + fname + ".log")

    ds = data_saver(fname)
    ds.save([])

    all_result = []

    for _ in trange(20):
        logger.info(f"sample #{_}")
        set_random_seed(_)

        MX_SAMPLE = min(n0 - 1, 27)

        if n0 - 1 == MX_SAMPLE:
            s0_samples_gpu = cuda.to_device(np.random.permutation(1 << (n0 - 1)))
            logger.info("taken random shuffle shortcut")
        else:
            s0_samples_gpu = cuda.to_device(
                rnd.sample(range(1 << (n0 - 1)), 1 << MX_SAMPLE)
            )  # fix the color of node n-1
            logger.info("sampled")

        arr = []
        for _ in trange(20):
            logger.info(f"seed #{_}")
            set_random_seed(_)

            G = nx.erdos_renyi_graph(n, 0.8)
            c0, c1 = split(G, split_ratio)

            Ts = []
            for i in range(1, MX_SAMPLE + 1):
                logger.info(f"s0 shots = 2^{i}")
                T = partial_T_greedy(
                    beta, G, c0, c1, s0_shots=1 << i, s0_samples_gpu=s0_samples_gpu
                )
                Ts.append(len(T))

            arr.append(Ts)
            if len(all_result) == 0:
                ds.save([arr])

        all_result.append(arr)
        ds.save(all_result)


if __name__ == "__main__":
    T_greedy_increment(0, 32, 0.5, 0.9)
