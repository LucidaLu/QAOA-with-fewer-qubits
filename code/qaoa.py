from util import *
from init_policy import *
from kernel_functions import *


def qaoa_maximize(n, H_C, callback=None, level=0):
    """
    compute the highest energy of H_C
    and in qaoa this is done by computing the ground energy of -H_C
    params = (beta, gamma)
    """
    assert cuda.is_cuda_array(H_C)

    if level == 0:
        p = n
    else:
        p = 2 * n

    def compute_x_sum(s):
        @cuda.jit
        def compute_x_sum_kernel(s):
            x = cuda.grid(1)
            s[x] = 0
            for i in range(n):
                s[x] += -1 if x >> i & 1 else 1

        compute_x_sum_kernel[get_grid(1 << n)](s)

    def phase_shift(s, k, p):
        phase_shift_kernel[get_grid(1 << n)](s, k, p)

    def diff_phase_shift(s, k, p):
        diff_phase_shift_kernel[get_grid(1 << n)](s, k, p)

    def multiply_by_constant(s, k):
        multiply_by_constant_kernel[get_grid(1 << n)](s, k)

    def assign_constant(s, k):
        assign_constant_kernel[get_grid(1 << n)](s, k)

    def compute_expectation(s, h):
        compute_expectation_kernel[get_grid(1 << n)](s, h)

    def compute_expectation_2(s0, h, s1):
        compute_expectation_2_kernel[get_grid(1 << n)](s0, h, s1)

    def apply_dc_sum(s):
        for i in range(n):
            t = min(PTHREAD_MAX, 1 << (n - i - 1))
            b = (1 << (n - i - 1)) // t
            binary_add_kernel[b, t](s, i)

    def fwht(a):
        for h in range(n):
            fwht_step_kernel[get_grid(1 << (n - 1))](a, h)

    logger.info("inside QAOA")

    H_B_computational_basis = cuda.device_array(2**n, dtype=np.int8)
    compute_x_sum(H_B_computational_basis)

    state = cuda.device_array(2**n, dtype=np.complex128)

    def compute_qaoa_state(state, x):
        # putting kernel function inside results in repeated compiling
        assign_constant(state, 1 / 2 ** (n / 2))
        p = len(x) // 2
        for i in range(p):
            b, g = x[i], x[i + p]
            phase_shift(state, -1j * g, H_C)
            fwht(state)
            phase_shift(state, -1j * b, H_B_computational_basis)
            fwht(state)
            multiply_by_constant(state, 1.0 / len(state))

    def variational_expectation(x):
        compute_qaoa_state(state, x)
        compute_expectation(state, H_C)
        apply_dc_sum(state)
        return -np.real(state[0])  # H_C isn't minus-ed, so we add - here

    def optm_trial(T, arg_policy, maxiter=100):
        step_cnt = 0

        def print_step(x):
            nonlocal step_cnt
            if step_cnt % 10 == 0:
                logger.info(f"#{step_cnt}: {callback(variational_expectation(x))}")
            step_cnt += 1

        res = BFGS(
            variational_expectation,
            arg_policy(p, T),
            full_output=1,
            disp=0,
            callback=print_step,
            maxiter=maxiter,
        )

        return (res[1], res[0])

    if level == 0:
        v, x = optm_trial(0.56 * p, linear_args)
    else:
        v, x = optm_trial(0.56 * p, linear_args, maxiter=500)

    compute_qaoa_state(state, x)
    get_probability_kernel[get_grid(1 << n)](state)
    dstr = state.copy_to_host()
    return (-v, sorted([(dstr[x].real, x) for x in range(len(dstr))], reverse=True))
