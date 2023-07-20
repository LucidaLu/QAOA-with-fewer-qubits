from util import *


def linear_args(p, T):
    samples = (np.arange(1, p + 1) - 0.5) / p
    dt = T / p
    gamma = samples * dt
    beta = (1 - samples) * dt
    return np.concatenate((beta, gamma))
