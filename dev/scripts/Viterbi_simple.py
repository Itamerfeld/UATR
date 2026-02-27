
import numpy as np
import matplotlib.pyplot as plt

def make_synthetic_spectrogram(n_freq=200, n_time=150, noise_level=0.3):
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    freqs = np.arange(n_freq)
    times = np.arange(n_time)
    ridge = (n_freq/4) + (n_freq/8) * np.sin(2 * np.pi * times / n_time * 3) + 0.5 * np.cos(2 * np.pi * times / (n_time/2))
    ridge = np.clip(ridge, 0, n_freq-1)
    S = noise_level * np.random.rand(n_freq, n_time)
    for t in range(n_time):
        center = ridge[t]
        peak = np.exp(-0.5 * ((freqs - center) / 2.0)**2)
        S[:, t] += peak * 8.0
    S = gaussian_filter1d(S, sigma=1.0, axis=0)
    return S, ridge

def build_transition_matrix(n_states, stay_prob=0.85, transition_sigma=3.0):
    freqs = np.arange(n_states)
    A = np.zeros((n_states, n_states), dtype=float)
    for i in range(n_states):
        diff = freqs - i
        kernel = np.exp(-0.5 * (diff / transition_sigma)**2)
        kernel[i] *= 1.0
        A[i, :] = kernel / kernel.sum()
    A_mixed = (1 - stay_prob) * A + stay_prob * np.eye(n_states)
    A_mixed /= A_mixed.sum(axis=1, keepdims=True)
    return A_mixed

def viterbi_log(emission_logp, A_log, pi_log=None):
    n_states, T = emission_logp.shape
    if pi_log is None:
        pi_log = -np.log(n_states) * np.ones(n_states)
    delta = np.full((n_states, T), -np.inf)
    psi = np.zeros((n_states, T), dtype=int)
    delta[:, 0] = pi_log + emission_logp[:, 0]
    for t in range(1, T):
        scores = delta[:, t-1][:, None] + A_log
        psi[:, t] = np.argmax(scores, axis=0)
        delta[:, t] = scores[psi[:, t], np.arange(n_states)] + emission_logp[:, t]
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[:, -1])
    for t in range(T-2, -1, -1):
        path[t] = psi[path[t+1], t+1]
    logprob = delta[path[-1], -1]
    return path, logprob

def viterbi_banded_log(emission_logp, stay_prob=0.85, transition_sigma=3.0, band=10, pi_log=None):
    n_states, T = emission_logp.shape
    if pi_log is None:
        pi_log = -np.log(n_states) * np.ones(n_states)
    delta = np.full((n_states, T), -np.inf)
    psi = np.zeros((n_states, T), dtype=int)
    delta[:, 0] = pi_log + emission_logp[:, 0]
    freqs = np.arange(n_states)
    for t in range(1, T):
        for j in range(n_states):
            i_min = max(0, j - band)
            i_max = min(n_states-1, j + band)
            i_range = np.arange(i_min, i_max+1)
            diff = freqs[j] - i_range
            kernel = np.exp(-0.5 * (diff / transition_sigma)**2)
            weights = (1 - stay_prob) * kernel
            if j >= i_min and j <= i_max:
                weights[i_range == j] += stay_prob
            weights /= weights.sum()
            scores = delta[i_range, t-1] + np.log(weights)
            psi[j, t] = i_range[np.argmax(scores)]
            delta[j, t] = np.max(scores) + emission_logp[j, t]
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[:, -1])
    for t in range(T-2, -1, -1):
        path[t] = psi[path[t+1], t+1]
    logprob = delta[path[-1], -1]
    return path, logprob
