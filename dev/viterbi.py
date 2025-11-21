"""
Multi-track tonal extraction over a spectrogram with automatic estimation
of the number of tracks. Uses iterative Viterbi decoding with masking:
1. Run Viterbi to find strongest tonal ridge.
2. Remove (mask) the ridge from the spectrogram.
3. Stop when remaining energy falls below threshold.

Author: ChatGPT
"""

import numpy as np


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

def viterbi_single_track(log_emissions, log_transition, start_state_logprob=None):
    """
    Standard log-domain Viterbi for a single discrete track.
    log_emissions: (T, N)
    log_transition: (N, N)
    start_state_logprob: (N,) or None
    """
    T, N = log_emissions.shape
    dp = np.full((T, N), -np.inf)
    bp = np.zeros((T, N), dtype=np.int32)

    if start_state_logprob is None:
        dp[0] = log_emissions[0]
    else:
        dp[0] = start_state_logprob + log_emissions[0]

    for t in range(1, T):
        prev = dp[t - 1][:, None] + log_transition
        bp[t] = np.argmax(prev, axis=0)
        dp[t] = np.max(prev, axis=0) + log_emissions[t]

    path = np.zeros(T, dtype=np.int32)
    path[-1] = np.argmax(dp[-1])
    for t in reversed(range(1, T)):
        path[t - 1] = bp[t, path[t]]
    return path, dp

def extract_single_track(S, band=5):
    """
    Extract a single track using banded Viterbi.
    S: (T, N) spectrogram magnitude or power
    band: transition bandwidth
    Returns: path (T,), score, log_emissions
    """
    T, N = S.shape
    # log emissions: higher S means higher probability
    log_emissions = np.log(S + 1e-12)

    # banded log transition
    log_transition = np.full((N, N), -np.inf)
    for i in range(N):
        lo = max(0, i - band)
        hi = min(N, i + band + 1)
        log_transition[i, lo:hi] = 0  # uniform in band

    path, dp = viterbi_single_track(log_emissions, log_transition)
    score = dp[-1, path[-1]]
    return path, score, log_emissions

def mask_track(S, path, width=3):
    """
    Zero out energy around the extracted track.
    width: number of bins around track to remove
    """
    T, N = S.shape
    S2 = S.copy()
    for t in range(T):
        lo = max(0, path[t] - width)
        hi = min(N, path[t] + width + 1)
        S2[t, lo:hi] = 0
    return S2

def extract_multiple_tracks(S, band=5, width=3, min_energy_ratio=0.02):
    """
    Iteratively extract tonal tracks until remaining energy is small.

    S: spectrogram (T, N)
    band: Viterbi transition bandwidth
    width: mask width
    min_energy_ratio: stop when remaining energy < ratio of original energy

    Returns: list of (path, score)
    """
    S_work = S.copy().astype(float)
    original_energy = np.sum(S_work)
    tracks = []

    while True:
        current_energy = np.sum(S_work)
        if current_energy < min_energy_ratio * original_energy:
            break

        path, score, _ = extract_single_track(S_work, band=band)

        # If the track has too little energy, stop
        track_energy = np.sum([S_work[t, path[t]] for t in range(S_work.shape[0])])
        if track_energy < 0.001 * original_energy:
            break

        tracks.append((path, score))
        S_work = mask_track(S_work, path, width=width)

    return tracks

if __name__ == "__main__":
    # Example synthetic test
    T = 200
    N = 128
    S = np.random.rand(T, N) * 0.05  # noise

    # Create two synthetic moving tones
    f1 = np.linspace(20, 80, T).astype(int)
    f2 = np.linspace(90, 40, T).astype(int)

    for t in range(T):
        S[t, f1[t]] += 2.0
        S[t, f2[t]] += 1.5

    tracks = extract_multiple_tracks(S, band=4, width=2, min_energy_ratio=0.01)
    print(f"Extracted {len(tracks)} tracks")
    for i, (path, score) in enumerate(tracks):
        print(f"Track {i}: score={score:.2f}, mean bin={np.mean(path):.1f}")
