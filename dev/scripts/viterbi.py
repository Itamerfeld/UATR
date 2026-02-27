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

"""
Multi-track spectrogram ridge extraction with explicit GAP state.

Key ideas:
- States: frequency bins f=0..N-1 and one GAP state at index N.
- The GAP state has no emission (flat) allowing paths to skip (track absent).
- Banded transitions between frequencies encourage local moves.
- Transition probabilities include transitions to/from GAP (birth/death/gap continuation).
- Iterative extraction: extract strongest track with Viterbi, mask it (soft or hard),
  then extract next, using a repulsion penalty derived from already-extracted tracks.

Author: ChatGPT (provided to user)
"""

# -----------------------
# Utility: stable log-sum / logmax
# -----------------------
def logsumexp(a):
    a_max = np.max(a)
    return a_max + np.log(np.sum(np.exp(a - a_max)))

# -----------------------
# Viterbi with GAP state (log-domain)
# -----------------------
def viterbi_with_gap(log_emissions_TN, band=6,
                     trans_sigma=2.0,
                     log_p_stay=0.0,
                     log_p_to_gap=-2.0,
                     log_p_from_gap=-2.0,
                     log_p_gap_stay=-0.5):
    """
    Run Viterbi over states {f=0..N-1, GAP=N} for a single track.
    Inputs:
      log_emissions_TN : np.array shape (T, N)  (log emission for each frequency bin)
                         Note: GAP has no emission; it will be handled separately.
      band : max frequency jump (half-width) allowed (int)
      trans_sigma : gaussian std (in bins) for frequency-to-frequency transition weights
      log_p_stay : extra log-weight favoring staying in the same frequency bin (added to self-transition)
      log_p_to_gap : log-probability weight for transitions from any freq -> GAP
      log_p_from_gap : log-probability weight for transitions from GAP -> any freq (birth)
      log_p_gap_stay : log-prob of GAP -> GAP
    Returns:
      path : np.array shape (T,) values in 0..N (N is GAP index)
      best_score : log-score (dp[T-1, path[-1]])
      dp : full dp matrix shape (T, N+1) if you want to inspect
    """

    T, N = log_emissions_TN.shape
    GAP = N
    dp = np.full((T, N + 1), -np.inf)
    bp = np.zeros((T, N + 1), dtype=np.int32)

    # Precompute gaussian kernel values for transitions between freq bins
    freqs = np.arange(N)
    # For each source i, we will consider destinations j in [i-band, i+band].
    # But easier: for each dest j, consider src i in [j-band, j+band].
    # We'll compute relative weights as needed.

    # Initialization (t=0)
    # Can either allow starting in GAP or in any freq (with birth weight).
    # We give GAP a small baseline score (0) and allow GAP->freq via log_p_from_gap later.
    dp[0, :N] = log_emissions_TN[0]  # start at any frequency with its emission
    dp[0, GAP] = 0.0  # starting in gap (no emission cost)

    # forward dynamic programming
    for t in range(1, T):
        # compute transitions to each destination state s
        # destination = GAP
        # prev could be GAP or any freq
        # dp_prev + log transition prev->GAP
        # choose best prev for destination GAP
        # transitions:
        #   if prev = GAP: add log_p_gap_stay
        #   if prev = freq: add log_p_to_gap
        prev_vals = dp[t - 1, :N] + log_p_to_gap   # prev freq -> GAP
        prev_gap_val = dp[t - 1, GAP] + log_p_gap_stay  # prev gap -> gap
        # choose best previous state
        if prev_gap_val >= np.max(prev_vals):
            dp[t, GAP] = prev_gap_val
            bp[t, GAP] = GAP
        else:
            best_prev_idx = np.argmax(prev_vals)
            dp[t, GAP] = prev_vals[best_prev_idx]
            bp[t, GAP] = best_prev_idx

        # destinations are frequency bins
        for j in range(N):
            # candidate transitions:
            # 1) from previous freq i in [j-band, j+band] with gaussian weights
            i_min = max(0, j - band)
            i_max = min(N - 1, j + band)
            i_range = np.arange(i_min, i_max + 1)

            # compute transition log-weights up to additive constant:
            diffs = i_range - j  # src - dest
            # gaussian shape (higher for small diff)
            trans_weights = np.exp(-0.5 * (diffs / trans_sigma) ** 2)
            # convert to log and normalize so they form a distribution over i_range
            trans_log = np.log(trans_weights + 1e-300)
            trans_log = trans_log - logsumexp(trans_log)  # normalized over the i_range

            # optionally add extra bias for staying in same frequency
            # we handle that by boosting the term where i_range == j
            if (j >= i_min) and (j <= i_max):
                idx_same = j - i_min
                trans_log[idx_same] += log_p_stay

            # Now compute candidates from prev freq states:
            prev_candidates = dp[t - 1, i_range] + trans_log

            best_prev_local_idx = np.argmax(prev_candidates)
            best_prev_local_state = i_range[best_prev_local_idx]
            best_prev_local_val = prev_candidates[best_prev_local_idx]

            # 2) from previous GAP -> j (birth), weight = log_p_from_gap
            gap_candidate = dp[t - 1, GAP] + log_p_from_gap

            # choose best of these two
            if gap_candidate >= best_prev_local_val:
                dp[t, j] = gap_candidate + log_emissions_TN[t, j]
                bp[t, j] = GAP
            else:
                dp[t, j] = best_prev_local_val + log_emissions_TN[t, j]
                bp[t, j] = best_prev_local_state

    # backtrack
    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = int(np.argmax(dp[T - 1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = int(bp[t, path[t]])

    best_score = dp[T - 1, path[T - 1]]
    return path, best_score, dp

# -----------------------
# Iterative multi-track extraction with repulsion and GAP support
# -----------------------
def extract_tracks_with_gap(S_TN, n_max_tracks=10,
                            band=6,
                            trans_sigma=2.0,
                            mask_width=3,
                            repulsion_strength=3.0,
                            min_track_energy_ratio=1e-3,
                            **viterbi_kwargs):
    """
    Iteratively extract up to n_max_tracks from spectrogram S (shape T x N).
    Each extraction:
      - compute log_emissions from current residual
      - subtract repulsion penalty derived from already extracted tracks
      - run viterbi_with_gap to get a path (which may include GAP states)
      - compute energy along path (only frequency bins)
      - if energy is strong enough, accept and mask (soft-zero) around it
      - repeat until stopping criterion is met
    Returns:
      tracks : list of dicts with keys:
         'path' : array shape (T,) values in 0..N (N means GAP)
         'start' : first frame index where state != GAP (or None)
         'end'   : last frame index where state != GAP (or None)
         'energy': scalar energy along the track
         'score' : viterbi log-score
    """

    T, N = S_TN.shape
    GAP = N
    S_work = S_TN.copy().astype(float)
    total_energy = np.sum(S_work) + 1e-300

    tracks = []

    def compute_repulsion_penalty(N, existing_paths, t):
        # returns a vector length N (penalty to subtract from log-emission at time t)
        if len(existing_paths) == 0:
            return np.zeros(N)
        penalty = np.zeros(N)
        for p in existing_paths:
            idx = p['path'][t]
            if idx == GAP:
                continue
            # gaussian penalty centered at idx
            bins = np.arange(N)
            penalty += np.exp(-0.5 * ((bins - idx) / 1.5) ** 2)
        # scale penalty so it's on log-prob scale: larger penalty = lower log-emission
        penalty = penalty / (np.max(penalty) + 1e-300)
        return repulsion_strength * penalty

    for k in range(n_max_tracks):
        # build log_emissions (T x N) from S_work and subtract repulsion penalty
        # emission = log(S + eps) ; we'll subtract penalty (positive) from log-emission
        eps = 1e-12
        log_emissions = np.log(S_work + eps)  # shape (T, N)

        # subtract repulsion: for each t, compute penalty and subtract (in additive log domain)
        if len(tracks) > 0:
            existing = tracks
            for t in range(T):
                penalty = compute_repulsion_penalty(N, existing, t)
                log_emissions[t, :] = log_emissions[t, :] - penalty

        # run Viterbi with GAP
        path, score, dp = viterbi_with_gap(log_emissions, band=band,
                                           trans_sigma=trans_sigma, **viterbi_kwargs)

        # compute track energy (sum of S_work over time only at freq bins where path != GAP)
        energy = 0.0
        freq_bins = []
        for t in range(T):
            sidx = path[t]
            if sidx != GAP:
                energy += S_work[t, sidx]
                freq_bins.append(sidx)
        mean_energy = energy / (len(freq_bins) + 1e-300)

        # stopping rule: if energy too small relative to total energy -> stop
        if energy < min_track_energy_ratio * total_energy:
            break

        # record track info (compute start/end indices where path != GAP)
        active_indices = np.where(path != GAP)[0]
        if active_indices.size == 0:
            # track is entirely gap: reject and stop
            break
        start = int(active_indices[0])
        end = int(active_indices[-1])

        tracks.append({
            'path': path,
            'start': start,
            'end': end,
            'energy': float(energy),
            'score': float(score)
        })

        # mask (suppress) the extracted track from S_work (soft masking)
        # set energy around the path to a fraction (e.g., multiply by 0.05)
        for t in range(T):
            sidx = path[t]
            if sidx == GAP:
                continue
            lo = max(0, sidx - mask_width)
            hi = min(N, sidx + mask_width + 1)
            # reduce but not zero-out to avoid numerical issues
            S_work[t, lo:hi] *= 0.05

    return tracks

# -----------------------
# Example usage (synthetic test)
# -----------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # synthesize a spectrogram with two tones that start/end in the middle
    T = 300
    N = 160
    S = np.random.rand(T, N) * 0.01  # background noise

    # track A: present from t=30..220
    tA_start, tA_end = 30, 220
    fA = (40 + 10 * np.sin(np.linspace(0, 10, tA_end - tA_start))).astype(int)
    for i, t in enumerate(range(tA_start, tA_end)):
        S[t, fA[i]] += 2.5
        # small spread
        if fA[i] + 1 < N:
            S[t, fA[i] + 1] += 0.8

    # track B: present from t=120..280 (overlaps with A between 120..220)
    tB_start, tB_end = 120, 280
    fB = (110 - 20 * np.cos(np.linspace(0, 6, tB_end - tB_start))).astype(int)
    for i, t in enumerate(range(tB_start, tB_end)):
        S[t, fB[i]] += 1.8
        if fB[i] - 1 >= 0:
            S[t, fB[i] - 1] += 0.5

    # run extraction
    tracks = extract_tracks_with_gap(S, n_max_tracks=6,
                                     band=5,
                                     trans_sigma=2.0,
                                     mask_width=2,
                                     repulsion_strength=2.5,
                                     min_track_energy_ratio=1e-4,
                                     log_p_stay=0.5,
                                     log_p_to_gap=-3.0,
                                     log_p_from_gap=-2.5,
                                     log_p_gap_stay=-0.2)

    print(f"Found {len(tracks)} tracks")
    for i, tr in enumerate(tracks):
        print(f"Track {i}: start={tr['start']} end={tr['end']} energy={tr['energy']:.3f} score={tr['score']:.1f}")

    # plot
    plt.figure(figsize=(10, 4))
    plt.imshow(10 * np.log10(S + 1e-12).T, origin='lower', aspect='auto')
    for i, tr in enumerate(tracks):
        path = tr['path']
        # plot only active segments (non-GAP)
        active_mask = (path < N)
        times = np.arange(T)[active_mask]
        freqs = path[active_mask]
        plt.plot(times, freqs, linewidth=2, label=f"track {i}")
    plt.xlabel('time')
    plt.ylabel('freq bin')
    plt.title('Extracted tracks (only active parts shown)')
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------
# Multi-track Viterbi with GAP state (vectorized over multiple tracks)
# -----------------------

def build_transition_matrix(F, sigma=2.0, p_birth=0.001, p_death=0.001, p_stay_gap=0.99):
    """
    Returns (F+1)x(F+1) transition matrix.
    Last state index = GAP.
    """
    A = np.zeros((F+1, F+1))

    # --- Frequency → Frequency ---
    freq = np.arange(F)
    for f in range(F):
        A[f, :F] = np.exp(-0.5 * ((freq - f) / sigma)**2)

    # Normalize rows
    A[:F, :F] /= A[:F, :F].sum(axis=1, keepdims=True)

    # --- Add death transitions ---
    A[:F, F] = p_death
    A[:F, :F] *= (1 - p_death)

    # --- GAP transitions ---
    A[F, F] = p_stay_gap
    A[F, :F] = (1 - p_stay_gap) / F  # uniform birth
    
    # Normalize GAP row
    A[F] /= A[F].sum()

    return A


def viterbi_multitrack_gap(S, K, sigma=2.0, p_birth=0.001, p_death=0.001, p_stay_gap=0.99):
    """
    Multi-track Viterbi with GAP state.
    
    S: (T, F) log-likelihood spectrogram
    K: number of tracks
    """
    T, F = S.shape
    GAP = F  # index
    A = build_transition_matrix(F, sigma, p_birth, p_death, p_stay_gap=p_stay_gap)

    # Observation matrix extended with GAP
    S_ext = np.zeros((T, F+1))
    S_ext[:, :F] = S  # GAP has zero likelihood

    # DP tables
    V = np.zeros((T, K, F+1))       # V[t, track, state]
    BP = np.zeros((T, K, F+1), dtype=np.int32)

    # Initialization: all tracks start in GAP
    V[0] = S_ext[0]  # all tracks identical at t=0

    A_log = np.log(A + 1e-300)

    # Main Viterbi recursion (vectorized)
    for t in range(1, T):
        """
        For each track:
            V[t, k, s_new] = S[t, s_new] + max over s_old: V[t-1, k, s_old] + log A[s_old, s_new]
        """

        # shape: (K, F+1, F+1)
        scores = V[t-1][:, :, None] + A_log[None, :, :]

        # max over previous state
        V[t] = S_ext[t][None, :] + np.max(scores, axis=1)

        # store argmax
        BP[t] = np.argmax(scores, axis=1)

    # Backtracking for each track
    tracks = []
    for k in range(K):
        seq = np.zeros(T, dtype=int)
        seq[T-1] = np.argmax(V[T-1, k])
        for t in reversed(range(1, T)):
            seq[t-1] = BP[t, k, seq[t]]
        tracks.append(seq)

    return tracks, V, BP
