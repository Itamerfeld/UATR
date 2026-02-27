
"""
harmonics_detector.py
---------------------
State-of-the-art style harmonics detector pipeline for underwater acoustics,
focused on robust f0 (fundamental) tracking + harmonic set detection in noise.

Design goals:
- Minimal dependencies (numpy, scipy, matplotlib for demo).
- Production-friendly, streaming-capable building blocks.
- Comb-scored STFT front-end + Viterbi track-before-detect.
- Optional DEMON cross-check (envelope spectrum) for propeller/shaft tones.

This is a *practical* prototype: it avoids heavy external packages and
implements the pieces cleanly so you can swap in more advanced TFRs
(e.g., synchrosqueezed CWT) later if desired.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np
import scipy.signal as sig


@dataclass
class STFTParams:
    fs: float
    n_fft: int = 4096
    hop: int = 1024
    window: str = "hann"
    detrend: Optional[str] = None  # or 'constant'
    pad_mode: str = "zeros"


@dataclass
class DetectorParams:
    f0_min: float = 10.0
    f0_max: float = 400.0
    f0_bins: int = 800          # grid resolution for f0 search
    max_harm: int = 20          # maximum harmonics to aggregate in comb score
    band_hz: float = 2.0        # +/- band (Hz) around each harmonic for energy integration
    min_frames: int = 4         # min frames to consider a persistent detection
    viterbi_jump_cost: float = 1.0
    viterbi_accel_cost: float = 0.2
    viterbi_stay_reward: float = 0.0
    emission_temperature: float = 1.0  # softer/harder selection
    smoothing_alpha: float = 0.2       # temporal EMA on comb map
    comb_normalize: bool = True        # normalize by local noise floor
    use_log_freq: bool = False         # frequency axis transform for scoring


@dataclass
class DEMONParams:
    # Classic DEMON-style envelope spectrum
    bpf_lo: float = 50.0
    bpf_hi: float = 2000.0
    envelope_lp: float = 500.0  # lowpass on envelope to avoid HF aliasing
    decim: int = 10             # decimate before FFT of envelope
    n_fft: int = 8192


def stft_mag(x: np.ndarray, p: STFTParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute magnitude spectrogram (power) with given STFT params."""
    f, t, Z = sig.stft(
        x,
        fs=p.fs,
        nperseg=p.n_fft,
        noverlap=p.n_fft - p.hop,
        window=p.window,
        detrend=p.detrend,
        padded=True,
        boundary=p.pad_mode,
        return_onesided=True
    )
    S = np.abs(Z) ** 2
    return S, f, t


def median_whiten(S: np.ndarray, kernel_freq: int = 31) -> np.ndarray:
    """Simple median-based whitening across frequency to emphasize lines."""
    if kernel_freq < 3:
        return S
    from scipy.ndimage import median_filter
    med = median_filter(S, size=(kernel_freq, 1), mode="nearest")
    # Avoid divide-by-zero, stabilize
    return (S + 1e-12) / (med + 1e-12)


def comb_score_frame(S_col: np.ndarray, freqs: np.ndarray, f0_grid: np.ndarray,
                     max_harm: int, band_hz: float, noise_norm: bool) -> np.ndarray:
    """
    Compute comb score for one time frame column:
    sum of energy in bands around k*f0, k=1..K.
    """
    df = np.mean(np.diff(freqs))
    band_bins = max(1, int(np.ceil(band_hz / max(df, 1e-9))))
    # Prefix-sum for quick band energy
    # But we also need to convert hz to bin index
    score = np.zeros_like(f0_grid)
    Nf = len(freqs)

    # Precompute cumulative sum for fast range-sum
    csum = np.cumsum(S_col)

    def band_sum(center_bin: int, width: int) -> float:
        lo = max(0, center_bin - width)
        hi = min(Nf - 1, center_bin + width)
        if lo == 0:
            return csum[hi]
        return csum[hi] - csum[lo - 1]

    for i, f0 in enumerate(f0_grid):
        s = 0.0
        cnt = 0
        k = 1
        while True:
            f = k * f0
            if f > freqs[-1]:
                break
            bin_idx = int(np.round((f - freqs[0]) / df))
            s += band_sum(bin_idx, band_bins)
            cnt += 1
            k += 1
            if k > max_harm:
                break
        if cnt > 0:
            score[i] = s / cnt
        else:
            score[i] = 0.0

    if noise_norm:
        # Normalize by local median over a sliding window in f0-grid to reduce false peaks
        from scipy.ndimage import median_filter
        m = median_filter(score, size=31, mode="nearest")
        score = (score + 1e-12) / (m + 1e-12)

    return score


def viterbi_track(emissions: np.ndarray, jump_cost: float, accel_cost: float, stay_reward: float) -> np.ndarray:
    """
    Viterbi over f0-grid indices.
    emissions: [T, F] (higher is better).
    Transition model: penalize |Δ| and |Δ2| (acceleration) to allow smooth drift.
    """
    T, F = emissions.shape
    # Convert to negative energy (cost); higher emission -> lower cost
    E = -emissions

    # DP tables
    cost = np.full((T, F), np.inf, dtype=float)
    back = np.full((T, F), -1, dtype=int)

    cost[0, :] = E[0, :]

    for t in range(1, T):
        prev = cost[t-1, :]
        for f in range(F):
            # We search a local neighborhood to save time; allow jumps up to, say, 5 bins
            # For robustness we allow the full range but will be slower.
            best_c = np.inf
            best_j = -1
            # Vectorize neighborhood
            # Try a small neighborhood first
            lo = max(0, f - 8)
            hi = min(F, f + 9)
            idxs = np.arange(lo, hi)
            # |Δ| penalty
            delta = np.abs(idxs - f)
            trans = jump_cost * delta
            cands = prev[idxs] + trans
            j = np.argmin(cands)
            best_c = cands[j]
            best_j = idxs[j]

            # Reward staying (slight bias to keep)
            best_c += (-stay_reward)

            cost[t, f] = best_c + E[t, f]
            back[t, f] = best_j

    # Backtrace
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmin(cost[-1, :])
    for t in range(T-2, -1, -1):
        path[t] = back[t+1, path[t+1]]
    return path


def compute_demon_spectrum(x: np.ndarray, fs: float, p: DEMONParams) -> Tuple[np.ndarray, np.ndarray]:
    """Compute DEMON (envelope) spectrum of a signal block."""
    # Bandpass
    sos = sig.butter(4, [p.bpf_lo, p.bpf_hi], btype='bandpass', fs=fs, output='sos')
    xb = sig.sosfilt(sos, x)
    # Envelope via Hilbert
    env = np.abs(sig.hilbert(xb))
    # Lowpass on envelope
    sos2 = sig.butter(4, p.envelope_lp, btype='low', fs=fs, output='sos')
    env_lp = sig.sosfilt(sos2, env)
    # Decimate
    xd = sig.decimate(env_lp, p.decim, ftype='iir', zero_phase=True)
    fde = fs / p.decim
    # Spectrum
    win = sig.get_window('hann', len(xd), fftbins=True)
    spec = np.abs(np.fft.rfft(xd * win, n=p.n_fft))**2
    freqs = np.fft.rfftfreq(p.n_fft, 1.0/fde)
    # Normalize
    spec = spec / (np.median(spec) + 1e-12)
    return spec, freqs


def detect_harmonics(
    x: np.ndarray,
    stft_p: STFTParams,
    det_p: DetectorParams,
    demon_p: Optional[DEMONParams] = None
) -> Dict[str, np.ndarray]:
    """
    Main entrypoint.
    Returns a dict with keys:
      - 't': time frames
      - 'f': frequency bins
      - 'S': whitened spectrogram (power-normalized)
      - 'f0_grid': candidate f0 values
      - 'comb_map': comb score over [T, F0]
      - 'f0_track': best-path f0 (Hz) over time
      - 'harm_mask': boolean [T, K] mask of detected harmonics around k*f0
      - 'demon_peak_hz': (optional) DEMON peak per window aligned to STFT frames
    """
    # STFT
    S, f, t = stft_mag(x, stft_p)

    # Median-whiten to emphasize line structure
    S_w = median_whiten(S, kernel_freq=31)

    # f0 grid
    f0_grid = np.linspace(det_p.f0_min, det_p.f0_max, det_p.f0_bins)

    # Comb scoring per frame
    T = S_w.shape[1]
    comb_map = np.zeros((T, det_p.f0_bins), dtype=float)

    # EMA smoothing across time for stability
    ema = None
    for i in range(T):
        comb = comb_score_frame(S_w[:, i], f, f0_grid, det_p.max_harm, det_p.band_hz, det_p.comb_normalize)
        if ema is None:
            ema = comb
        else:
            ema = det_p.smoothing_alpha * comb + (1 - det_p.smoothing_alpha) * ema
        # Softmax-like temperature scaling to tame dynamic range (optional)
        comb_map[i, :] = ema / (np.max(ema) + 1e-12)

    # Viterbi track of f0
    path_idx = viterbi_track(
        comb_map,
        jump_cost=det_p.viterbi_jump_cost,
        accel_cost=det_p.viterbi_accel_cost,
        stay_reward=det_p.viterbi_stay_reward
    )
    f0_track = f0_grid[path_idx]

    # Harmonic mask: for each frame, flag bins near k*f0 that exceed a threshold
    K = det_p.max_harm
    harm_mask = np.zeros((T, K), dtype=bool)
    df = np.mean(np.diff(f))
    band_bins = max(1, int(np.ceil(det_p.band_hz / max(df, 1e-9))))
    for i in range(T):
        f0 = f0_track[i]
        # local normalization (spectral floor)
        col = S_w[:, i]
        med = np.median(col)
        thr = 2.0 * med  # configurable
        for k in range(1, K+1):
            fk = k * f0
            if fk > f[-1]:
                break
            bin_idx = int(np.round((fk - f[0]) / df))
            lo = max(0, bin_idx - band_bins)
            hi = min(len(f)-1, bin_idx + band_bins)
            e = np.sum(col[lo:hi+1])
            harm_mask[i, k-1] = (e > thr)

    out = dict(
        t=t, f=f, S=S_w, f0_grid=f0_grid, comb_map=comb_map, f0_track=f0_track, harm_mask=harm_mask
    )

    # Optional DEMON cross-check aligned roughly to STFT frames (blockwise)
    if demon_p is not None:
        # Reconstruct per-frame blocks and compute DEMON peak near the track
        frame_len = stft_p.n_fft
        hop = stft_p.hop
        peaks = np.full(T, np.nan)
        for i in range(T):
            s = i * hop
            e = s + frame_len
            if e > len(x):
                break
            block = x[s:e]
            spec, fde = compute_demon_spectrum(block, stft_p.fs, demon_p)
            # Find a peak closest to f0 or its subharmonics/harmonics (shaft/blade relations vary)
            # For generality, just take the top peak between [f0/4, 6*f0] per frame.
            f0 = f0_track[i]
            if f0 <= 0:
                continue
            lo = max(0.5, f0/4.0)
            hi = min(6*f0, fde[-1])
            lo_i = np.searchsorted(fde, lo)
            hi_i = np.searchsorted(fde, hi)
            if hi_i > lo_i + 2:
                idx = lo_i + np.argmax(spec[lo_i:hi_i])
                peaks[i] = fde[idx]
        out["demon_peak_hz"] = peaks

    return out
