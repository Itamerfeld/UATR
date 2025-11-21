import numpy as np
import scipy.signal as sg
import pywt
import matplotlib.pyplot as plt

# -------------------------
# Config object
# -------------------------
class PipelineConfig:
    def __init__(self,
                 fs,
                 stft_nfft=4096,
                 stft_hop=256,
                 f0_min=10.0,
                 f0_max=200.0,
                 f0_bins=400,
                 max_harm=12,
                 demon_band=(80, 3000),
                 demon_frame=2.0,
                 demon_hop=0.5):
        self.fs = fs
        self.stft_nfft = stft_nfft
        self.stft_hop = stft_hop
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_bins = f0_bins
        self.max_harm = max_harm
        self.demon_band = demon_band
        self.demon_frame = demon_frame
        self.demon_hop = demon_hop

# -------------------------
# STFT + peak picking
# -------------------------
def compute_stft(x, cfg):
    f, t, S = sg.stft(
        x,
        fs=cfg.fs,
        nperseg=cfg.stft_nfft,
        noverlap=cfg.stft_nfft - cfg.stft_hop,
        nfft=cfg.stft_nfft,
        return_onesided=True
    )
    return f, t, np.abs(S)

def stft_peak_pick(S, f, cfg, prominence=6.0):
    peaks_per_frame = []
    for frame in range(S.shape[1]):
        spec = 20 * np.log10(S[:, frame] + 1e-9)
        med = np.median(spec)
        idx, _ = sg.find_peaks(spec, prominence=prominence, height=med + prominence)
        peaks_per_frame.append(f[idx])
    return peaks_per_frame

# -------------------------
# Cepstrum analysis
# -------------------------
def cepstrum_f0(x, cfg):
    n = int(cfg.fs * cfg.demon_frame)
    step = int(cfg.fs * cfg.demon_hop)
    frames = []
    for i in range(0, len(x) - n, step):
        frame = x[i:i+n] * sg.hann(n)
        spectrum = np.fft.rfft(frame)
        log_mag = np.log(np.abs(spectrum) + 1e-9)
        cep = np.fft.irfft(log_mag)
        q = np.arange(len(cep)) / cfg.fs
        mask = (q >= 1/cfg.f0_max) & (q <= 1/cfg.f0_min)
        if np.any(mask):
            q_sel = q[mask]
            idx = np.argmax(cep[mask])
            f0 = 1.0 / q_sel[idx]
            frames.append(f0)
        else:
            frames.append(None)
    return frames

# -------------------------
# DEMON analysis
# -------------------------
def demon_f0(x, cfg):
    b, a = sg.butter(4, np.array(cfg.demon_band)/(cfg.fs/2), btype='band')
    y = sg.filtfilt(b, a, x)
    env = np.abs(sg.hilbert(y))
    n = int(cfg.fs * cfg.demon_frame)
    step = int(cfg.fs * cfg.demon_hop)
    f0s = []
    for i in range(0, len(env) - n, step):
        frame = env[i:i+n] * sg.hann(n)
        spec = np.fft.rfft(frame)
        f = np.fft.rfftfreq(len(frame), 1.0/cfg.fs)
        idx = np.argmax(np.abs(spec))
        f0s.append(f[idx])
    return f0s

# -------------------------
# Wavelet scalogram energy (optional score)
# -------------------------
def wavelet_score(x, cfg):
    scales = np.arange(1, 256)
    coeffs, freqs = pywt.cwt(x, scales, 'morl', sampling_period=1.0/cfg.fs)
    energy = np.abs(coeffs) ** 2
    return freqs, energy

# -------------------------
# HMM / Viterbi tracker
# -------------------------
def fuse_scores_viterbi(score_matrix, f0_candidates, trans_penalty=1.0):
    T, K = score_matrix.shape
    dp = np.zeros((T, K))
    ptr = np.zeros((T, K), dtype=int)

    dp[0, :] = score_matrix[0, :]
    for t in range(1, T):
        for k in range(K):
            costs = dp[t-1, :] - trans_penalty * np.abs(f0_candidates[k] - f0_candidates)
            ptr[t, k] = np.argmax(costs)
            dp[t, k] = score_matrix[t, k] + costs[ptr[t, k]]

    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(dp[-1, :])
    for t in range(T-2, -1, -1):
        path[t] = ptr[t+1, path[t+1]]
    return f0_candidates[path]

# -------------------------
# Main pipeline with fusion
# -------------------------
def harmonic_detection_pipeline(x, cfg):
    # STFT
    f, t, S = compute_stft(x, cfg)
    stft_peaks = stft_peak_pick(S, f, cfg)

    # Cepstrum
    cep_f0 = cepstrum_f0(x, cfg)

    # DEMON
    demon_candidates = demon_f0(x, cfg)

    # Candidate f0 grid
    f0_grid = np.linspace(cfg.f0_min, cfg.f0_max, cfg.f0_bins)
    score_matrix = np.zeros((len(t), len(f0_grid)))

    # Fill scores frame by frame
    for ti in range(len(t)):
        for ki, f0 in enumerate(f0_grid):
            score = 0.0

            # STFT: reward if harmonics align with peaks
            if ti < len(stft_peaks):
                harm_freqs = f0 * np.arange(1, cfg.max_harm+1)
                close = np.any([np.min(np.abs(stft_peaks[ti] - hf)) < 2.0 for hf in harm_freqs])
                if close:
                    score += 1.0

            # Cepstrum: reward proximity to cepstrum-estimated f0
            if ti < len(cep_f0) and cep_f0[ti] is not None:
                score += np.exp(-0.5*((f0 - cep_f0[ti]) / 2.0)**2)

            # DEMON: reward proximity to DEMON-estimated f0
            if ti < len(demon_candidates):
                score += np.exp(-0.5*((f0 - demon_candidates[ti]) / 2.0)**2)

            score_matrix[ti, ki] = score

    # Fuse with Viterbi
    f0_track = fuse_scores_viterbi(score_matrix, f0_grid, trans_penalty=0.5)

    harmonics = np.array([f0_track * (h+1) for h in range(cfg.max_harm)])
    return {"t": t, "f0_track": f0_track, "harmonics": harmonics, "score_matrix": score_matrix}

def visualize_results(x, cfg, results):
    """
    Plots spectrogram with estimated f0 track + harmonics.
    """
    t = results["t"]
    f0_track = results["f0_track"]
    harmonics = results["harmonics"]

    # Compute spectrogram again for plotting
    f, t_spec, Sxx = sg.spectrogram(
        x,
        fs=cfg.fs,
        nperseg=cfg.stft_nfft,
        noverlap=cfg.stft_nfft - cfg.stft_hop,
        nfft=cfg.stft_nfft
    )

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t_spec, f, 20*np.log10(Sxx+1e-12), shading='gouraud', cmap='magma')
    plt.colorbar(label="Power [dB]")
    plt.ylim(0, cfg.f0_max * cfg.max_harm * 1.2)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Harmonic Series Detection")

    # Plot estimated f0 track
    plt.plot(t, f0_track, 'c-', linewidth=2, label="Estimated f0")

    # Overlay harmonics
    for h in range(harmonics.shape[0]):
        plt.plot(t, harmonics[h, :], 'w--', alpha=0.6)

    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    cfg = PipelineConfig(fs=48000)
    results = harmonic_detection_pipeline(audio_signal, cfg)
    visualize_results(audio_signal, cfg, results)
