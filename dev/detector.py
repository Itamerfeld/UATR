import librosa
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import find_peaks, butter, filtfilt, sosfiltfilt, welch, hilbert
from scipy.ndimage import gaussian_filter
from scipy.io import wavfile
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class WorkingDetector:
    def __init__(self, fs, nperseg=8192, overlap=0.5, nfft=None, window='hanning', dc=20, crop_freq=None, norm_size=9, default_distance=3, threshold=10):
        self.fs = fs
        self.nperseg = nperseg
        self.overlap = overlap
        self.nfft = nfft
        self.window = window
        self.dc = dc
        self.crop_freq = crop_freq
        self.norm_size = norm_size
        self.default_distance = default_distance
        self.default_threshold = threshold

    def detect(self, rx, threshold=None, mode='self_correlation'):
        if mode == 'self_correlation':
            scores, F = self.get_self_correlation_vector(rx)
        elif mode == 'demon_spectrum':
            scores, F = self.get_demon_spectrum(rx)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        scores = scores - np.mean(scores)
        scores = scores / (np.std(scores) + 1e-10)
        if threshold is None:
            threshold = self.default_threshold
        detections = find_peaks(scores, height=threshold, distance=self.default_distance)[0]
        is_detected = len(detections) > 0
        return is_detected, detections, (scores, F)
    
    def get_self_correlation_vector(self, rx):
        F, T, Sxx, _ = calc_spectrogram(rx, self.fs, nperseg=self.nperseg, percent_overlap=self.overlap, nfft=self.nfft, window=self.window, remove_dc=self.dc, crop_freq=self.crop_freq)
        Sxx_odd = Sxx[:, 1::2]
        Sxx_even = Sxx[:, ::2]

        corr_scores = np.zeros((len(F)))
        for i in range(len(F)):
            n = min(Sxx_odd.shape[1], Sxx_even.shape[1])
            corr_scores[i] = np.corrcoef(Sxx_odd[i, :n], Sxx_even[i, :n])[0][1]
        
        return corr_scores, F

    def get_demon_spectrum(self, rx):
        analytic = hilbert(rx)
        envelope = np.abs(analytic)

        spectrum = np.fft.rfft(envelope)
        freqs = np.fft.rfftfreq(len(envelope), d=1/self.fs)

        if self.dc is not None:
            dc_ix = np.where(freqs > self.dc)[0][0]
            spectrum = spectrum[dc_ix:]
            freqs = freqs[dc_ix:]

        if self.crop_freq is not None:
            crop_ix = np.where(freqs > self.crop_freq)[0][0]
            spectrum = spectrum[:crop_ix]
            freqs = freqs[:crop_ix]

        return np.abs(spectrum), freqs

def calc_spectrogram(x, fs, nperseg, percent_overlap, nfft=None, window='hamming', remove_dc=None, crop_freq=None, logscale=True):
    # prepare for PSD calculation
    noverlap = int(nperseg * percent_overlap)
    step = nperseg - noverlap
    
    if window == 'flattop':
        window_vals = signal.windows.flattop(nperseg)
    elif hasattr(np, window):
        window_vals = getattr(np, window)(nperseg)
    elif hasattr(signal.windows, window):
        window_vals = signal.get_window(window, nperseg)
    else:
        raise ValueError(f"Window '{window}' not found in numpy or scipy.signal.windows")
        
    window_power = np.sum(window_vals**2)

    if nfft is None:
        nfft = nperseg

    n_segments = (len(x) - noverlap) // step
    f = np.fft.rfftfreq(nfft, 1/fs)
    t = np.arange(n_segments) * step / fs
    Sxx = np.zeros((n_segments, len(f)))
    Phase = np.zeros((n_segments, len(f)))

    for i in range(n_segments):
        start = i * step
        segment = x[start:start+nperseg]
        if len(segment) < nperseg:
            break
        segment = segment * window_vals
        spectrum = np.fft.rfft(segment, n=nfft)
        Sxx[i, :] = (np.abs(spectrum)**2) / (fs * window_power)
        Phase[i, :] = np.angle(spectrum)

    # remove DC component
    if remove_dc is not None:
        dc_band = f <= remove_dc
        f = f[~dc_band]
        Sxx = Sxx[:, ~dc_band]
        Phase = Phase[:, ~dc_band]

    if crop_freq is not None:
        if crop_freq > fs / 2:
            crop_freq = fs // 2 - 1  # Nyquist limit
        crop_band = f <= crop_freq
        f = f[crop_band]
        Sxx = Sxx[:, crop_band]
        Phase = Phase[:, crop_band]

    if logscale:
        Sxx = 10 * np.log10(Sxx + 1e-100)

    return f, t, Sxx.T, Phase.T

if __name__ == "__main__":
    # parameters
    nperseg = 8192
    overlap = 0.5
    nfft = None
    window = 'hanning'
    dc = 20
    crop_freq = None
    norm_size = 9
    default_distance = 3
    threshold = 10

    # simulate signal with noise
    f0 = 1000 # Hz
    fs = 8000 # Hz
    duration = 60 # seconds
    snr_db = -5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * f0 * t)
    signal_power = np.mean(sig ** 2)    
    desired_noise_power_in_band = signal_power / (10 ** (snr_db / 10))
    bandwidth_ratio = (fs / 2) / 1
    total_desired_noise_power = desired_noise_power_in_band * bandwidth_ratio
    noise = np.random.rand(len(sig))
    current_noise_power = np.mean(noise ** 2)
    scaling_factor = np.sqrt(total_desired_noise_power / current_noise_power)
    scaled_noise = noise * scaling_factor
    noisy_signal = sig + scaled_noise

    ## upload signal from file
    # path = 'data/ds2/pos/dpv1_1m.wav'
    # fs, noisy_signal = wavfile.read(path)

    # calc spectrogram
    F, T, Sxx, _ = calc_spectrogram(noisy_signal, fs, nperseg=nperseg, percent_overlap=overlap, nfft=nfft, window=window, remove_dc=dc, crop_freq=crop_freq, logscale=True)

    # detect
    detector = WorkingDetector(fs=fs, nperseg=nperseg, overlap=overlap, nfft=nfft, window=window, dc=dc, crop_freq=crop_freq, norm_size=norm_size, default_distance=default_distance, threshold=threshold)
    is_detected_corr, detections_corr, (scores_corr, F_corr) = detector.detect(noisy_signal, mode='self_correlation')
    is_detected_demon, detections_demon, (scores_demon, F_demon) = detector.detect(noisy_signal, mode='demon_spectrum')

    # plot
    fig = make_subplots(rows=1, cols=3, column_titles=['spectrogram', 'self_correlation', 'demon_spectrum'], column_widths=[0.8, 0.1, 0.1])
    fig.add_trace(go.Heatmap(z=Sxx, x=T, y=F, colorscale='viridis', showlegend=False, showscale=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=scores_corr, y=F_corr, mode='lines', marker=dict(color='blue', size=10), showlegend=False), row=1, col=2)
    if is_detected_corr:
        fig.add_trace(go.Scatter(x=scores_corr[detections_corr], y=F_corr[detections_corr], mode='markers', marker=dict(color='red', size=6), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=scores_demon, y=F_demon, mode='lines', marker=dict(color='blue', size=10), showlegend=False), row=1, col=3)
    if is_detected_demon:
        fig.add_trace(go.Scatter(x=scores_demon[detections_demon], y=F_demon[detections_demon], mode='markers', marker=dict(color='red', size=6), showlegend=False), row=1, col=3)
    fig.show()