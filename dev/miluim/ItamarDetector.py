# imports for detector
import numpy as np
from scipy import signal
from scipy.signal import find_peaks

# imports for test and plotting
from scipy.io import wavfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# detector class
class ItamarDetector:
    def __init__(self, fs, nperseg, overlap, window, dc, crop_freq, window_size, default_distance=3):
        self.fs = fs
        self.nperseg = nperseg
        self.overlap = overlap
        self.window = window
        self.dc = dc
        self.crop_freq = crop_freq
        self.window_size = window_size
        self.default_distance = default_distance

    def detect(self, rx, threshold):
        pxx, F = self.get_feature_vector(rx)
        TH = threshold
        detections = find_peaks(pxx, height=TH, distance=self.default_distance)[0]
        is_detected = len(detections) > 0
        return is_detected, detections, (pxx, F)
    
    def get_feature_vector(self, rx):
        F, T, Sxx, phasogram = self.calc_spectrogram(rx, self.fs, nperseg=self.nperseg, percent_overlap=self.overlap, window=self.window, remove_dc=self.dc, crop_freq=self.crop_freq)
        pxx = self._calc_welch_from_spectrogram(Sxx)
        return pxx, F

    @staticmethod
    def calc_spectrogram(x, fs, nperseg, percent_overlap, window='hanning', remove_dc=None, crop_freq=None, logscale=True):
        # prepare for PSD calculation
        noverlap = int(nperseg * percent_overlap)
        step = nperseg - noverlap
        window_vals = getattr(np, window)(nperseg)
        window_power = np.sum(window_vals**2)

        n_segments = (len(x) - noverlap) // step
        f = np.fft.rfftfreq(nperseg, 1/fs)
        t = np.arange(n_segments) * step / fs
        Sxx = np.zeros((n_segments, len(f)))
        Phase = np.zeros((n_segments, len(f)))

        for i in range(n_segments):
            start = i * step
            segment = x[start:start+nperseg]
            if len(segment) < nperseg:
                break
            segment = segment * window_vals
            spectrum = np.fft.rfft(segment)
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

    def _calc_welch_from_spectrogram(self, Sxx):
        Pxx = np.mean(Sxx, axis=1)
        if self.window_size is not None:
            Pxx = self._rw_normalization(Pxx)
        Pxx = np.abs(Pxx - 1)
        return Pxx

    def _rw_normalization(self, x):
        if self.window_size % 2 == 0:
            self.window_size += 1  # make it odd
        normalization_kernel = np.ones((self.window_size,)) / (self.window_size-1)
        normalization_kernel[self.window_size // 2] = 0
        smooth_x = signal.convolve(x, normalization_kernel, mode='same')
        ret = x / (smooth_x + 1e-10)

        # handle edge cases
        ret[:self.window_size//2] = ret[self.window_size//2]
        ret[-self.window_size//2:] = ret[-self.window_size//2-1]
        return ret

    def _cfar_normalization(self, x, guard_size=2):
        if window_size % 2 == 0:
            window_size += 1  # make it odd
        normalization_kernel = np.ones((window_size,))
        normalization_kernel[guard_size:-guard_size] = 0
        normalization_kernel = normalization_kernel / np.sum(normalization_kernel)
        smooth_x = signal.convolve(x, normalization_kernel, mode='same')
        ret = x / (smooth_x + 1e-10)
        return ret


# test the detector on a single file (sanity check)
if __name__ == "__main__":

    file_path = 'data/ds2/pos/dpv1_1m.wav'
    fs, data = wavfile.read(file_path)

    nperseg = 16384
    overlap = 0.5
    window = 'hanning'
    dc = 100
    crop_freq = 32000
    window_size = 5
    default_distance = 3
    threshold = 0.034

    detector = ItamarDetector(fs=fs, nperseg=nperseg, overlap=overlap, window=window, dc=dc, crop_freq=crop_freq, window_size=window_size, default_distance=default_distance)
    F, T, Sxx, Phase = detector.calc_spectrogram(data, fs, nperseg=nperseg, percent_overlap=overlap, window=window, remove_dc=dc, crop_freq=crop_freq, logscale=True)
    is_detected, detections, (pxx, F) = detector.detect(data, threshold)
    
    fig = make_subplots(rows=1, cols=2, column_widths=[0.9, 0.1], shared_yaxes=True)
    fig.add_trace(go.Heatmap(z=Sxx, x=T, y=F, colorscale='Viridis', showlegend=False, showscale=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=pxx, y=F, mode='lines', line=dict(color='blue'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=pxx[detections], y=F[detections], mode='markers', marker=dict(color='red', size=6), showlegend=False), row=1, col=2)
    fig.update_layout(height=900, width=1200, title_text="Spectrogram and Detection Metrics for All Recordings")
    fig.show(renderer="browser")
