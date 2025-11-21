import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import librosa
import librosa.display
from scipy import signal
import soundfile as sf


class Audio:

    def __init__(self, file_path):
        self.file_path = file_path
        self.data, self.fs = librosa.load(file_path, sr=None)
        self.fft_params = {
            'n_fft': 16384,
            'window': 'hann',
            'overlap': 0.5,
            'color_map': 'magma',
            'crop_freq': 4000  # Hz
        }
        self.f, self.t, self.Sxx = signal.spectrogram(data, fs, 
                                                    window=self.fft_params['window'],
                                                    nperseg=self.fft_params['n_fft'],
                                                    noverlap=int(self.fft_params['n_fft'] * (1 - self.fft_params['overlap'])),
                                                    scaling='spectrum')
    def plot_spectrogram(self):
        plt.figure(figsize=(15, 8))
        plt.pcolormesh(self.t, self.f, 10 * np.log10(self.Sxx + 1e-10), cmap=self.fft_params['color_map'])
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Spectrogram')
        plt.colorbar(label='Power Spectral Density [dB]')
        plt.ylim(0, self.fft_params['crop_freq'])  # Limit frequency axis to crop_freq
        plt.show()


# Usage example:
if __name__ == "__main__":
    data_file = "../data/scooter_example_1.wav"
    audio = Audio(data_file)
    audio.plot_spectrogram()
