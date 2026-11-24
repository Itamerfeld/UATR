# ==============================
# Acoustic Detection Baselines
# Minimal Python Repository
# ==============================

# File: requirements.txt
numpy
scipy
matplotlib

# ==============================
# File: src/utils.py
# ==============================
import numpy as np
from scipy.signal import stft


def compute_stft(signal, fs, nperseg=1024, noverlap=512):
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, Zxx


# ==============================
# File: src/demon.py
# ==============================
import numpy as np
from scipy.signal import hilbert


def demon_detector(signal, fs):
    """
    DEMON: Detection of Envelope Modulation on Noise
    Returns envelope spectrum
    """
    analytic = hilbert(signal)
    envelope = np.abs(analytic)

    spectrum = np.fft.rfft(envelope)
    freqs = np.fft.rfftfreq(len(envelope), d=1/fs)

    return freqs, np.abs(spectrum)


# ==============================
# File: src/spectral_entropy.py
# ==============================
import numpy as np
from .utils import compute_stft


def spectral_entropy_detector(signal, fs):
    f, t, Zxx = compute_stft(signal, fs)

    psd = np.abs(Zxx) ** 2
    psd_sum = np.sum(psd, axis=0, keepdims=True) + 1e-12
    p = psd / psd_sum

    entropy = -(p * np.log(p + 1e-12)).sum(axis=0)

    return t, entropy


# ==============================
# File: src/cepstrum.py
# ==============================
import numpy as np
from .utils import compute_stft


def cepstrum_detector(signal, fs):
    f, t, Zxx = compute_stft(signal, fs)

    log_spectrum = np.log(np.abs(Zxx) + 1e-12)
    cepstrum = np.fft.ifft(log_spectrum, axis=0).real

    return t, cepstrum


# ==============================
# File: src/synthetic.py
# ==============================
import numpy as np


def generate_motor_signal(fs=8000, duration=5.0, f0=50, harmonics=5, noise_level=0.5):
    t = np.arange(0, duration, 1/fs)
    signal = np.zeros_like(t)

    for k in range(1, harmonics + 1):
        signal += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t)

    # amplitude modulation (prop/motor coupling)
    mod = 1 + 0.3 * np.sin(2 * np.pi * 2 * t)
    signal *= mod

    noise = noise_level * np.random.randn(len(t))

    return t, signal + noise


# ==============================
# File: main.py
# ==============================
import matplotlib.pyplot as plt

from src.synthetic import generate_motor_signal
from src.demon import demon_detector
from src.spectral_entropy import spectral_entropy_detector
from src.cepstrum import cepstrum_detector


def main():
    fs = 8000

    # Generate synthetic motor signal
    t, signal = generate_motor_signal(fs=fs)

    # DEMON
    f_demon, demon_spec = demon_detector(signal, fs)

    # Spectral entropy
    t_ent, entropy = spectral_entropy_detector(signal, fs)

    # Cepstrum
    t_cep, cepstrum = cepstrum_detector(signal, fs)

    # ==============================
    # Plot
    # ==============================
    plt.figure()
    plt.plot(f_demon, demon_spec)
    plt.title("DEMON Spectrum")

    plt.figure()
    plt.plot(t_ent, entropy)
    plt.title("Spectral Entropy")

    plt.figure()
    plt.imshow(cepstrum, aspect='auto', origin='lower')
    plt.title("Cepstrum")

    plt.show()


if __name__ == "__main__":
    main()
