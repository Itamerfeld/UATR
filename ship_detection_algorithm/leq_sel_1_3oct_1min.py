import numpy as np
import soundfile as sf
from scipy.io import loadmat
import config

def leq_SEL_1_3oct_1min(filename):
    signal, Fs = sf.read(filename)
    # 1/3 octave center frequencies
    Fc = np.array([
        63, 80, 100, 125, 160, 200, 250, 315,
        400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
        4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000,
        25000, 31500, 40000
    ], dtype=float)

    p_ref = 1.0

    # 1/3 octave edges
    g = 10 ** (1/20)
    Fl = Fc / g
    Fu = Fc * g
    Nbands = len(Fc)

    signal = np.asarray(signal).flatten()
    signal = signal - np.mean(signal)

    N = len(signal)
    T = N / Fs
    K = N // 2 + 1
    df = Fs / N
    Fnyq = Fs / 2

    # Frequency axis (positive frequencies only)
    Faxis = np.arange(K) * df

    # FFT
    X = np.fft.fft(signal)
    Xpos = X[:K]

    # One-sided PSD
    Sxx = (1 / (Fs * N)) * np.abs(Xpos) ** 2
    if K > 2:
        Sxx[1:-1] *= 2   # double non-DC, non-Nyquist bins

    # Output arrays
    Leq_dB = np.full(Nbands, np.nan)
    SEL_dB = np.full(Nbands, np.nan)

    for k in range(Nbands):

        # If upper edge > Nyquist → skip band
        if Fu[k] > Fnyq:
            continue

        # Frequency bin selection
        idx = (Faxis >= Fl[k]) & (Faxis < Fu[k])

        if not np.any(idx):
            continue

        # Integrate power over the band
        bandPower = np.sum(Sxx[idx]) * df

        if bandPower <= 0:
            Leq_dB[k] = -np.inf
            SEL_dB[k] = -np.inf
        else:
            Leq_dB[k] = 10 * np.log10(bandPower / (p_ref ** 2))
            SEL_dB[k] = Leq_dB[k] + 10 * np.log10(T)

    # Save results to text file
    filename = filename.split('/')[-1]
    filename = filename.replace('.wav','')
    filename = filename.replace('record_','')
    np.savetxt(f'{config.BASE_PATH}/ship_noise_sys/leq_vectors/{filename}.txt', Leq_dB)
    return Leq_dB #, SEL_dB


# Example usage

# Leq_dB = leq_SEL_1_3oct_1min('../ship_noise_sys/data_post_process/record_20251209_102958.wav')

# mat = loadmat('res.mat')
# matlab_res = mat['ans'][0]

# print('matlab answer: ',matlab_res)
# print('python answer: ',Leq_dB)

# # Compare results
# for i in range(len(Leq_dB)):
#     if Leq_dB[i] != matlab_res[i]:
#         print(f"Difference at band {i}: Python {Leq_dB[i]}, MATLAB {matlab_res[i]}")

