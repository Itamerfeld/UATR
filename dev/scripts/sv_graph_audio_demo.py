import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import networkx as nx
from scipy.signal import find_peaks

# ----------------------------
# Utility: build Spectral Visibility Graph (SVg)
# ----------------------------
def spectral_visibility_graph(spectrum):
    """
    Build a visibility graph from a 1D magnitude spectrum.
    Returns a NetworkX graph and the degree sequence.
    """
    n = len(spectrum)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i+1, n):
            # check visibility condition
            visible = True
            for k in range(i+1, j):
                if spectrum[k] >= spectrum[i] + (spectrum[j]-spectrum[i]) * (k-i)/(j-i):
                    visible = False
                    break
            if visible:
                G.add_edge(i, j)
    degree_seq = np.array([d for _, d in G.degree()])
    return G, degree_seq

# ----------------------------
# Load audio
# ----------------------------
filename = librosa.example('trumpet')  # demo signal (replace with your WAV)
y, sr = librosa.load(filename, sr=None)

# Short segment for demo
y = y[:5*sr]

# ----------------------------
# STFT
# ----------------------------
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

# Pick one frame for demonstration
frame = S[:, 50]

# ----------------------------
# Compute visibility graph
# ----------------------------
G, degree_seq = spectral_visibility_graph(frame)

# ----------------------------
# Plot spectrum vs degree
# ----------------------------
fig, ax = plt.subplots(2, 1, figsize=(10,6))
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         sr=sr, hop_length=512, y_axis='log', x_axis='time', ax=ax[0])
ax[0].set_title('Spectrogram')

ax[1].plot(frame, label='Spectrum')
ax[1].plot(degree_seq, label='VG Degree')
ax[1].set_title('Spectrum vs. VG Degree (frame 50)')
ax[1].legend()
plt.tight_layout()
plt.show()

# ----------------------------
# Simple tonal detector
# ----------------------------
# Rule: tonals ↔ bins with degree above mean+std
threshold = degree_seq.mean() + degree_seq.std()
tonals = np.where(degree_seq > threshold)[0]

print("Detected tonal bins (indices):", tonals)

# Baseline: simple peak counting
peaks, _ = find_peaks(frame, height=np.median(frame)*2)
print("Baseline peaks (indices):", peaks)
