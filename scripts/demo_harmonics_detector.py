
"""
demo_harmonics_detector.py
--------------------------
Creates a synthetic underwater-like signal with slowly varying f0 and harmonics,
adds noise and occasional dropouts, then runs the harmonics detector.
Saves figures and a JSON with the tracked f0 for quick inspection.
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import json
from pathlib import Path

from harmonics_detector import STFTParams, DetectorParams, DEMONParams, detect_harmonics

outdir = Path(".")
fs = 96000
dur = 6.0
t = np.arange(int(fs*dur)) / fs

# Ground-truth f0: slow chirp 60->85 Hz with slight jitter
f0_true = 60 + 25 * (t / dur) + 1.5*np.sin(2*np.pi*0.2*t)
phi = 2*np.pi * np.cumsum(f0_true) / fs

# Build harmonic series (K up to 12), with amplitude roll-off
x = np.zeros_like(t)
rng = np.random.default_rng(0xBEEF)
K = 12
for k in range(1, K+1):
    amp = 1.0 / (k**1.2)
    # occasional harmonic dropouts
    mask = (rng.random(len(t)) > (0.02*k/12.0)).astype(float)
    x += amp * np.sin(k*phi) * mask

# Add broadband noise and a couple of impulsive events
x += 0.6*rng.standard_normal(len(t))
for spike_t in [2.0, 4.0, 5.5]:
    n0 = int(spike_t*fs)
    x[n0:n0+200] += 8.0*sig.windows.hann(200)

# Optional: mild bandpass to mimic hydrophone front-end (not required)
sos = sig.butter(2, [20, 10000], btype='bandpass', fs=fs, output='sos')
x = sig.sosfilt(sos, x)

# Detector params
stft_p = STFTParams(fs=fs, n_fft=4096, hop=1024, window='hann')
det_p = DetectorParams(
    f0_min=20, f0_max=150, f0_bins=300,
    max_harm=16, band_hz=2.0,
    viterbi_jump_cost=0.6, viterbi_accel_cost=0.1, viterbi_stay_reward=0.05,
    smoothing_alpha=0.25, comb_normalize=True
)
demon_p = DEMONParams(
    bpf_lo=40, bpf_hi=4000, envelope_lp=600, decim=20, n_fft=4096
)

# Run detection
res = detect_harmonics(x, stft_p, det_p, demon_p)

# Save f0 track to JSON
with open(outdir/"f0_track.json", "w") as f:
    json.dump({"t": res["t"].tolist(), "f0": res["f0_track"].tolist()}, f, indent=2)

# ---- Plots (no explicit colors set) ----
# 1) Spectrogram with f0 track
from matplotlib.ticker import MaxNLocator

fig1 = plt.figure(figsize=(10,6))
S_db = 10*np.log10(res["S"] + 1e-12)
plt.pcolormesh(res["t"], res["f"], S_db, shading="auto")
plt.plot(res["t"], res["f0_track"], linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("Whitened Spectrogram + f0 track")
plt.ylim([0, 2000])
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10, prune='upper'))
plt.tight_layout()
plt.savefig(outdir/"spectrogram_f0.png", dpi=150)
plt.show()

# 2) Comb map (time vs f0 candidates)
fig2 = plt.figure(figsize=(10,4))
plt.imshow(res["comb_map"].T, aspect="auto", origin="lower",
           extent=[res["t"][0], res["t"][-1], det_p.f0_min, det_p.f0_max])
plt.xlabel("Time [s]")
plt.ylabel("f0 candidate [Hz]")
plt.title("Comb score map (normalized)")
plt.tight_layout()
plt.savefig(outdir/"comb_map.png", dpi=150)
plt.show()

# 3) f0 vs ground truth (downsample gt to STFT frames for visual check)
# Ground truth per-frame average
t_frames = res["t"]
f0_gt_per_frame = np.interp(t_frames, t, f0_true)
fig3 = plt.figure(figsize=(10,3))
plt.plot(t_frames, f0_gt_per_frame, linewidth=1.5, label="True f0")
plt.plot(t_frames, res["f0_track"], linewidth=1.5, label="Tracked f0")
plt.xlabel("Time [s]")
plt.ylabel("Hz")
plt.title("f0: true vs tracked")
plt.legend()
plt.tight_layout()
plt.savefig(outdir/"f0_track_vs_truth.png", dpi=150)
plt.show()

# 4) Harmonic mask heatmap
fig4 = plt.figure(figsize=(10,3))
plt.imshow(res["harm_mask"].T, aspect="auto", origin="lower",
           extent=[t_frames[0], t_frames[-1], 1, det_p.max_harm])
plt.xlabel("Time [s]")
plt.ylabel("Harmonic index k")
plt.title("Detected harmonics mask")
plt.tight_layout()
plt.savefig(outdir/"harm_mask.png", dpi=150)
plt.show()

# 5) Optional DEMON peak track
if "demon_peak_hz" in res:
    fig5 = plt.figure(figsize=(10,3))
    plt.plot(t_frames, res["demon_peak_hz"])
    plt.xlabel("Time [s]")
    plt.ylabel("DEMON peak [Hz]")
    plt.title("DEMON peak per frame (if any)")
    plt.tight_layout()
    plt.savefig(outdir/"demon_peaks.png", dpi=150)
    plt.show()

print("Saved: spectrogram_f0.png, comb_map.png, f0_track_vs_truth.png, harm_mask.png, demon_peaks.png (optional), f0_track.json")
