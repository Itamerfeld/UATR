########################################################
# DPV ANALYSIS - SCOOTER EXPERIMENT 600m
########################################################

# imports

import os
import gpxpy
import time
import torch
import librosa
import pickle
import warnings
import pandas as pd
import networkx as nx

import numpy as np
from numpy import linalg as LA
from numpy import histogram2d

from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import find_peaks, butter, filtfilt, welch, get_window, hilbert
from scipy.ndimage import gaussian_filter
from scipy.io import wavfile
from scipy.stats import wasserstein_distance_nd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import write_image

import utils as ut


########################################################
# PARAMETERS
########################################################

# output
output_folder = 'results/plots'
to_show = True
to_save = True
plot_title = 'AUV leg 2 straight line'
output_file_name = 'auv_leg2_straight_line_detection_scores_plot'

# Spectrogram Parameters
nperseg = 16000
overlap = 0.5
window = 'hanning'
dc = 100
crop_freq = 8000
norm_size = 5

# S2G Parameters
s2g_quantization_levels = 3
s2g_mode = 'wasserstein'

# F0 Parameters
# f1 = 653  # Hz
# f2 = 850  # Hz
f1 = 441
f2 = 992

########################################################
# LOAD DATA
########################################################
# AUV data
data, fs = librosa.load('data/Itamar_AUV/Part1_StraightLine/IcListen6692/leg1_straight_line_2_8m_s_6922_827-837.wav')
distance_file = 'data/Itamar_AUV/Part1_StraightLine/IcListen6692/leg1_straight_line_2_8m_s_6922_827-837_distance_log.txt'

# data, fs = librosa.load('data/Itamar_AUV/Part1_StraightLine/IcListen6692/leg2_straight_line_1_8m_s_6922_840-857.wav')
# distance_file = 'data/Itamar_AUV/Part1_StraightLine/IcListen6692/leg2_straight_line_1_8m_s_6922_840-857_distance_log.txt'

df = pd.read_csv(distance_file, sep=' ')
df.head()
distances = df['distance_m'].values
t_distances = df['second'].values
print("completed loading WAV data and GPS data")


# BG Noise WAVs
noise_src_folder = 'data/croatia/2507_1_1000m'
file_list = os.listdir(noise_src_folder)
file_list = [f for f in file_list if f.endswith('.wav')]
file_list.sort()

bg_data = []
for file in file_list[:5]:
    _data, fs = librosa.load(os.path.join(noise_src_folder, file))
    bg_data.append(_data)
bg_data = np.concatenate(bg_data)
print(f"bg_data final shape: {bg_data.shape} of duration: {len(bg_data)/fs} seconds")

# Add BG Noise to data
noise_factor = 10
bg_in_shape = np.concatenate([bg_data, bg_data, bg_data, bg_data, bg_data, bg_data, bg_data, bg_data])
noisy_data = data + noise_factor * bg_in_shape[:len(data)]
print("completed adding noise to data")


########################################################
# FEATURE EXTRACTION - no noise
########################################################

welch_detector = ut.WelchDetector(fs, nperseg=nperseg, overlap=overlap, window=window, dc=dc, crop_freq=crop_freq, norm_size=norm_size)
talmon_detector = ut.TalmonDetector(fs, nperseg=nperseg, overlap=overlap, window=window, dc=dc, crop_freq=crop_freq, norm_size=norm_size)
s2g_detector = ut.S2GDetector(fs, nperseg=nperseg, overlap=overlap, window=window, dc=dc, crop_freq=crop_freq, mode=s2g_mode, quantization_levels=s2g_quantization_levels)


F, T, Sxx, phasogram = ut.calc_spectrogram(data, fs=fs, nperseg=nperseg, percent_overlap=overlap, window=window, remove_dc=dc, crop_freq=crop_freq)

f1_demon_scores = []
f2_demon_scores = []

f1_correlation_scores = []
f2_correlation_scores = []

f1_entropy_scores = []
f2_entropy_scores = []

f1_s2g_scores = []
f2_s2g_scores = []

step_size = 10 * fs
for si in np.arange(start=0, stop=(len(data)-60*fs), step=step_size):
    _data = data[si:si+60*fs]
    _time = (si + len(_data) / 2) // fs  # time corresponding to the center of the window

    # demon
    demon, _, _ = ut.demon_spectrum_narrowband(_data, fs, f1, 3, remove_dc=dc, crop_freq=crop_freq, agg='max')
    f1_demon_scores.append(demon)
    demon, _, _ = ut.demon_spectrum_narrowband(_data, fs, f2, 3, remove_dc=dc, crop_freq=crop_freq, agg='max')
    f2_demon_scores.append(demon)

    # correlation
    TD = talmon_detector.get_score_1fb(_data, f1)
    f1_correlation_scores.append(TD)
    TD = talmon_detector.get_score_1fb(_data, f2)
    f2_correlation_scores.append(TD)

    # entropy
    entropy, _, _ = ut.spectral_entropy_narrowband(_data, fs, nperseg, overlap, window, target_freq=f1, bandwidth=3, remove_dc=dc, crop_freq=crop_freq, agg='max')
    f1_entropy_scores.append(entropy)
    entropy, _, _ = ut.spectral_entropy_narrowband(_data, fs, nperseg, overlap, window, target_freq=f2, bandwidth=3, remove_dc=dc, crop_freq=crop_freq, agg='max')
    f2_entropy_scores.append(entropy)

    # # s2g score
    # K = s2g_detector.get_score_1fb(_data, f1)
    # f1_s2g_scores.append(K)
    # K = s2g_detector.get_score_1fb(_data, f2)
    # f2_s2g_scores.append(K)

    # s2g wasserstein score
    K, Kf = s2g_detector.get_feature_vector(_data)
    fix = np.where(Kf >= f1)[0][0]
    f1_s2g_scores.append(K[fix])
    fix = np.where(Kf >= f2)[0][0]
    f2_s2g_scores.append(K[fix])

detectors_taxis_clean = np.linspace(T[0], T[-1], len(f1_s2g_scores))

print("completed all feature extraction - no noise")

########################################################
# FEATURE EXTRACTION - with noise
########################################################

nF, nT, nSxx, nphasogram = ut.calc_spectrogram(noisy_data, fs=fs, nperseg=nperseg, percent_overlap=overlap, window=window, remove_dc=dc, crop_freq=crop_freq)

f1_td_scores_n = []
f2_td_scores_n = []

f1_s2g_scores_n = []
f2_s2g_scores_n = []

f1_demon_scores_n = []
f2_demon_scores_n = []

f1_entropy_scores_n = []
f2_entropy_scores_n = []

step_size = 10 * fs
for si in np.arange(start=0, stop=(len(data)-60*fs), step=step_size):
    _data = noisy_data[si:si+60*fs]
    _time = (si + len(_data) / 2) // fs  # time corresponding to the center of the window

    # correlation
    TD = talmon_detector.get_score_1fb(_data, f1)
    f1_td_scores_n.append(TD)
    TD = talmon_detector.get_score_1fb(_data, f2)
    f2_td_scores_n.append(TD)

    # # s2g score
    # K = s2g_detector.get_score_1fb(_data, f1)
    # f1_s2g_scores_n.append(K)
    # K = s2g_detector.get_score_1fb(_data, f2)
    # f2_s2g_scores_n.append(K)

    # demon
    demon, _, _ = ut.demon_spectrum_narrowband(_data, fs, f1, 3, remove_dc=dc, crop_freq=crop_freq, agg='max')
    f1_demon_scores_n.append(demon)
    demon, _, _ = ut.demon_spectrum_narrowband(_data, fs, f2, 3, remove_dc=dc, crop_freq=crop_freq, agg='max')
    f2_demon_scores_n.append(demon)

    # entropy
    entropy, _, _ = ut.spectral_entropy_narrowband(_data, fs, nperseg, overlap, window, target_freq=f1, bandwidth=3, remove_dc=dc, crop_freq=crop_freq, agg='max')
    f1_entropy_scores_n.append(entropy)
    entropy, _, _ = ut.spectral_entropy_narrowband(_data, fs, nperseg, overlap, window, target_freq=f2, bandwidth=3, remove_dc=dc, crop_freq=crop_freq, agg='max')
    f2_entropy_scores_n.append(entropy)

    # s2g wasserstein score
    K, Kf = s2g_detector.get_feature_vector(_data)
    fix = np.where(Kf >= f1)[0][0]
    f1_s2g_scores_n.append(K[fix])
    fix = np.where(Kf >= f2)[0][0]
    f2_s2g_scores_n.append(K[fix])

detectors_taxis_noisy = np.linspace(nT[0], nT[-1], len(f1_s2g_scores_n))

print("completed all feature extraction - with noise")

########################################################
# PLOT RESULTS
########################################################

fig = make_subplots(rows=6,
                    cols=2, 
                    row_heights=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1], 
                    vertical_spacing=0.01, 
                    shared_xaxes=True,
                    column_titles=["No Noise", "SNR 1:10"],
                    row_titles=["Spectrogram", "Distance", "Demon", "Correlation", "Entropy", "S2G"])

# no noise
_ds = max(1, Sxx.shape[1] // 1000)
fig.add_trace(go.Heatmap(z=Sxx[:, ::_ds], x=T[::_ds], y=F, colorscale='viridis', showlegend=False, showscale=False), row=1, col=1)
fig.add_trace(go.Scatter(x=t_distances, y=distances, name='Distance', marker_color='black', showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=f1_demon_scores, name=f'Demon f={f1} Hz', marker_color='blue', showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=f2_demon_scores, name=f'Demon f={f2} Hz', marker_color='green', showlegend=False), row=3, col=1)
six = np.where(detectors_taxis_clean >= 0)[0][0]
eix = np.where(detectors_taxis_clean >= 167)[0][0]
fig.add_vrect(x0=detectors_taxis_clean[six], x1=detectors_taxis_clean[eix], line_dash="dash", line_color="red", line_width=2, row=3, col=1)
# fig.add_vline(x=detectors_taxis_clean[dix], line_dash="dash", line_color="red", line_width=2, row=3, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=f1_correlation_scores, name=f'Correlation f={f1} Hz', marker_color='blue', showlegend=False), row=4, col=1)
fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=f2_correlation_scores, name=f'Correlation f={f2} Hz', marker_color='green', showlegend=False), row=4, col=1)
six = np.where(detectors_taxis_clean >= 0)[0][0]
eix = np.where(detectors_taxis_clean >= 167)[0][0]
fig.add_vrect(x0=detectors_taxis_clean[six], x1=detectors_taxis_clean[eix], line_dash="dash", line_color="red", line_width=2, row=4, col=1)
# fig.add_vline(x=detectors_taxis_clean[dix], line_dash="dash", line_color="red", line_width=2, row=4, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=f1_entropy_scores, name=f'Entropy f={f1} Hz', marker_color='blue', showlegend=False), row=5, col=1)
fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=f2_entropy_scores, name=f'Entropy f={f2} Hz', marker_color='green', showlegend=False), row=5, col=1)
six = np.where(detectors_taxis_clean >= 0)[0][0]
eix = np.where(detectors_taxis_clean >= 167)[0][0]
fig.add_vrect(x0=detectors_taxis_clean[six], x1=detectors_taxis_clean[eix], line_dash="dash", line_color="red", line_width=2, row=5, col=1)
# fig.add_vline(x=detectors_taxis_clean[dix], line_dash="dash", line_color="red", line_width=2, row=5, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=f1_s2g_scores, name=f'Score f={f1} Hz', marker_color='blue'), row=6, col=1)
fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=f2_s2g_scores, name=f'Score f={f2} Hz', marker_color='green'), row=6, col=1)
six = np.where(detectors_taxis_clean >= 0)[0][0]
eix = np.where(detectors_taxis_clean >= 167)[0][0]
fig.add_vrect(x0=detectors_taxis_clean[six], x1=detectors_taxis_clean[eix], line_dash="dash", line_color="red", line_width=2, row=6, col=1)
# fig.add_vline(x=detectors_taxis_clean[dix], line_dash="dash", line_color="red", line_width=2, row=6, col=1)


# with noise
fig.add_trace(go.Heatmap(z=nSxx[:, ::_ds], x=nT[::_ds], y=nF, colorscale='viridis', showlegend=False, showscale=False), row=1, col=2)
fig.add_trace(go.Scatter(x=t_distances, y=distances, name='Distance', marker_color='black', showlegend=False), row=2, col=2)

fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=f1_demon_scores_n, name=f'Demon f={f1} Hz', marker_color='blue', showlegend=False), row=3, col=2)
fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=f2_demon_scores_n, name=f'Demon f={f2} Hz', marker_color='green', showlegend=False), row=3, col=2)
# dix = np.where(detectors_taxis_noisy >= 804)[0][0]
# fig.add_vline(x=detectors_taxis_noisy[dix], line_dash="dash", line_color="red", line_width=2, row=3, col=2)

fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=f1_td_scores_n, name=f'Correlation f={f1} Hz', marker_color='blue', showlegend=False), row=4, col=2)
fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=f2_td_scores_n, name=f'Correlation f={f2} Hz', marker_color='green', showlegend=False), row=4, col=2)
six = np.where(detectors_taxis_noisy >= 0)[0][0]
eix = np.where(detectors_taxis_noisy >= 167)[0][0]
fig.add_vrect(x0=detectors_taxis_noisy[six], x1=detectors_taxis_noisy[eix], line_dash="dash", line_color="red", line_width=2, row=4, col=2)
# fig.add_vline(x=detectors_taxis_noisy[dix], line_dash="dash", line_color="red", line_width=2, row=4, col=2)

fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=f1_entropy_scores_n, name=f'Entropy f={f1} Hz', marker_color='blue', showlegend=False), row=5, col=2)
fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=f2_entropy_scores_n, name=f'Entropy f={f2} Hz', marker_color='green', showlegend=False), row=5, col=2)
six = np.where(detectors_taxis_noisy >= 100)[0][0]
eix = np.where(detectors_taxis_noisy >= 167)[0][0]
fig.add_vrect(x0=detectors_taxis_noisy[six], x1=detectors_taxis_noisy[eix], line_dash="dash", line_color="red", line_width=2, row=5, col=2)
# fig.add_vline(x=detectors_taxis_noisy[dix], line_dash="dash", line_color="red", line_width=2, row=5, col=2)

fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=f1_s2g_scores_n, name=f'Score f={f1} Hz', marker_color='blue', showlegend=False), row=6, col=2)
fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=f2_s2g_scores_n, name=f'Score f={f2} Hz', marker_color='green', showlegend=False), row=6, col=2)
six = np.where(detectors_taxis_noisy >= 0)[0][0]
eix = np.where(detectors_taxis_noisy >= 167)[0][0]
fig.add_vrect(x0=detectors_taxis_noisy[six], x1=detectors_taxis_noisy[eix], line_dash="dash", line_color="red", line_width=2, row=6, col=2)
# fig.add_vline(x=detectors_taxis_noisy[dix], line_dash="dash", line_color="red", line_width=2, row=6, col=2)


# update axes
fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
fig.update_yaxes(title_text="Dist. (m)", row=2, col=1)
fig.update_yaxes(title_text="Score", row=3, col=1)
fig.update_yaxes(title_text="Score", row=4, col=1)
fig.update_yaxes(title_text="Score", row=5, col=1)
fig.update_yaxes(title_text="Score", row=6, col=1)
fig.update_xaxes(title_text="Time (s)", row=6, col=1)

fig.update_layout(height=800, width=1400, title_text=f"{plot_title}, scores for f0 = {f1}, {f2} Hz")
fig.update_annotations(font_size=12)

if to_show:
    fig.show(renderer="browser")
if to_save:
    fig.write_image(os.path.join(output_folder, f"{output_file_name}.pdf"))

print("DONE")
