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
plot_title = 'AUV experiment'
output_file_name = 'AUV_detection_scores_plot_for_presentation'

# SpectrogramParameters
nperseg = 2000
overlap = 0.25
window = 'hanning'
dc = 20
crop_freq = 8000
norm_size = 9

# S2G Parameters
s2g_quantization_levels = 5

# F0 Parameters
f1 = 997
f2 = 5545

########################################################
# LOAD DATA
########################################################
# AUV data
# data, fs = librosa.load('data/Itamar_AUV/Part1_StraightLine/IcListen6692/leg1_straight_line_2_8m_s_6922_827-837.wav')
# distance_file = 'data/Itamar_AUV/Part1_StraightLine/IcListen6692/leg1_straight_line_2_8m_s_6922_827-837_distance_log.txt'

data, fs = librosa.load('data/Itamar_AUV/Part1_StraightLine/IcListen6692/leg2_straight_line_1_8m_s_6922_840-857.wav')
distance_file = 'data/Itamar_AUV/Part1_StraightLine/IcListen6692/leg2_straight_line_1_8m_s_6922_840-857_distance_log.txt'

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
s2g_laplacian_detector = ut.S2GDetector(fs, nperseg=nperseg, overlap=overlap, window=window, dc=dc, crop_freq=crop_freq, mode='laplacian', quantization_levels=s2g_quantization_levels)
s2g_wasserstein_detector = ut.S2GDetector(fs, nperseg=nperseg, overlap=overlap, window=window, dc=dc, crop_freq=crop_freq, mode='wasserstein', quantization_levels=s2g_quantization_levels)


F, T, Sxx, phasogram = ut.calc_spectrogram(data, fs=fs, nperseg=nperseg, percent_overlap=overlap, window=window, remove_dc=dc, crop_freq=crop_freq)

f1_demon_scores = []
f2_demon_scores = []

f1_correlation_scores = []
f2_correlation_scores = []

f1_s2g_laplacian_scores = []
f2_s2g_laplacian_scores = []

f1_s2g_wasserstein_scores = []
f2_s2g_wasserstein_scores = []

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

    # s2g Laplacian score
    all_k, all_f = s2g_laplacian_detector.get_feature_vector(_data)
    f1_idx = np.where(all_f >= f1)[0][0]
    f2_idx = np.where(all_f >= f2)[0][0]
    f1_s2g_laplacian_scores.append(all_k[f1_idx])
    f2_s2g_laplacian_scores.append(all_k[f2_idx])

    # s2g Wasserstein score
    all_k, all_f = s2g_wasserstein_detector.get_feature_vector(_data)
    f1_idx = np.where(all_f >= f1)[0][0]
    f2_idx = np.where(all_f >= f2)[0][0]
    f1_s2g_wasserstein_scores.append(all_k[f1_idx])
    f2_s2g_wasserstein_scores.append(all_k[f2_idx])
    # K = s2g_wasserstein_detector.get_score_1fb(_data, f1)
    # f1_s2g_wasserstein_scores.append(K)
    # K = s2g_wasserstein_detector.get_score_1fb(_data, f2)
    # f2_s2g_wasserstein_scores.append(K)

detectors_taxis_clean = np.linspace(T[0], T[-1], len(f1_s2g_laplacian_scores))

print("completed all feature extraction - no noise")

########################################################
# FEATURE EXTRACTION - with noise
########################################################

nF, nT, nSxx, nphasogram = ut.calc_spectrogram(noisy_data, fs=fs, nperseg=nperseg, percent_overlap=overlap, window=window, remove_dc=dc, crop_freq=crop_freq)

f1_demon_scores_n = []
f2_demon_scores_n = []

f1_correlation_scores_n = []
f2_correlation_scores_n = []

f1_s2g_laplacian_scores_n = []
f2_s2g_laplacian_scores_n = []

f1_s2g_wasserstein_scores_n = []
f2_s2g_wasserstein_scores_n = []

step_size = 10 * fs
for si in np.arange(start=0, stop=(len(data)-60*fs), step=step_size):
    _data = noisy_data[si:si+60*fs]
    _time = (si + len(_data) / 2) // fs  # time corresponding to the center of the window

    # demon
    demon, _, _ = ut.demon_spectrum_narrowband(_data, fs, f1, 3, remove_dc=dc, crop_freq=crop_freq, agg='max')
    f1_demon_scores_n.append(demon)
    demon, _, _ = ut.demon_spectrum_narrowband(_data, fs, f2, 3, remove_dc=dc, crop_freq=crop_freq, agg='max')
    f2_demon_scores_n.append(demon)

    # correlation
    TD = talmon_detector.get_score_1fb(_data, f1)
    f1_correlation_scores_n.append(TD)
    TD = talmon_detector.get_score_1fb(_data, f2)
    f2_correlation_scores_n.append(TD)

    # s2g Laplacian score
    all_k, all_f = s2g_laplacian_detector.get_feature_vector(_data)
    f1_idx = np.where(all_f >= f1)[0][0]
    f2_idx = np.where(all_f >= f2)[0][0]
    f1_s2g_laplacian_scores_n.append(all_k[f1_idx])
    f2_s2g_laplacian_scores_n.append(all_k[f2_idx])
    # K = s2g_laplacian_detector.get_score_1fb(_data, f1)
    # f1_s2g_laplacian_scores_n.append(K)
    # K = s2g_laplacian_detector.get_score_1fb(_data, f2)
    # f2_s2g_laplacian_scores_n.append(K)

    # s2g Wasserstein score
    all_k, all_f = s2g_wasserstein_detector.get_feature_vector(_data)
    f1_idx = np.where(all_f >= f1)[0][0]
    f2_idx = np.where(all_f >= f2)[0][0]
    f1_s2g_wasserstein_scores_n.append(all_k[f1_idx])
    f2_s2g_wasserstein_scores_n.append(all_k[f2_idx])
    # K = s2g_wasserstein_detector.get_score_1fb(_data, f1)
    # f1_s2g_wasserstein_scores_n.append(K)
    # K = s2g_wasserstein_detector.get_score_1fb(_data, f2)
    # f2_s2g_wasserstein_scores_n.append(K)

detectors_taxis_noisy = np.linspace(nT[0], nT[-1], len(f1_s2g_laplacian_scores_n))

print("completed all feature extraction - with noise")

########################################################
# PLOT RESULTS
########################################################

column_title_texts = ["No Noise", "SNR 1:10"]
# row_title_texts = ["Spectrogram", "Distance", "Demon", "Correlation", "S2G Laplacian", "S2G Wasserstein"]

fig = make_subplots(rows=6,
                    cols=2, 
                    row_heights=[0.3, 0.1, 0.15, 0.15, 0.15, 0.15], 
                    vertical_spacing=0.01, 
                    shared_xaxes=True,
                    column_titles=column_title_texts)  #,
                    # row_titles=row_title_texts)

# no noise
_ds = max(1, Sxx.shape[1] // 1000)
fig.add_trace(go.Heatmap(z=Sxx[:, ::_ds], x=T[::_ds], y=F, colorscale='viridis', showlegend=False, showscale=False), row=1, col=1)
fig.add_trace(go.Scatter(x=t_distances, y=distances, name='Distance', marker_color='black', showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=f1_demon_scores, name=f'Demon f={f1} Hz', marker_color='blue', showlegend=True), row=3, col=1)
fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=f2_demon_scores, name=f'Demon f={f2} Hz', marker_color='green', showlegend=True), row=3, col=1)
dix = np.where(detectors_taxis_clean >= 20)[0][0]
fig.add_vline(x=detectors_taxis_clean[dix], line_dash="dash", line_color="red", line_width=2, row=3, col=1, showlegend=True, name="first detection")

fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=f1_correlation_scores, name=f'Correlation f={f1} Hz', marker_color='blue', showlegend=False), row=4, col=1)
fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=f2_correlation_scores, name=f'Correlation f={f2} Hz', marker_color='green', showlegend=False), row=4, col=1)
dix = np.where(detectors_taxis_clean >= 20)[0][0]
fig.add_vline(x=detectors_taxis_clean[dix], line_dash="dash", line_color="red", line_width=2, row=4, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=f1_s2g_laplacian_scores, name=f'Laplacian f={f1} Hz', marker_color='blue', showlegend=False), row=5, col=1)
fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=f2_s2g_laplacian_scores, name=f'Laplacian f={f2} Hz', marker_color='green', showlegend=False), row=5, col=1)
dix = np.where(detectors_taxis_clean >= 20)[0][0]
fig.add_vline(x=detectors_taxis_clean[dix], line_dash="dash", line_color="red", line_width=2, row=5, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=f1_s2g_wasserstein_scores, name=f'Wasserstein f={f1} Hz', marker_color='blue', showlegend=False), row=6, col=1)
fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=f2_s2g_wasserstein_scores, name=f'Wasserstein f={f2} Hz', marker_color='green', showlegend=False), row=6, col=1)
dix = np.where(detectors_taxis_clean >= 20)[0][0]
fig.add_vline(x=detectors_taxis_clean[dix], line_dash="dash", line_color="red", line_width=2, row=6, col=1)


# with noise
fig.add_trace(go.Heatmap(z=nSxx[:, ::_ds], x=nT[::_ds], y=nF, colorscale='viridis', showlegend=False, showscale=False), row=1, col=2)
fig.add_trace(go.Scatter(x=t_distances, y=distances, name='Distance', marker_color='black', showlegend=False), row=2, col=2)

fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=f1_demon_scores_n, name=f'Demon f={f1} Hz', marker_color='blue', showlegend=False), row=3, col=2)
fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=f2_demon_scores_n, name=f'Demon f={f2} Hz', marker_color='green', showlegend=False), row=3, col=2)
dix = np.where(detectors_taxis_noisy >= 20)[0][0]
fig.add_vline(x=detectors_taxis_noisy[dix], line_dash="dash", line_color="red", line_width=2, row=3, col=2)

fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=f1_correlation_scores_n, name=f'Correlation f={f1} Hz', marker_color='blue', showlegend=False), row=4, col=2)
fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=f2_correlation_scores_n, name=f'Correlation f={f2} Hz', marker_color='green', showlegend=False), row=4, col=2)
dix = np.where(detectors_taxis_noisy >= 20)[0][0]
fig.add_vline(x=detectors_taxis_noisy[dix], line_dash="dash", line_color="red", line_width=2, row=4, col=2)

fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=f1_s2g_laplacian_scores_n, name=f'Laplacian f={f1} Hz', marker_color='blue', showlegend=False), row=5, col=2)
fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=f2_s2g_laplacian_scores_n, name=f'Laplacian f={f2} Hz', marker_color='green', showlegend=False), row=5, col=2)
dix = np.where(detectors_taxis_noisy >= 20)[0][0]
fig.add_vline(x=detectors_taxis_noisy[dix], line_dash="dash", line_color="red", line_width=2, row=5, col=2)

fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=f1_s2g_wasserstein_scores_n, name=f'Wasserstein f={f1} Hz', marker_color='blue', showlegend=False), row=6, col=2)
fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=f2_s2g_wasserstein_scores_n, name=f'Wasserstein f={f2} Hz', marker_color='green', showlegend=False), row=6, col=2)
dix = np.where(detectors_taxis_noisy >= 20)[0][0]
fig.add_vline(x=detectors_taxis_noisy[dix], line_dash="dash", line_color="red", line_width=2, row=6, col=2)


# update axes
# fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
# fig.update_yaxes(title_text="Dist. (m)", row=2, col=1)
# fig.update_yaxes(title_text="DEMON", row=3, col=1)
# fig.update_yaxes(title_text="Correlation", row=4, col=1)
# fig.update_yaxes(title_text="S2G Laplacian", row=5, col=1)
# fig.update_yaxes(title_text="S2G Wasserstein", row=6, col=1)
# fig.update_xaxes(title_text="Time (s)", row=6, col=1)
# fig.update_xaxes(title_text="Time (s)", row=6, col=2)

fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=12))

fig.update_annotations(font_size=20)

fig.update_layout(
    height=800, width=1400,
    font=dict(size=20),
    title_font=dict(size=20),
    margin=dict(t=140, r=140),   # leave room for lifted column titles + shifted row titles
)

# Match column/row title offsets and bold them.
# Column titles sit above row 1 at y = 1 + offset; row titles sit to the
# right of the last column at x = 1 + offset (in paper coords).
title_offset = 0.06
for ann in fig.layout.annotations:
    ann.font = dict(size=20)
    if ann.text in column_title_texts:
        ann.text = f"<b>{ann.text}</b>"
        ann.y = 1 + title_offset
        ann.yanchor = 'bottom'
    # elif ann.text in row_title_texts:
    #     ann.text = f"<b>{ann.text}</b>"
    #     ann.x = 1 + title_offset
    #     ann.xanchor = 'left'


if to_save:
    fig.write_image(os.path.join(output_folder, f"{output_file_name}.pdf"))
if to_show:
    fig.show(renderer="browser")

print("DONE")
