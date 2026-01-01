
# imports and settings

from dash import Dash, dcc, html
from dash import Input, Output

import os
import time
import pickle
import warnings
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import numpy as np
from numpy import linalg as LA
from numpy import histogram2d

from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import find_peaks, butter, filtfilt, welch
from scipy.ndimage import gaussian_filter
from scipy.io import wavfile

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import utils as ut

## =================================
# dpv1 - croatia
data_file = "data/dpv1_1m.wav"
dpv1_fs, dpv1_data = wavfile.read(data_file)
dpv1_slice = dpv1_data[int(35*dpv1_fs):int(45*dpv1_fs)]

# dpv2 - haifa
data_file = "data/dpv2_1m.wav"
dpv2_fs, dpv2_data = wavfile.read(data_file)
dpv2_slice = dpv2_data[int(15*dpv2_fs):int(25*dpv2_fs)]

# motorboat
data_file = "data/motorboat_1m.wav"
motorboat_fs, motorboat_data = wavfile.read(data_file)
motorboat_slice = motorboat_data[int(5*motorboat_fs):int(15*motorboat_fs)]

# large ship
data_file = "data/large_ship_1m.wav"
large_ship_fs, large_ship_data = wavfile.read(data_file)
large_ship_slice = large_ship_data[int(5*large_ship_fs):int(15*large_ship_fs)]

# BG noise
data_file = "data/bg_noise_1m.wav"
bg_fs, bg_data = wavfile.read(data_file)
bg_slice = bg_data[int(5*bg_fs):int(15*bg_fs)]

# organize samples
names = [
    "dpv1", 
    "dpv2", 
    "motorboat", 
    "large_ship", 
    "bg_noise"
    ]

fss = {
    "dpv1": dpv1_fs,
    "dpv2": dpv2_fs,
    "motorboat": motorboat_fs,
    "large_ship": large_ship_fs,
    "bg_noise": bg_fs
    }

all_data = {
    "dpv1": dpv1_data,
    "dpv2": dpv2_data,
    "motorboat": motorboat_data,
    "large_ship": large_ship_data,
    "bg_noise": bg_data
    }

slices = {
    "dpv1": dpv1_slice,
    "dpv2": dpv2_slice,
    "motorboat": motorboat_slice,
    "large_ship": large_ship_slice,
    "bg_noise": bg_slice
    }

## =================================

# parameters and processing loop
height = 800
width = 1400
font_size = 16
fft_nperseg = 64536
percent_overlap = 0.5
window = 'hamming'
remove_dc = 20
crop_freq = 4000
normalization_window_size = 17
detection_threshold = 3
p_gap = 0.2
band_width = normalization_window_size // 2
sigma = 0.1
p_scale = 1
slice_len = 10  # seconds

## =================================

# Create Dash app
app = Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Spectrogram and Welch PSD with Detected Peaks"),
    html.Label("Select Signal:"),
    dcc.Dropdown(
        id='name-dropdown',
        options=[{'label': n, 'value': n} for n in names],
        value=names[0],
        clearable=False
    ),
    dcc.Graph(id='spectrogram-graph')
])

# Define callback to update figure based on dropdown selection
@app.callback(
    Output('spectrogram-graph', 'figure'),
    Input('name-dropdown', 'value')
)
def update_figure(selected_name):
    fs_selected = fss[selected_name]
    data_selected = all_data[selected_name]
    
    F_sel, T_sel, Sxx_sel, pxx_sel, peaks_sel, track_ixs_sel, tracks_sel = ut.calc_tracks(
        data_selected, 
        fs=fs_selected, 
        fft_nperseg=fft_nperseg, 
        percent_overlap=percent_overlap, 
        window=window, 
        remove_dc=remove_dc, 
        crop_freq=crop_freq, 
        normalization_window_size=normalization_window_size, 
        detection_threshold=detection_threshold, 
        p_gap=p_gap, 
        band_width=band_width, 
        sigma=sigma, 
        p_scale=p_scale
    )
    
    fig_new = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.8, 0.2], 
                            horizontal_spacing=0.15, vertical_spacing=0.15, 
                            column_titles=("Spectrogram", "Welch Power Spectral Density"))
    fig_new.add_trace(go.Heatmap(x=T_sel, y=F_sel, z=Sxx_sel.T, colorscale='Viridis', showscale=False), row=1, col=1)
    fig_new.add_trace(go.Scatter(x=pxx_sel, y=F_sel, mode='lines', name='Welch PSD', showlegend=False, 
                                 line=dict(color='blue', width=2)), row=1, col=2)
    fig_new.add_trace(go.Scatter(x=pxx_sel[peaks_sel], y=F_sel[peaks_sel], mode='markers', name='Detected Peaks', 
                                 marker=dict(color='cyan', size=8), showlegend=False), row=1, col=2)
    
    for track in track_ixs_sel:
        fig_new.add_trace(go.Scatter(x=T_sel, y=F_sel[track], mode='markers', 
                                     line=dict(color='cyan', width=1), name='Viterbi Track', 
                                     showlegend=False), row=1, col=1)
    
    fig_new.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig_new.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig_new.update_yaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig_new.update_xaxes(title_text="Power/Frequency (dB/Hz)", row=1, col=2)
    fig_new.update_layout(height='1000', width='1400', 
                            title_text=f"Spectrogram and Welch PSD with Detected Peaks of {selected_name}", 
                            font=dict(size=font_size),
                            margin=dict(l=20, r=20, t=60, b=20))
    
    return fig_new

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)
