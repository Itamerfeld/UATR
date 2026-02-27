# imports and settings

import os
from statistics import mode
import time
import pickle
import librosa
import warnings
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import numpy as np
from numpy import linalg as LA
from numpy import histogram2d

from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import find_peaks, butter, filtfilt, sosfiltfilt, welch
from scipy.ndimage import gaussian_filter
from scipy.io import wavfile
from scipy.stats import wasserstein_distance_nd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# import utils as ut
# %load_ext autoreload
# %autoreload 2

# do not show warnings
warnings.filterwarnings("ignore")

print("Imports complete.")

# plotting parameters
height = 800
width = 1400
font_size = 16

print(f"Settings: height={height}, width={width}, font_size={font_size}")

## ====================
## DETECTORS
## ====================

class WelchDetector:
    def __init__(self, fs, nperseg, overlap, window, dc, crop_freq, norm_size, default_distance=3):
        self.fs = fs
        self.nperseg = nperseg
        self.overlap = overlap
        self.window = window
        self.dc = dc
        self.crop_freq = crop_freq
        self.norm_size = norm_size
        self.default_distance = default_distance

    def detect(self, rx, threshold):
        pxx, F = self.get_feature_vector(rx)
        # TH = np.mean(pxx) + threshold * np.std(pxx)
        TH = threshold
        detections = find_peaks(pxx, height=TH, distance=self.default_distance)[0]
        is_detected = len(detections) > 0
        return is_detected, detections, (pxx, F)
    
    def get_feature_vector(self, rx):
        F, T, Sxx, phasogram = calc_spectrogram(rx, self.fs, nperseg=self.nperseg, percent_overlap=self.overlap, window=self.window, remove_dc=self.dc, crop_freq=self.crop_freq)
        pxx = calc_welch_from_spectrogram(Sxx, normalization_window_size=self.norm_size)
        return pxx, F
        
class S2GDetector:
    def __init__(self, fs, nperseg, overlap, window, dc, crop_freq, quantization_levels, mode="wasserstein", default_distance=2):
        self.fs = fs
        self.nperseg = nperseg
        self.overlap = overlap
        self.window = window
        self.dc = dc
        self.crop_freq = crop_freq
        self.quantization_levels = quantization_levels
        self.mode = mode
        self.default_distance = default_distance

    def detect(self, rx, threshold):
        K, F = self.get_feature_vector(rx)
        # TH = np.mean(K) + (2+threshold) * np.std(K)
        TH = threshold
        detections = find_peaks(K, height=TH, distance=self.default_distance)[0]
        is_detected = len(detections) > 0
        return is_detected, detections, (K, F)
    
    def get_feature_vector(self, rx):
        F, T, Sxx, phasogram = calc_spectrogram(rx, self.fs, nperseg=self.nperseg, percent_overlap=self.overlap, window=self.window, remove_dc=self.dc, crop_freq=self.crop_freq)
        K = get_all_Ks(phasogram, F, n_levels=self.quantization_levels, mode=self.mode)
        if self.mode == "edge_count":
            K = 1 - K
        if self.mode == "laplacian":
            # K = (K - np.min(K)) / (np.max(K) - np.min(K))  # normalize to [0,1]
            # K = np.mean(K) - K
            K = K / np.mean(K)
            K = np.mean(K) - K
        if self.mode == "wasserstein":
            K = (K - np.min(K)) / (np.max(K) - np.min(K))  # normalize to [0,1]
        return K, F

## ====================
## S2G UTILITIES
## ====================

def normalize_data(x):
    x = x / 2  # this is necessary to avoid overflow in some cases due to super large values (probably a bug somewhere else)
    xmin = np.min(x)
    xmax = np.max(x)
    normalized_data = (x - xmin) / (xmax - xmin) if xmax != xmin else 0
    return normalized_data

def quantize_data(x, n_levels):
    x_quantized = np.floor(x * n_levels).astype(int)
    x_quantized[x_quantized == n_levels] = n_levels - 1  # Handle edge case
    return x_quantized

def get_s2g_transition_matrix(x_quantized, n_levels):
    transitions = np.zeros((n_levels, n_levels), dtype=int)
    for i in range(len(x_quantized)-1):
        transitions[x_quantized[i], x_quantized[i+1]] += 1
    return transitions

def get_s2g(x, n_levels):
    x = normalize_data(x)
    x = quantize_data(x, n_levels)
    transitions = get_s2g_transition_matrix(x, n_levels)
    return transitions

def get_s2g_edges(x_quantized):
    edges = []
    for i in range(1, len(x_quantized)):
        if x_quantized[i] != x_quantized[i-1]:
            edges.append((x_quantized[i-1], x_quantized[i]))
    return edges

def get_s2g_graph(x, n_levels):
    x = normalize_data(x)
    x = quantize_data(x, n_levels)
    nodes = np.unique(x)
    edges = get_s2g_edges(x)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def get_K(transition_matrix, mode='wasserstein', uniform_M=None):
    if mode == 'wasserstein':
        if uniform_M is not None:
            uniform_dist = uniform_M
        else:
            uniform_dist = np.ones(transition_matrix.shape) / transition_matrix.size
        K = wasserstein_distance_nd(transition_matrix, uniform_dist)
    elif mode == 'edge_count':    
        edge_count = np.count_nonzero(transition_matrix)
        K = edge_count / transition_matrix.size
    elif mode == 'laplacian':
        D = np.diag(np.sum(transition_matrix, axis=1))
        L = D - transition_matrix
        eigenvalues = LA.eigvals(L)  # only eigenvalues; avoids mixed tuple
        eigenvalues = np.real(eigenvalues)
        eigenvalues = np.sort(eigenvalues)
        # use Fiedler value when available; fallback to first eigenvalue for 1x1
        K = eigenvalues[1] if eigenvalues.size > 1 else eigenvalues[0]
    elif mode == 'all':
        K1 = get_K(transition_matrix, mode='wasserstein')
        K2 = get_K(transition_matrix, mode='edge_count')
        K3 = get_K(transition_matrix, mode='laplacian')
        K = (K1, K2, K3)
    else:
        raise ValueError(f"Unknown K calculation mode: {mode}")
    return K

def get_all_Ks(phase_matrix, frequencies, n_levels, mode='wasserstein'):
    Ks = []
    if mode == 'wasserstein':
        uniform_M = np.ones((n_levels, n_levels)) / (n_levels**2)
    else:
        uniform_M = None
    for f_idx, f in enumerate(frequencies):
        x = phase_matrix[f_idx, :]
        transition_matrix = get_s2g(x, n_levels=n_levels)
        K = get_K(transition_matrix, mode=mode, uniform_M=uniform_M)
        Ks.append(K)
    Ks = np.array(Ks[:-1])  # exclude last frequency if it's the Nyquist frequency (which may be less reliable)
    return Ks

## ====================
## SIMULATION UTILITIES
## ====================

def pink_noise(N):
    # Voss-McCartney algorithm
    n = int(np.ceil(np.log2(N)))
    array = np.random.randn(n, N)
    array = np.cumsum(array, axis=0)
    weights = 1 / (2 ** np.arange(n))
    pink = np.dot(weights, array)
    return pink[:N]

def white_noise(N):
    return np.random.rand(N)

def simulate_raw_signal(f0, fs, duration):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * f0 * t)
    return sig

# def add_noise_to_signal(signal, snr_db, noise_type='white'):

#     if noise_type == 'white':
#         noise = white_noise(len(signal))
#     elif noise_type == 'pink':
#         noise = pink_noise(len(signal))
#     else:
#         raise ValueError(f"Unknown noise type: {noise_type}")
    
#     signal_power = np.mean(signal ** 2)
#     noise_power = np.mean(noise ** 2)
#     desired_noise_power = signal_power / (10 ** (snr_db / 10))
#     scaling_factor = np.sqrt(desired_noise_power / noise_power)
#     noise = noise * scaling_factor
#     noisy_signal = signal + noise
#     return noisy_signal

def add_noise_to_signal(signal, snr_db, fs, signal_bw, noise_type='white'):
    """
    Adds broad-band noise to a narrow-band signal based on an in-band SNR.
    """
    if noise_type == 'white':
        noise = white_noise(len(signal)) # Assuming this returns unscaled white noise
    elif noise_type == 'pink':
        noise = pink_noise(len(signal))  # Assuming this returns unscaled pink noise
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # 1. Calculate actual signal power
    signal_power = np.mean(signal ** 2)
    
    # 2. Calculate the desired noise power *within the signal's bandwidth*
    desired_noise_power_in_band = signal_power / (10 ** (snr_db / 10))
    
    # 3. Scale up to total noise power across the entire Nyquist band (fs/2)
    # Note: This linear scaling assumes a flat noise spectrum (White noise). 
    bandwidth_ratio = (fs / 2) / signal_bw
    total_desired_noise_power = desired_noise_power_in_band * bandwidth_ratio
    
    # 4. Calculate current noise power and scale
    current_noise_power = np.mean(noise ** 2)
    scaling_factor = np.sqrt(total_desired_noise_power / current_noise_power)
    
    scaled_noise = noise * scaling_factor
    noisy_signal = signal + scaled_noise
    
    return noisy_signal

## ====================
## SIGNAL PROCESSING UTILITIES
## ====================

def rw_normalization(x, window_size=17):
    if window_size % 2 == 0:
        window_size += 1  # make it odd
    normalization_kernel = np.ones((window_size,)) / (window_size-1)
    normalization_kernel[window_size // 2] = 0
    smooth_x = signal.convolve(x, normalization_kernel, mode='same')
    ret = x / (smooth_x + 1e-10)

    # handle edge cases
    ret[:window_size//2] = ret[window_size//2]
    ret[-window_size//2:] = ret[-window_size//2-1]
    return ret

def rw_normalization2d(x, window_size=17):
    if window_size % 2 == 0:
        window_size += 1  # make it odd
    normalization_kernel = np.ones((window_size, window_size)) / (window_size**2 - 1)
    normalization_kernel[window_size // 2, window_size // 2] = 0
    smooth_x = signal.convolve2d(x, normalization_kernel, mode='same')
    ret = x / (smooth_x + 1e-10)
    return ret

def calc_spectrogram(x, fs, nperseg, percent_overlap, window='hamming', remove_dc=None, crop_freq=None, logscale=True):
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

def calc_welch_from_spectrogram(Sxx, normalization_window_size=None):
    Pxx = np.mean(Sxx, axis=1)
    if normalization_window_size is not None:
        Pxx = rw_normalization(Pxx, window_size=normalization_window_size)
    Pxx = np.abs(Pxx - 1)
    return Pxx

def calc_std_from_spectrogram(Sxx, normalization_window_size=None):
    P_std = np.std(Sxx, axis=1)
    if normalization_window_size is not None:
        P_std = rw_normalization(P_std, window_size=normalization_window_size)
    return P_std

def calc_avg_diff_1d(x):
    return np.log(np.average(np.abs(np.diff(x))) + 1e-100)

def calc_avg_diff_from_spectrogram(Sxx, normalization_window_size=None):
    avg_diff = np.log(np.average(np.abs(np.diff(Sxx, axis=0)), axis=0))
    if normalization_window_size is not None:
            avg_diff = rw_normalization(avg_diff, window_size=normalization_window_size)
    return avg_diff

def calc_rw_symmetry(pxx, window_size):
    half = window_size // 2
    vals = np.zeros(pxx.shape[0])
    for i in range(pxx.shape[0]):
        # compute a local half-window without modifying the outer 'half' value
        local_half = min(half, i, pxx.shape[0] - i - 1)
        if local_half <= 0:
            vals[i] = 0
            continue

        fw = pxx[i-local_half:i]
        bw = pxx[i+1:i+local_half+1]  #[::-1]

        # ensure equal lengths (safety)
        if fw.shape[0] != bw.shape[0]:
            minlen = min(fw.shape[0], bw.shape[0])
            fw = fw[-minlen:]
            bw = bw[:minlen]
        
        vals[i] = np.convolve(fw, bw, mode='valid')[0]
    return vals

def calc_local_curvature(pxx, window_size):
    half = window_size // 2
    vals = np.zeros(pxx.shape[0])
    for i in range(pxx.shape[0]):
        # compute a local half-window without modifying the outer 'half' value
        local_half = min(half, i, pxx.shape[0] - i - 1)
        if local_half <= 0:
            vals[i] = 0
            continue

        fw = pxx[i-local_half:i]
        bw = pxx[i+1:i+local_half+1][::-1]

        # ensure equal lengths (safety)
        if fw.shape[0] != bw.shape[0]:
            minlen = min(fw.shape[0], bw.shape[0])
            fw = fw[-minlen:]
            bw = bw[:minlen]
        
        vals[i] = np.sum((fw - bw)**2)

    vals = rw_normalization(vals)
    return vals

def calc_curvature(pxx, normalization_window_size=None):
    curvature = np.zeros_like(pxx)
    for i in range(1, len(pxx)-1):
        curvature[i] = np.abs(pxx[i+1] - 2*pxx[i] + pxx[i-1])
    if normalization_window_size is not None:
        curvature = rw_normalization(curvature, window_size=normalization_window_size)
    return curvature

## ====================
## GRAPH UTILITIES
## ====================

def graph_decomposition(adj_matrix):
    D = np.diag(np.sum(adj_matrix, axis=1))  # compute the Degree matrix
    L = D - adj_matrix  # compute Laplacian matrix
    eigenvalues, eigenvectors = np.linalg.eigh(L)  # compute eigenvalues and eigenvectors
    eigenvalue_counts = np.sum(np.isclose(eigenvalues, 0, atol=1e-5))  # count connected components
    return eigenvalues, eigenvectors, eigenvalue_counts

def calc_degree_distribution_from_graph_matrix(adj_matrix):
    degrees = np.sum(adj_matrix, axis=1)
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    degree_likelihood = counts  #  / len(degrees) + 1e-100
    return unique_degrees, degree_likelihood

def plot_graph_from_matrix(M, show=False):
    fig = go.Figure()
    dots = np.where(M == 1)
    fig.add_trace(go.Scatter(x=dots[0], y=dots[1], mode='markers', marker=dict(size=0.5, color='black'), showlegend=False))
    fig.update_layout(title='Visibility Graph', xaxis_title='Node Index', yaxis_title='Node Index', height=600, width=600)
    if show:
        fig.show()
    return fig

def draw_graph(G):
    # Calculate node positions using spring layout
    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), mode='lines', showlegend=False)

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        showlegend=False,
        marker=dict(showscale=False, colorscale='YlGnBu', reversescale=True, color=[], size=10, colorbar=dict(thickness=15, title=dict(text='Node Connections', side='right'), xanchor='left'), line_width=2))

    node_adjacencies = []
    # node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        # node_text.append('# of connections: '+str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = list(G.nodes())

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest', height=600, width=600,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

## ===============
## INFORMATION THEORETIC UTILITIES
## ===============

def entropy(data, nbins=None):
    n_samples = data.shape[0]
    if nbins == None:
        nbins = int((n_samples/5)**.5)
    histogram, _ = np.histogram(data, bins=nbins)
    if np.any(np.isnan(histogram)):
        print("Warning: NaN histogram encountered in entropy calculation. len data =", len(data))

    probs = histogram / len(data) + 1e-100
    if np.any(np.isnan(probs)):
        print(f"Warning: NaN probabilities encountered in entropy calculation.  len data = {len(data)}")
    entropy = -(probs * np.log2(probs)).sum()
    return entropy

def mutual_information(x, y, nbins=None):
    n_samples = x.shape[0]
    if nbins == None:
        nbins = int((n_samples/5)**.5)
    hist_x, _ = np.histogram(x, bins=nbins)
    hist_y, _ = np.histogram(y, bins=nbins)
    hist_xy, _, _ = np.histogram2d(x, y, bins=nbins)

    p_x = hist_x / n_samples + 1e-100
    p_y = hist_y / n_samples + 1e-100
    p_xy = hist_xy / n_samples + 1e-100

    H_x = -(p_x * np.log2(p_x)).sum()
    H_y = -(p_y * np.log2(p_y)).sum()
    H_xy = -(p_xy * np.log2(p_xy)).sum()

    mi = H_x + H_y - H_xy
    return mi

def entropy_of_frequencies(data, nbins="auto"):
    n_samples = data.shape[0]
    if nbins is None:
        nbins = int((n_samples/5)**.5)
    entropies = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        entropies[i] = entropy(data[i, :], nbins=nbins)
    return entropies

def joint_entropies(data, nbins=None):  
    n_variables = data.shape[-1]
    n_samples = data.shape[0]
    if nbins == None:
        nbins = int((n_samples/5)**.5)
        if nbins < 2:
            nbins = n_samples
    histograms2d = np.zeros((n_variables, n_variables, nbins, nbins))
    for i in range(n_variables):
        for j in range(n_variables):
            histograms2d[i,j] = np.histogram2d(data[:,i], data[:,j], bins=nbins)[0]
    probs = histograms2d / len(data) + 1e-100
    joint_entropies = -(probs * np.log2(probs)).sum((2,3))
    return joint_entropies

def mutual_info_matrix(data, nbins=None, normalized=True):
    n_variables = data.shape[-1]
    j_entropies = joint_entropies(data, nbins)
    entropies = j_entropies.diagonal()
    entropies_tile = np.tile(entropies, (n_variables, 1))
    sum_entropies = entropies_tile + entropies_tile.T
    mi_matrix = sum_entropies - j_entropies
    if normalized:
        mi_matrix = mi_matrix * 2 / sum_entropies    
    return mi_matrix

def calc_mi_adjacency_matrix(data, nbins=None, percentile_threshold=None, normalized=True):
    mi_matrix = mutual_info_matrix(data.T, nbins, normalized)
    if percentile_threshold is not None:
        threshold = np.percentile(mi_matrix, 100 - percentile_threshold)
        adj_matrix = (mi_matrix >= threshold).astype(int)
    else:
        adj_matrix = mi_matrix
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix

## ====================
## VISIBILITY GRAPH UTILITIES
## ====================

def simple_calc_vis_graph(f, Pxx):
    vis = np.zeros((len(f), len(f)))
    for i in range(len(f)):
        for j in range(i+1, len(f)):
            blocked = False
            for k in range(i+1, j):
                if Pxx[k] >= (Pxx[i] + (Pxx[j] - Pxx[i]) * (k - i) / (j - i)):
                    blocked = True
                    break
            if not blocked:
                vis[i, j] = 1
                vis[j, i] = 1
    return vis

def nvg_dc_np(Pxx, left, right, all_visible=None):

    if all_visible == None : all_visible = []
    node_visible = []

    if left < right : # there must be at least two nodes in the time series
        i = np.argmax(Pxx[left:right]) + left
        
        # check if i can see each node of series[left...right]
        for k in np.arange(left, right):
            if k != i:
                a = min(i,k)
                b = max(i,k)
                if np.all(Pxx[a+1:b] < (Pxx[a] + (Pxx[b] - Pxx[a])*(i-a)/(b-a))):
                    node_visible.append(k)

        if len(node_visible) > 0 : all_visible.append([i, node_visible])

        nvg_dc_np(Pxx, left, i, all_visible = all_visible)
        nvg_dc_np(Pxx, i+1, right, all_visible = all_visible)

    return all_visible

def get_vis_counts(f, Pxx):
    vis_list = nvg_dc_np(Pxx, 0, len(Pxx))
    vis_count = np.zeros(len(f))
    for center, visibles in vis_list:
        vis_count[center] = len(visibles)
    return vis_count

def calc_vis_graph(f, Pxx):
    vis_list = nvg_dc_np(Pxx, left=0, right=len(Pxx))
    vis = np.zeros((len(f), len(f)))
    for center, visibles in vis_list:
        for v in visibles:
            vis[center, v] = 1
            vis[v, center] = 1
    return vis

def nvg_dc_np_uncontinouos(x, y, left, right, all_visible = None):

    if all_visible == None : all_visible = []
    node_visible = []

    if left < right : # there must be at least two nodes in the time series
        k = np.argmax(y[left:right]) + left

        # check if k can see each node of series[left...right]
        for j in np.arange(left,right):
            if j != k :
                a = min(j,k)
                b = max(j,k)

                ya = y[a]
                ta = x[a]
                yb = y[b]
                tb = x[b]
                yc = y[a+1:b]
                tc = x[a+1:b]

                if np.all( yc < (ya + (yb - ya)*(tc - ta)/(tb-ta))):
                    node_visible.append(x[j])

        if len(node_visible) > 0 : all_visible.append([x[k], node_visible])

        nvg_dc_np_uncontinouos(x, y, left, k, all_visible = all_visible)
        nvg_dc_np_uncontinouos(x, y, k+1, right, all_visible = all_visible)

    return all_visible

def calc_vis_graph_generalized(x, y, return_Adj=True):
    vis_list = nvg_dc_np_uncontinouos(x=x, y=y, left=0, right=len(y))
    if return_Adj:
        vis = np.zeros((len(x), len(x)))
    else: 
        vis = []
    for center, visibles in vis_list:
        if return_Adj:
            center_idx = np.where(x == center)[0][0]
        for v in visibles:
            if return_Adj:
                v_idx = np.where(x == v)[0][0]
                vis[center_idx, v_idx] = 1
                vis[v_idx, center_idx] = 1
            else:
                vis.append((center, v))
    return vis

## ====================
## Viterbi UTILITIES
## ====================

def _build_transition_matrix(f, sigma, p_birth, p_death, p_stay_gap):
    """
    Returns (F+1)x(F+1) transition matrix.
    Last state index = GAP.
    """
    A = np.zeros((f+1, f+1))

    # --- Frequency → Frequency ---
    freq = np.arange(f)
    for i in range(f):
        A[i, :f] = np.exp(-0.5 * ((freq - i) / sigma)**2)

    # Normalize rows
    A[:f, :f] /= A[:f, :f].sum(axis=1, keepdims=True)

    # --- Add death transitions ---
    A[:f, f] = p_death
    A[:f, :f] *= (1 - p_death)

    # --- GAP transitions ---
    A[f, f] = p_stay_gap
    A[f, :f] = (1 - p_stay_gap) / f  # uniform birth
    
    # Normalize GAP row
    A[f] /= A[f].sum()

    return A

def _band_limited_transition_matrix(nf, band_width, p_gap):
    """
    Returns (F+1)x(F+1) band-limited transition matrix.
    Last state index = GAP.
    """
    A = np.zeros((nf+1, nf+1))

    # --- Frequency → Frequency ---
    # freq = np.arange(f)
    for i in range(nf):
        lower_bound = max(0, i - band_width)
        upper_bound = min(nf, i + band_width + 1)
        A[i, lower_bound:upper_bound] = 1
    # Normalize rows
    A[:nf, :nf] /= A[:nf, :nf].sum(axis=1, keepdims=True)
    # --- Add death transitions ---
    A[:nf, nf] = p_gap
    A[:nf, :nf] *= (1 - p_gap)
    # --- Add birth transitions ---
    A[nf, :nf] = p_gap / nf  # uniform birth
    A[nf, nf] = 1 - p_gap
    # --- GAP transitions ---
    # p_stay_gap = 1 - p_birth
    # A[nf, nf] = p_stay_gap
    # A[nf, :nf] = (1 - p_stay_gap) / nf  # uniform birth
    # Normalize GAP row
    A[nf] /= A[nf].sum()
    return A

def viterbi_single_track_with_gap(Sxx, p_gap=None, prob_scale=1, band_width=None, sigma=None):
    """
    Viterbi algorithm for single track with GAP state.
    Sxx: (nt, nf) log-likelihood spectrogram
    Returns: track (nt,) frequency indices
    """
    nt, nf = Sxx.shape
    if band_width is None:
        if sigma is None:
            band_width = 5  # default value
        else:
            transition_props = _build_transition_matrix(nf, sigma=sigma, p_birth=p_gap, p_death=p_gap, p_stay_gap=1 - p_gap)
    else:
        transition_props = _band_limited_transition_matrix(nf, band_width=band_width, p_gap=p_gap)
    transition_props_log = np.log(transition_props + 1e-300)

    # Observation matrix extended with GAP
    S_ext = np.zeros((nt, nf+1))
    S_ext[:, :nf] = Sxx  # GAP has zero likelihood

    # DP tables
    V = np.zeros((nt, nf+1))       # V[t, state]
    BP = np.zeros((nt, nf+1), dtype=np.int32)

    # Initialization: all tracks start in GAP
    V[0, :] = S_ext[0, :]  # all tracks identical at t=0

    # Main Viterbi recursion (vectorized)
    for t in range(1, nt):
        # compute scores for all previous states to all current states

        scores = V[t-1] + (transition_props_log * prob_scale)

        # max over previous state
        V[t] = S_ext[t][None, :] + np.max(scores, axis=1)

        # store argmax
        BP[t] = np.argmax(scores, axis=1)

    track = np.zeros(nt, dtype=int)
    track[nt-1] = np.argmax(V[nt-1, :])
    for t in reversed(range(1, nt)):
        track[t-1] = BP[t, track[t]]

    score = V[nt-1, track[nt-1]]

    return track, score

def _mask_track(S, path, width):
    """
    Zero out energy around the extracted track.
    width: number of bins around track to remove
    """
    T, N = S.shape
    S2 = S.copy()
    for t in range(T):
        lo = max(0, path[t] - width)
        hi = min(N, path[t] + width + 1)
        S2[t, lo:hi] = 0
    return S2

def viterbi_multi_track(Sxx, band_width=5, p_gap=0.001, width=3, min_energy_ratio=0.2, prob_scale=100):
    _S = Sxx.copy().astype(np.float32)
    _S = (_S - np.min(_S)) / (np.max(_S) - np.min(_S) + 1e-300)  # normalize to [0, 1]
    original_energy = np.sum(_S)
    original_energy_percentile_per_time = np.percentile(_S, 5, axis=1)
    
    tracks = []
    # scores = []
    while True:
        current_energy = np.sum(_S)
        if np.log(current_energy) < np.log(original_energy) * 0.1:
            print(f"Stopping track extraction: current energy {current_energy:.4f} below threshold.")
            break

        track, score = viterbi_single_track_with_gap(_S, band_width=band_width, p_gap=p_gap, prob_scale=prob_scale)

        # # If the track has too little energy, stop

        active_mask = (track < _S.shape[1])  # ignore GAP states
        track_times = np.arange(_S.shape[0])[active_mask]
        track_frequencies = track[active_mask]
        track_values = _S[track_times, track_frequencies]
        track_energy = np.sum(track_values)

        if track_energy < np.percentile(original_energy_percentile_per_time, 5):
            print(f"Stopping track extraction: track energy {track_energy:.4f} below threshold {original_energy * min_energy_ratio:.4f}.")
            break

        # if score

        tracks.append((track_times, track_frequencies, track_energy))
        # scores.append(score)
        _S = _mask_track(_S, track, width=width)

    return tracks

def viterbi_from_welch_detection(Sxx, detected_frequencies, band_width, p_gap=None, prob_scale=1, sigma=None):
    tracks = []
    for d in detected_frequencies:
        lb = max(0, d - band_width // 2)
        ub = min(Sxx.shape[1], d + band_width // 2 + 1)
        _S = Sxx[:, lb:ub]
        track, score = viterbi_single_track_with_gap(_S, sigma=sigma, p_gap=p_gap, prob_scale=prob_scale)
        track += lb  # adjust frequency indices
        tracks.append(track)
    return tracks

def plot_tracks_over_spectrogram(Sxx, f, t, tracks, title="Spectrogram with Viterbi Tracks"):
    fig = go.Figure()

    # Spectrogram
    fig.add_trace(go.Heatmap(
        z=Sxx.T,
        x=t,
        y=f,
        colorscale='magma',
        showscale=False
    ))

    # Viterbi tracks
    for k, track in enumerate(tracks):
        track_times, track_freqs, track_energy = track
        track_times = t[track_times]
        track_freqs = f[track_freqs]
        fig.add_trace(go.Scatter(
            x=track_times,
            y=track_freqs,
            mode='lines',
            line=dict(width=2, color='cyan'),
            name=f'Track {k+1}'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        height=600,
        width=800,
        font=dict(size=12)
    )

    fig.show()

def welch_detector_with_viterbi_tracks(Sxx, 
                                       detection_height_threshold=None, 
                                       detection_prominence_threshold=None,
                                       detection_threshold=None,
                                       normalization_window_size=None,
                                       band_width=None,
                                       sigma=None,
                                       p_gap=None,
                                       prob_scaling_factor=1):
    pxx = calc_welch_from_spectrogram(Sxx, normalization_window_size=normalization_window_size)
    pxx = calc_curvature(pxx)
    height_threshold = np.average(pxx) + detection_height_threshold * np.std(pxx)
    peaks, properties = find_peaks(pxx, height=height_threshold, prominence=detection_prominence_threshold, threshold=detection_threshold)
    tracks = viterbi_from_welch_detection(Sxx, detected_frequencies=peaks, band_width=band_width, sigma=sigma, p_gap=p_gap, prob_scale=prob_scaling_factor)
    return peaks, tracks

def calc_tracks(data, 
                fs, 
                fft_nperseg, 
                percent_overlap, 
                window,
                remove_dc, 
                crop_freq, 
                normalization_window_size, 
                detection_threshold, 
                p_gap, 
                band_width, 
                sigma, 
                p_scale):
    
    F, T, Sxx, phase = calc_spectrogram(data, fs=fs, nperseg=fft_nperseg, percent_overlap=percent_overlap, window=window, remove_dc=remove_dc, crop_freq=crop_freq)
    Sxx = 10 * np.log10(Sxx + 1e-12)
    pxx = calc_welch_from_spectrogram(Sxx, normalization_window_size=normalization_window_size)
    pxx = calc_curvature(pxx)
    peaks, track_ixs = welch_detector_with_viterbi_tracks(Sxx, 
                                                        normalization_window_size=normalization_window_size, 
                                                        detection_height_threshold=detection_threshold, 
                                                        sigma=sigma, 
                                                        p_gap=p_gap, 
                                                        band_width=band_width,
                                                        prob_scaling_factor=p_scale)
    tracks = []
    for tix in track_ixs:
        tracks.append(Sxx[np.arange(Sxx.shape[0]), tix])
        
    return F, T, Sxx, pxx, peaks, track_ixs, tracks

def plot_tracks(F, T, Sxx, pxx, peaks, track_ixs, name, title="Spectrogram with Tracks", show=True):
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.8, 0.2], horizontal_spacing=0.15, vertical_spacing=0.15, column_titles=("Spectrogram", "Welch Power Spectral Density"))
    fig.add_trace(go.Heatmap(x=T, y=F, z=Sxx.T, colorscale='Viridis', showscale=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=pxx, y=F, mode='lines', name='Welch PSD', showlegend=False, line=dict(color='blue', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=pxx[peaks], y=F[peaks], mode='markers', name='Detected Peaks', marker=dict(color='cyan', size=8), showlegend=False), row=1, col=2)
    for track in track_ixs:
        fig.add_trace(go.Scatter(x=T, y=F[track], mode='markers', line=dict(color='cyan', width=1), name='Viterbi Track', showlegend=False), row=1, col=1)

    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_xaxes(title_text="Power/Frequency (dB/Hz)", row=1, col=2)
    fig.update_layout(height=height, width=width, title_text=f"Spectrogram and Welch PSD with Detected Peaks of {name}", font=dict(size=font_size))
    if show:
        fig.show()
    return fig

## ====================
## GENERAL UTILITIES
## ====================

def slice_signal(data, annotations, fs, slice_duration=1.0, overlap=0.5, annotation_threshold=0.25):
    slice_length = int(slice_duration * fs)
    step = int(slice_length * (1 - overlap))
    slices = []
    slice_annotations = []
    for start in range(0, len(data) - slice_length + 1, step):
        end = start + slice_length
        slices.append(data[start:end])
        # Get annotations for the current slice
        _annotation = 1 if np.mean(annotations[start:end]) >= annotation_threshold else 0
        slice_annotations.append(_annotation)
    return np.array(slices), np.array(slice_annotations)

def load_ds_samples(with_print=True, target_fs=None):
    # dpv1 - croatia - speed 1
    data_file = "../data/dpv1_1m.wav"
    # dpv1_fs, dpv1_data = wavfile.read(data_file)
    dpv1_data, dpv1_fs = librosa.load(data_file, sr=target_fs)
    dpv1_slice = dpv1_data[int(35*dpv1_fs):int(45*dpv1_fs)]

    # #dpv1 - croatia - speed 2
    # data_file = "../data/dpv1_1m_3.wav"
    # dpv1_2_fs, dpv1_2_data = wavfile.read(data_file)
    # dpv1_2_slice = dpv1_2_data[int(35*dpv1_2_fs):int(45*dpv1_2_fs)]

    # dpv2 - haifa
    data_file = "../data/dpv2_1m.wav"
    # dpv2_fs, dpv2_data = wavfile.read(data_file)
    dpv2_data, dpv2_fs = librosa.load(data_file, sr=target_fs)
    dpv2_slice = dpv2_data[int(15*dpv2_fs):int(25*dpv2_fs)]

    # croatia boat noise
    data_file = "../data/croatia_boat_2.wav"
    # croatia_boat_fs, croatia_boat_data = wavfile.read(data_file)
    croatia_boat_data, croatia_boat_fs = librosa.load(data_file, sr=target_fs)
    croatia_boat_slice = croatia_boat_data[int(5*croatia_boat_fs):int(15*croatia_boat_fs)]

    # # motorboat
    # data_file = "../data/motorboat_1m.wav"
    # motorboat_fs, motorboat_data = wavfile.read(data_file)
    # motorboat_slice = motorboat_data[int(5*motorboat_fs):int(15*motorboat_fs)]

    # # large ship
    # data_file = "../data/large_ship_1m.wav"
    # large_ship_fs, large_ship_data = wavfile.read(data_file)
    # large_ship_slice = large_ship_data[int(5*large_ship_fs):int(15*large_ship_fs)]

    # BG noise
    data_file = "../data/bg_noise_1m.wav"
    # bg_fs, bg_data = wavfile.read(data_file)
    bg_data, bg_fs = librosa.load(data_file, sr=target_fs)
    bg_slice = bg_data[int(5*bg_fs):int(15*bg_fs)]

    # # BG noise 2
    # data_file = "../data/bg_noise_2_1m.wav"
    # bg2_fs, bg2_data = wavfile.read(data_file)
    # bg2_slice = bg2_data[int(5*bg2_fs):int(15*bg2_fs)]

    # organize samples
    names = [
        "dpv1",
        # "dpv1_2",
        "dpv2",
        "croatia_ship",     
        # "motorboat", 
        # "large_ship", 
        "bg_noise"
        # "bg_noise_2"
        ]

    fss = {
        "dpv1": dpv1_fs,
        # "dpv1_2": dpv1_2_fs,
        "dpv2": dpv2_fs,
        "croatia_ship": croatia_boat_fs,
        # "motorboat": motorboat_fs,
        # "large_ship": large_ship_fs,
        "bg_noise": bg_fs
        # "bg_noise_2": bg2_fs,
        }

    all_data = {
        # "dpv1_2": dpv1_2_data,
        "dpv2": dpv2_data,
        "croatia_ship": croatia_boat_data,
        # "motorboat": motorboat_data,
        # "large_ship": large_ship_data,
        "bg_noise": bg_data
        # "bg_noise_2": bg2_data,
        }

    slices = {
        "dpv1": dpv1_slice,
        # "dpv1_2": dpv1_2_slice,
        "dpv2": dpv2_slice,
        "croatia_ship": croatia_boat_slice,
        # "motorboat": motorboat_slice,
        # "large_ship": large_ship_slice,
        "bg_noise": bg_slice
        # "bg_noise_2": bg2_slice,
        }
    
    # print info
    if with_print:
        print(f"Data dicts: ")
        print(f"names={names}")
        print(f"fss keys={list(fss.keys())}")
        print(f"all_data keys={list(all_data.keys())}")
        print(f"slices keys={list(slices.keys())}")

    return names, fss, all_data, slices

def explore_data(name_list, fs_dict, data_dict, fft_nperseg=32000, percent_overlap=0.5, window='hamming', remove_dc=20, crop_freq=None, detection_threshold=2, height=1600, width=1200, font_size=16):
    all_peaks = {}
    titles = [item for pair in zip(name_list, [""]*len(name_list)) for item in pair]
    fig = make_subplots(rows=len(name_list), cols=2, vertical_spacing=0.02, subplot_titles=titles, horizontal_spacing=0.01, column_widths=[0.8, 0.2], shared_yaxes=True, shared_xaxes=True)

    for i, name in enumerate(name_list, start=1):
        fs = fs_dict[name]
        data = data_dict[name]
        F, T, Sxx, _ = calc_spectrogram(data, fs, fft_nperseg, percent_overlap, window, remove_dc, crop_freq)
        pxx = calc_welch_from_spectrogram(Sxx, normalization_window_size=9) - 1
        peaks = find_peaks(pxx, height=np.average(pxx) + detection_threshold * np.std(pxx))[0][1:]  # ignore DC peak
        all_peaks[name] = peaks

        fig.add_trace(go.Heatmap(x=T, y=F, z=Sxx, colorscale='Viridis', showscale=False), row=i, col=1)
        fig.add_trace(go.Scatter(x=pxx, y=F, mode='lines', line=dict(color='blue')), row=i, col=2)
        fig.add_trace(go.Scatter(x=pxx[peaks], y=F[peaks], mode='markers', marker=dict(color='red', size=6)), row=i, col=2)
        fig.update_yaxes(title_text="Frequency (Hz)", row=i, col=1, title_font_size=font_size)
        if i == len(name_list):
            fig.update_xaxes(title_text="Time (s)", row=i, col=1, title_font_size=font_size) 
        fig.update_yaxes(range=[0, crop_freq], row=i, col=1)

    fig.update_layout(height=height, width=width, title_text="Spectrograms of Different Boat Noises", title_font_size=font_size+4)
    fig.show()

