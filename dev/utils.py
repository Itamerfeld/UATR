import os
import time
import pickle
import networkx as nx
import matplotlib.pyplot as plt

import mlx
import mlx.core as mx
import mlx.core.linalg as mxla

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

## ====================
## GENERAL UTILITIES
## ====================

def try_except_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    return wrapper

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

## ====================
## SIGNAL PROCESSING UTILITIES
## ====================

def get_spectrogram(data, 
                    fs, 
                    nperseg=65536,
                    hop=None, 
                    noverlap=None, 
                    window='hann', 
                    crop_freq=None):

    if hop is None:
        noverlap = nperseg // 8
    else:
        noverlap = int(nperseg * hop)

    # Compute spectrogram
    frequencies, times, Sxx = signal.spectrogram(
        data,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap
    )

    # Crop frequencies if specified
    if crop_freq is not None:
        freq_mask = frequencies <= crop_freq
        frequencies = frequencies[freq_mask]
        Sxx = Sxx[freq_mask, :]

    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Add small value to avoid log(0)

    return frequencies, times, Sxx_db

def pwelch(data, fs, remove_dc=20, crop_freq=None, nperseg=None, normalization_window_size=101):
    if nperseg is None:
        nperseg = fs * 2

    f, Pxx = welch(data, fs=fs, nperseg=nperseg)

    if normalization_window_size is not None:
        if normalization_window_size % 2 == 0:
            normalization_window_size += 1  # make it odd
        normalization_kernel = np.ones((normalization_window_size,)) / (normalization_window_size-1)
        normalization_kernel[normalization_window_size // 2] = 0
        smoothed_Pxx = signal.convolve(Pxx, normalization_kernel, mode='same')
        Pxx = Pxx / (smoothed_Pxx + 1e-10)
    if remove_dc is not None:
        dc_band = f <= remove_dc
        Pxx[dc_band] = 0
    if crop_freq is not None:
        crop_band = f <= crop_freq
        f = f[crop_band]
        Pxx = Pxx[crop_band]

    return f, Pxx

def pwelch_line_detection(data, fs, threshold=5, remove_dc=20, crop_freq=5000, nperseg=None, normalization_window_size=None):
    f, Pxx = pwelch(data, fs, remove_dc=remove_dc, crop_freq=crop_freq, nperseg=nperseg, normalization_window_size=normalization_window_size)
    threshold = np.mean(Pxx) + threshold * np.std(Pxx)
    peaks, properties = find_peaks(Pxx, height=threshold)
    detected_frequencies = f[peaks]
    detected_powers = Pxx[peaks]
    return detected_frequencies, detected_powers, f, Pxx

def my_spec(x, fs, nperseg, noverlap, window, normalization_window_size=None, remove_dc=None, crop_freq=None, with_entropy=False):

    # bandpass filter
    if remove_dc is not None and crop_freq is not None:
        b, a = butter(4, [remove_dc/(fs/2), crop_freq/(fs/2)], btype='band')
        x = filtfilt(b, a, x)
    elif remove_dc is not None:
        b, a = butter(4, remove_dc/(fs/2), btype='high')
        x = filtfilt(b, a, x)
    elif crop_freq is not None:
        b, a = butter(4, crop_freq/(fs/2), btype='low')
        x = filtfilt(b, a, x)
    
    # prepare for PSD calculation
    step = nperseg - noverlap
    window_vals = getattr(np, window)(nperseg) if hasattr(np, window) else np.hamming(nperseg)
    window_power = np.sum(window_vals**2)

    n_segments = (len(x) - noverlap) // step
    f = np.fft.rfftfreq(nperseg, 1/fs)
    p_matrix = np.zeros((n_segments, len(f)))

    for i in range(n_segments):
        start = i * step
        segment = x[start:start+nperseg]
        if len(segment) < nperseg:
            break
        segment = segment * window_vals
        spectrum = np.fft.rfft(segment)
        p_matrix[i, :] = (np.abs(spectrum)**2) / (fs * window_power)

    # Pxx = psd / n_segments
    Pxx = np.mean(p_matrix, axis=0)
    P_std = np.std(p_matrix, axis=0)
    # if with_entropy:
    #     P_entropy = entropy_of_frequencies(p_matrix.T)

    # normalization
    if normalization_window_size is not None:
        if normalization_window_size % 2 == 0:
            normalization_window_size += 1  # make it odd
        normalization_kernel = np.ones((normalization_window_size,)) / (normalization_window_size-1)
        normalization_kernel[normalization_window_size // 2] = 0

        smoothed_Pxx = signal.convolve(Pxx, normalization_kernel, mode='same')
        Pxx = Pxx / (smoothed_Pxx + 1e-10)
        smoothed_P_std = signal.convolve(P_std, normalization_kernel, mode='same')
        P_std = P_std / (smoothed_P_std + 1e-10)
        # if with_entropy:
        #     smoothed_P_entropy = signal.convolve(P_entropy, normalization_kernel, mode='same')
        #     P_entropy = P_entropy / (smoothed_P_entropy + 1e-10)

    # # remove DC and crop frequencies  # TODO: consider replacing with bandpass filter
    # if remove_dc is not None:
    #     dc_band = f >= remove_dc
    #     f = f[dc_band]
    #     Pxx = Pxx[dc_band]
    #     P_std = P_std[dc_band]
    #     if with_entropy:
    #         P_entropy = P_entropy[dc_band]

    # if crop_freq is not None:
    #     crop_band = f <= crop_freq
    #     f = f[crop_band]
    #     Pxx = Pxx[crop_band]
    #     P_std = P_std[crop_band]
    #     if with_entropy:
    #         P_entropy = P_entropy[crop_band]

    # prepare output
    # ans = (f, Pxx, P_std, P_entropy) if with_entropy else (f, Pxx, P_std)

    return f, Pxx, P_std, p_matrix

def running_window_spectrogram(x, fs, nperseg, percent_overlap, window='hamming', remove_dc=None, crop_freq=None):

    # bandpass filter
    if remove_dc is not None and crop_freq is not None:
        b, a = butter(4, [remove_dc/(fs/2), crop_freq/(fs/2)], btype='band')
        x = filtfilt(b, a, x)
    elif remove_dc is not None:
        b, a = butter(4, remove_dc/(fs/2), btype='high')
        x = filtfilt(b, a, x)
    elif crop_freq is not None:
        b, a = butter(4, crop_freq/(fs/2), btype='low')
        x = filtfilt(b, a, x)
    
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

    return f, t, Sxx, Phase

def welch_from_pmatrix(p_matrix, normalization_window_size=101):
    Pxx = np.mean(p_matrix, axis=0)
    if normalization_window_size is not None:
        Pxx = rw_normalization(Pxx, window_size=normalization_window_size)
    return Pxx

def std_from_pmatrix(p_matrix, normalization_window_size=101):
    P_std = np.std(p_matrix, axis=0)
    if normalization_window_size is not None:
        if normalization_window_size % 2 == 0:
            normalization_window_size += 1  # make it odd
        normalization_kernel = np.ones((normalization_window_size,)) / (normalization_window_size-1)
        normalization_kernel[normalization_window_size // 2] = 0
        smoothed_P_std = signal.convolve(P_std, normalization_kernel, mode='same')
        P_std = P_std / (smoothed_P_std + 1e-100)
    return P_std

def avg_diff_from_pmatrix(p_matrix, normalization_window_size=101):
    avg_diff = np.log(np.average(np.abs(np.diff(p_matrix, axis=0)), axis=0))
    if normalization_window_size is not None:
            if normalization_window_size % 2 == 0:
                normalization_window_size += 1  # make it odd
            normalization_kernel = np.ones((normalization_window_size,)) / (normalization_window_size-1)
            normalization_kernel[normalization_window_size // 2] = 0
            smooth_avg_diff = signal.convolve(avg_diff, normalization_kernel, mode='same')
            avg_diff = avg_diff / (smooth_avg_diff + 1e-10)
    return avg_diff

def detect_frequencies(x, fs, nperseg, remove_dc, crop_freq, normalization_window_size, threshold):
    f, pxx, std, p_matrix = ut.my_spec(x, fs=fs, nperseg=nperseg, noverlap=nperseg//2, window='hamming', normalization_window_size=normalization_win, remove_dc=remove_dc, crop_freq=crop_freq)
    pxx = pxx / np.percentile(pxx, 90)
    pxx_peak_indices = find_peaks(pxx, height=(np.average(pxx) + np.std(pxx)*threshold))[0]
    std_peak_indices = find_peaks(std, height=(np.average(std) + np.std(std)*threshold))[0]
    peak_ixs = np.intersect1d(pxx_peak_indices, std_peak_indices)
    detected_frequencies = f[peak_ixs]
    return detected_frequencies, (f, pxx, std)

def running_window_convolution(pxx, nperseg):
    half_window = nperseg // 2
    convolved = np.zeros_like(pxx)
    for i in range(len(pxx)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(pxx), i + half_window + 1)
        segment = pxx[start_idx:end_idx]
        convolved[i] = np.max(segment * np.flip(segment))
    return convolved

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

def symmetry_enhancement(pxx, normalization_win=101, nperseg=1024):
    x = running_window_convolution(pxx, nperseg=nperseg)
    x = rw_normalization(x, window_size=normalization_win)
    x = x / (np.mean(x) + 1e-10)
    # x = x[100:-100]
    x = np.log(x)
    return x

def rw_symmetry(pxx, window_size):
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

    #     # vals[i] = np.sum((fw - bw)**2)
    #     vals[i] = np.sum(fw**2 - bw**2)
    #     vals[i] = vals[i] / (np.sum(fw**2) + np.sum(bw**2) + 1e-10)
    
    # vals = np.diff(vals, n=1, prepend=0)
    # vals = np.where(vals < 0, 0, vals)
    # vals = np.abs(vals)
    # vals = rw_normalization(vals)
    return vals

## ====================
## VISUALIZATION UTILITIES
## ====================

def plot_spectrogram(frequencies, 
                     times, 
                     Sxx_db, 
                     colorscale='Viridis',
                     title='Spectrogram',
                     xlabel='Time [s]',
                     ylabel='Frequency [Hz]',
                     show=False,
                     annotations=None):
    
    fig = px.imshow(
        Sxx_db,
        x=times,
        y=frequencies,
        aspect='auto',
        origin='lower',
        color_continuous_scale=colorscale,
        labels={'x': 'Time [s]', 'y': 'Frequency [Hz]', 'color': 'Intensity [dB]'}
    )

    if annotations:
        for annotation in annotations:
            fig.add_shape(
                type="rect",
                x0=annotation['x0'],
                y0=annotation['y0'],
                x1=annotation['x1'],
                y1=annotation['y1'],
                line=dict(color='Yellow', width=2, dash='dash'),
                label=dict(text=annotation['text'], font=dict(size=10, color='yellow'), textposition="top left")
            )
        
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    if show:
        fig.show()
    return fig

def plot_batch(slices, fs, slice_annotations, 
               ixs=None, 
               batch_size=4, 
               crop_freq=None, 
               remove_dc=20, 
               nperseg=32768, 
               show=False,
               with_vis_graph=False):
    if ixs is not None:
        batch_size = len(ixs)
    if ixs is None:
        ixs = np.random.choice(len(slices), size=batch_size, replace=False)

    _rows = 3 if with_vis_graph else 2
    fig = make_subplots(rows=_rows, cols=batch_size)

    for i, ix in enumerate(ixs):
        s = slices[ix]
        a = slice_annotations[ix]
        
        f, t, Sxx = get_spectrogram(s, fs, crop_freq=crop_freq, nperseg=nperseg)
        fig.add_trace(go.Heatmap(z=Sxx, x=t, y=f, colorscale='Viridis', showscale=False), row=1, col=i+1)

        f_pxx, Pxx = pwelch(s, fs, remove_dc=remove_dc, crop_freq=crop_freq)
        fig.add_trace(go.Scatter(x=f_pxx, y=Pxx, mode='lines'), row=2, col=i+1)

        if with_vis_graph:
            vis_adjencency = calc_vis_graph(f_pxx, Pxx)
            # show visibility adjacency as an image
            dots = np.where(vis_adjencency == 1)
            fig.add_trace(go.Scatter(x=dots[0], y=dots[1], mode='markers', marker=dict(size=0.5, color='black')), row=3, col=i+1)

        fig.update_xaxes(title_text=f'Slice {ix} - Annotation: {a}', row=1, col=i+1)

    if show:
        fig.show()
    return fig

def plot_graph_from_matrix(M, show=False):
    fig = go.Figure()
    dots = np.where(M == 1)
    fig.add_trace(go.Scatter(x=dots[0], y=dots[1], mode='markers', marker=dict(size=0.5, color='black')))
    fig.update_layout(title='Visibility Graph', xaxis_title='Node Index', yaxis_title='Node Index', height=600, width=600)
    if show:
        fig.show()
    return fig

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

## ===============
## INFORMATION THEORETIC UTILITIES
## ===============

def entropy(data, nbins="auto"):
    n_samples = data.shape[0]
    if nbins == None:
        nbins = int((n_samples/5)**.5)
    histogram, _ = np.histogram(data, bins=nbins)
    probs = histogram / len(data) + 1e-100
    entropy = -(probs * np.log2(probs)).sum()
    return entropy

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

def compute_mi_adjacency_matrix(data, nbins=None, percentile_threshold=None, normalized=True):
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
    for i in range(1, len(x_quantized)):
        if x_quantized[i] != x_quantized[i-1]:
            transitions[x_quantized[i-1], x_quantized[i]] += 1
    return transitions # this is not normalized yet #TODO

def get_s2g_edges(x_quantized):
    edges = []
    for i in range(1, len(x_quantized)):
        if x_quantized[i] != x_quantized[i-1]:
            edges.append((x_quantized[i-1], x_quantized[i]))
    return edges

def get_K(transition_matrix):
    edge_count = np.count_nonzero(transition_matrix, keepdims=True)
    K = edge_count / (transition_matrix.shape[0] * transition_matrix.shape[1])
    return K


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

def simulate_raw_signal(f0, fs, duration, snr_db):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * f0 * t)
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * pink_noise(len(signal))
    return signal + noise

def add_noise_to_signal(signal, noise, snr_db):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    desired_noise_power = signal_power / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(desired_noise_power / noise_power)
    noise = noise * scaling_factor
    noisy_signal = signal + noise
    return noisy_signal
