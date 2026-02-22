import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full", app_title="UATR")


app._unparsable_cell(
    r"""

    # imports and settings

    import os
    import time
    import pickle
    import warnings
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    from copy import deepcopy

    import numpy as np
    from numpy import linalg as LA
    from numpy import histogram2d

    from scipy import signal
    from scipy.fft import fft, fftfreq, fftshift
    from scipy.signal import find_peaks, butter, filtfilt, welch
    from scipy.ndimage import gaussian_filter
    from scipy.io import wavfile
    from scipy.stats import wasserstein_distance_nd

    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    import utils as ut
    import marimo as mo

    # do not show warnings
    warnings.filterwarnings(\\"ignore\\")

    print(\\"Imports complete.\\")

    """,
    name="_"
)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
