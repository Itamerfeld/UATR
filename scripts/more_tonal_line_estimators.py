import numpy as np
from scipy.signal import hilbert, medfilt
from scipy.interpolate import RegularGridInterpolator

def lea_Baumgartner(p, f, t, Ts):
    """
    Python translation of the MATLAB function lea_Baumgartner.m

    Parameters
    ----------
    p : 2D ndarray
        Spectrogram magnitude values, shape (len(f), len(t))
    f : 1D ndarray
        Frequency vector
    t : 1D ndarray
        Time vector
    Ts : float
        Threshold in dB

    Returns
    -------
    f0 : 1D ndarray
        Detected frequency track
    time : 1D ndarray
        Same as input t
    ampl : 1D ndarray
        Amplitudes of detected track
    """

    # Convert dB threshold
    threshold = 10 ** (Ts / 10)

    # Apply threshold
    p_threshold = np.full_like(p, np.nan)
    mask = p >= threshold
    p_threshold[mask] = p[mask]

    # Find detected points (non-NaNs)
    line, col = np.where(~np.isnan(p_threshold))
    weight = 20

    # Allocate output arrays
    Cost = np.zeros(len(t))
    Detected_freq = np.full(len(t), np.nan)
    Detected_amp = np.full(len(t), np.nan)

    # ---- FIRST detected value
    # The MATLAB code uses i=2, j=2 and last_j=j
    i = 1     # MATLAB 2 → Python index 1
    j = 1
    last_j = j

    P = weight * abs(np.log2(f[line[j-1]] / f[line[j]]))
    Cost[i] = P - p_threshold[line[j], col[i]]

    # ---- Remaining time steps
    for i in range(2, len(t) - 2):  # MATLAB: 3:(end-2)
        # Frequencies detected at this time index
        frequency_index = np.where(~np.isnan(p_threshold[:, i]))[0]

        if len(frequency_index) > 0:
            K = len(frequency_index)
            A = p_threshold[frequency_index, i]

            # Compute frequency deviation cost P(k)
            P = weight * np.abs(np.log2(f[last_j] / f[frequency_index]))

            # Dynamic programming cost
            costs = Cost[i - 1] + P - A
            Cost[i] = np.min(costs)

            if not np.isnan(Cost[i]):
                ind_min = np.argmin(costs)
                Detected_freq[i] = f[frequency_index[ind_min]]
                Detected_amp[i] = A[ind_min]
                last_j = frequency_index[ind_min]

    # Backward loop in MATLAB is empty — we keep it empty
    # for i = length(t)-2:-1:1 ; end

    return Detected_freq, t, Detected_amp

def lea_HPS(p, f, t):
    """
    Python translation of the MATLAB function lea_HPS.m

    Parameters
    ----------
    p : 2D ndarray (F × T)
        Spectrogram magnitudes
    f : 1D ndarray (F)
        Frequency vector
    t : 1D ndarray (T)
        Time vector

    Returns
    -------
    f0   : fundamental frequency estimates
    time : same as input t
    ampl : amplitude estimates (sqrt of HPS peak)
    """

    F, T = p.shape

    # Initialize p_HPS to ones, same size as p
    p_HPS = np.ones((F, T))

    # Number of harmonic downsamplings
    nb_harmo = 2

    # --- Harmonic Product Spectrum computation ---
    for i in range(T):
        h = p[:, i]  # spectrum at time i

        for n in range(nb_harmo, 0, -1):  # n = 2, 1
            h_down = h[::n]  # MATLAB: downsample(h, n)
            diff_size = F - len(h_down)

            # pad with NaNs to match original length
            if diff_size > 0:
                h_down_sized = np.concatenate([h_down, np.full(diff_size, np.nan)])
            else:
                h_down_sized = h_down[:F]

            p_HPS[:, i] *= h_down_sized

    # --- Construct HPS frequency vector ---
    # MATLAB: f_HPS = linspace(0,50,a)*nb_harmo;
    f_HPS = np.linspace(0, 50, F) * nb_harmo

    # --- Find maxima and amplitudes ---
    f0 = np.zeros(T)
    ampl = np.zeros(T)

    for i in range(T):
        # ignore NaNs during max
        col = p_HPS[:, i]
        idx = np.nanargmax(col)
        M = col[idx]

        f0[i] = f_HPS[idx]
        ampl[i] = np.sqrt(M)

    return f0, t, ampl

def lea_inst_freq(p, f, t, x, fs):
    """
    Python translation of MATLAB lea_inst_freq.m
    """

    # -------- INSTANTANEOUS FREQUENCY --------
    z = hilbert(x)
    inst_phase = np.unwrap(np.angle(z))
    instfreq = fs / (2 * np.pi) * np.diff(inst_phase)

    # Make x a row vector like MATLAB vector_orientation(...,'line')
    x = np.ravel(x)

    # -------- MEDIAN FILTERING --------
    instfreq_filt = medfilt(instfreq, kernel_size=21)

    # -------- SPECTROGRAM INTERPOLATION --------
    # MATLAB: p_interp = interp2(t,f,p,tx,f)
    # In Python we use RegularGridInterpolator on (f, t)

    interpolator = RegularGridInterpolator(
        (f, t), p, bounds_error=False, fill_value=np.nan
    )

    tx = np.arange(len(x)) / fs
    ampl_inst_freq = np.full_like(instfreq_filt, np.nan)

    # Interpolate p(f, t) at instantaneous frequencies
    for i in range(len(instfreq_filt)):
        freq_val = instfreq_filt[i]
        if freq_val >= f[0] and freq_val <= f[-1]:
            ampl_inst_freq[i] = interpolator((freq_val, tx[i]))

    # -------- (Commented-out MATLAB variance rejection left out) --------

    # MATLAB output logic:
    # f0 = [instfreq_filt 0];
    # ampl = [ampl_inst_freq 0];
    f0 = np.concatenate([instfreq_filt, [0]])
    ampl = np.concatenate([ampl_inst_freq, [0]])
    time = tx  # same as MATLAB

    return f0, time, ampl

def lea_yin_estimator(x, fs, delta_t, delta_f, signal_mini_duration_s,
                      silent_classifier=None):
    """
    Python translation of the MATLAB YIN pitch extraction function
    lea_yin_estimator.m
    """

    x = np.asarray(x).reshape(-1)
    N = len(x)

    # ---------- STEP 0: WINDOWING ----------
    win = int(round(0.25 * fs))          # minimum expected f0 ~ 4 Hz
    nframes = int(np.ceil(N / win))

    # zero-padding
    pad_len = win * nframes - N
    x_pad = np.concatenate([x, np.zeros(pad_len)])

    # frame the signal
    x_frame = x_pad.reshape(nframes, win)

    # ---------- STEP 1: DIFFERENCE FUNCTION ----------
    d = np.zeros((nframes, win))
    x_temp = np.concatenate([x_frame, np.zeros((nframes, win))], axis=1)

    for tau in range(win):
        for j in range(win):
            diff = x_temp[:, j] - x_temp[:, j + tau]
            d[:, tau] += diff ** 2

    # ---------- STEP 2: CMND (cumulative mean normalized difference) ----------
    d_norm = np.zeros_like(d)
    d_norm[:, 0] = 1

    for i in range(nframes):
        for tau in range(1, win):
            d_norm[i, tau] = d[i, tau] / (np.sum(d[i, :tau+1]) / tau)

    # ---------- STEP 3: ABSOLUTE THRESHOLD ----------
    th = 0.1
    lag = np.zeros(nframes, dtype=int)

    for i in range(nframes):
        idx = np.where(d_norm[i, :] < th)[0]
        if len(idx) > 0:
            lag[i] = idx[0]
        else:
            lag[i] = np.argmin(d_norm[i, :])

    # ---------- STEP 4: PARABOLIC INTERPOLATION ----------
    period = np.zeros(nframes)

    for i in range(nframes):
        l = lag[i]
        if 1 < l < (win - 1):
            alpha = d_norm[i, l - 1]
            beta = d_norm[i, l]
            gamma = d_norm[i, l + 1]
            peak = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
        else:
            peak = 0  # edge case
        period[i] = (l - 1) + peak

    f0_yin = fs / period  # instantaneous pitch estimate

    # ---------- STEP 5: FILTER OUT <10 Hz ----------
    time_yin = np.linspace(0, len(x) / fs, len(f0_yin))
    f0_yin[f0_yin < 10] = np.nan
    time_yin[f0_yin < 10] = np.nan

    # ---------- OPTIONAL silent frame classification ----------
    if silent_classifier is not None:
        f0_yin = silent_classifier(x_frame, f0_yin)

    # ---------- Median filter smoothing ----------
    f0_yin = medfilt(f0_yin, kernel_size=11)

    # If nothing detected:
    if np.all(np.isnan(f0_yin)):
        return np.nan, np.nan, np.nan, np.nan

    # ---------- STEP 6: SEGMENT DETECTION ----------
    idx = ~np.isnan(f0_yin)
    f0_short = f0_yin[idx]
    t_short = time_yin[idx]

    # Detect segment boundaries
    Start = [0]
    End = []

    for i in range(1, len(f0_short) - 1):
        if (abs(f0_short[i] - f0_short[i - 1]) > delta_f or
            abs(t_short[i] - t_short[i - 1]) > delta_t):
            Start.append(i)
            End.append(i - 1)

    End.append(len(f0_short) - 1)

    # Remove segments shorter than minimum duration
    dt = time_yin[1] - time_yin[0]
    min_len = int(round(signal_mini_duration_s / dt))

    S2 = []
    E2 = []
    for s, e in zip(Start, End):
        if (e - s) >= min_len:
            S2.append(s)
            E2.append(e)
    Start, End = S2, E2

    # ---------- Segment merging ----------
    S_new, E_new = Start.copy(), End.copy()

    for i in range(1, len(Start)):
        if t_short[Start[i]] - t_short[End[i - 1]] < delta_t:
            S_new[i] = None
            E_new[i - 1] = None

    Start = [s for s in S_new if s is not None]
    End = [e for e in E_new if e is not None]

    if len(End) > 0:
        End[-1] = len(t_short) - 1

    # Reject invalid segments (end <= start)
    valid = [i for i, (s, e) in enumerate(zip(Start, End)) if e > s]
    Start = [Start[i] for i in valid]
    End = [End[i] for i in valid]

    # ---------- FINAL reconstruction on full time vector ----------
    time_full = np.arange(N) / fs
    f0_final = np.full_like(time_full, np.nan)

    for s, e in zip(Start, End):
        t_seg = t_short[s:e+1]
        f_seg = f0_short[s:e+1]
        # interpolate into continuous time
        mask = (time_full >= t_seg[0]) & (time_full <= t_seg[-1])
        f0_final[mask] = np.interp(time_full[mask], t_seg, f_seg)

    return time_full, f0_final, Start, End

def pitch_track_segments(f0, time, ampl, delta_t, delta_f, signal_mini_duration):
    f0 = np.asarray(f0).copy()
    time = np.asarray(time).copy()
    ampl = np.asarray(ampl).copy()

    # 1. Find valid (non-NaN) samples
    index = ~np.isnan(f0)
    f0_short = f0[index]
    ampl_short = ampl[index]
    t_short = time[index]

    # -----------------------------------------------------------
    # 2. Detect contour boundaries (Start and End indices)
    # -----------------------------------------------------------
    Start = [0]    # MATLAB starts at 1 → Python at 0
    End = []

    for i in range(1, len(f0_short) - 1):
        if abs(f0_short[i] - f0_short[i - 1]) > delta_f or abs(t_short[i] - t_short[i - 1]) > delta_t:
            Start.append(i)
            End.append(i - 1)

    End.append(len(f0_short) - 1)
    Start = np.array(Start, dtype=float)
    End   = np.array(End, dtype=float)

    # -----------------------------------------------------------
    # 3. Remove tracks shorter than signal_mini_duration
    # -----------------------------------------------------------
    for i in range(len(Start)):
        s = int(Start[i])
        e = int(End[i])
        if (e - s) < signal_mini_duration:
            f0_short[s:e+1] = np.nan
            ampl_short[s:e+1] = np.nan
            t_short[s:e+1] = np.nan
            Start[i] = np.nan
            End[i] = np.nan

    # Filter out invalid after cleaning
    mask = ~np.isnan(Start)
    Start = Start[mask].astype(int)
    End = End[mask].astype(int)

    # -----------------------------------------------------------
    # 4. Remove the deleted samples from f0,time,ampl
    # -----------------------------------------------------------
    valid = ~np.isnan(f0_short)
    f0_short = f0_short[valid]
    ampl_short = ampl_short[valid]
    t_short = t_short[valid]

    # indices must be recomputed relative to the shortened arrays
    # Simplest approach: recompute Start/End on the cleaned short array
    # This is identical to MATLAB behavior
    if len(f0_short) == 0:
        return f0_short, t_short, ampl_short, np.array([]), np.array([])

    # After deletion Start/End must be mapped again
    # Recompute contiguous blocks of valid data
    blocks = np.where(np.diff(valid.astype(int)) != 0)[0] + 1
    # split f0_short by blocks
    segs = np.split(np.arange(len(valid)), blocks)

    Start = []
    End = []
    for seg in segs:
        if len(seg) >= signal_mini_duration:  # apply same rule
            Start.append(seg[0])
            End.append(seg[-1])

    Start = np.array(Start, dtype=int)
    End   = np.array(End, dtype=int)

    # -----------------------------------------------------------
    # 5. Merge neighbouring segments if the gap < delta_t
    # -----------------------------------------------------------
    if len(Start) > 1:
        Start_new = Start.copy()
        End_new = End.copy()

        for i in range(1, len(Start)):
            if t_short[Start[i]] - t_short[End[i-1]] < delta_t:
                Start_new[i] = -1   # mark for removal
                End_new[i-1] = -1

        Start = Start_new[Start_new >= 0]
        End   = End_new[End_new >= 0]

        if len(End) > 0:
            End[-1] = len(t_short) - 1

    return f0_short, t_short, ampl_short, Start, End
