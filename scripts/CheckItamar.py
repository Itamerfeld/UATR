import numpy as np

# --- Helper Functions ---

def awgn(signal, snr_linear):
    """
    Python implementation of MATLAB's awgn(sig, snr, 'measured', 'linear').
    
    Parameters:
        signal: The input signal vector
        snr_linear: The Signal-to-Noise ratio (linear scale, not dB)
    """
    # Measure signal power
    sig_power = np.mean(np.abs(signal) ** 2)
    
    # Calculate required noise power
    # SNR_linear = P_signal / P_noise  =>  P_noise = P_signal / SNR_linear
    noise_power = sig_power / snr_linear
    
    # Generate Gaussian noise
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    
    return signal + noise

# Placeholders for your custom detection functions
# You will need to replace the logic inside these with your actual python implementations
def detect_itamar(rx_signal, threshold):
    # TODO: Paste logic from DetectItamar.m here
    return 0 

def detect_demon(rx_signal, threshold):
    # TODO: Paste logic from DetectDemon.m here
    return 0

# --- Main Simulation Code ---

# Parameters
snr_vec = np.linspace(10, 5, 50)
fc = 10e3
fs = 96e3
ts = 20
num_sim = 100
num_th = 10

# Threshold vectors
th_itamar = np.linspace(0.1, 0.9, num_th)
th_demon = np.linspace(0.1, 0.9, num_th)

# Time and Signal
# Note: cast num_points to int for Python indexing
num_points = int(ts * fs)
t = np.linspace(0, ts, num_points)
sig = np.sqrt(2) * np.sin(2 * np.pi * t * fc)

# Pre-allocate storage arrays
# Shape is (SNR steps, Simulations, Threshold steps)
detect_vec_itamar = np.zeros((len(snr_vec), num_sim, num_th))
detect_vec_demon = np.zeros((len(snr_vec), num_sim, num_th))

# Loops
print("Starting simulation...")

for snr_ind, current_snr in enumerate(snr_vec):
    # Optional: Print progress
    # print(f"Processing SNR: {current_snr:.2f} (Index {snr_ind})")
    
    for sim_ind in range(num_sim):
        # Generate Noisy Signal (using 'measured' and 'linear' equivalent)
        rx = awgn(sig, current_snr)
        
        for th_ind in range(num_th):
            # Run detection functions
            detect_vec_itamar[snr_ind, sim_ind, th_ind] = detect_itamar(rx, th_itamar[th_ind])
            detect_vec_demon[snr_ind, sim_ind, th_ind] = detect_demon(rx, th_demon[th_ind])

print("Simulation complete.")