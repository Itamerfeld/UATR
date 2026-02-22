import os
import glob
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import firwin, lfilter, hilbert
# import matplotlib.pyplot as plt
import soundfile as sf
import shutil
import config

# from scipy.io import savemat


class process_demon():
    def __init__(self):
        self.current_folder = os.getcwd()
        # self.data_folder = os.path.join(self.current_folder,'data')

    def demo(self,y, Fs, filename, N=1):
        # Define filter parameters
        # fpass_band = [3000, 10000] # new is 100,1500
        # fpass_low = 300 # new is 100
        fpass_band = [100, 1500] # new is 100,1500
        fpass_low = 100 # new is 100
        # Design a bandpass filter (3-10 kHz)
        nyquist_rate = Fs / 2
        b_bandpass = firwin(1024, [fpass_band[0] / nyquist_rate, fpass_band[1] / nyquist_rate], pass_zero=False)
        y1 = lfilter(b_bandpass, 1.0, y)
        
        # Envelope detection
        analytic_signal = hilbert(y1)
        envelope = np.abs(analytic_signal) # type: ignore
        
        # FFT and extract desired frequency range
        y2 = np.abs(np.fft.fftshift(np.fft.fft(envelope)))
        Faxis = np.linspace(-Fs/2, Fs/2, len(y2))
        pos = np.where((Faxis > 2) & (Faxis < fpass_low))
        demon = y2[pos]
        
        # Calculate SNR
        M = np.max(demon)
        N = np.mean(demon)
        SNR = 10 * np.log10(M / N)
        
        # Normalize and save the demon signal
        demon = demon / np.max(demon)
        sf.write(filename[:-4] + 'demo.wav', demon, int(Fs))
        
        return SNR, demon

    def process_files(self,files, output_folder):
        SNR_all = []
        fileName_all = []
        
        for j, filepath in enumerate(files):

            file_name = filepath.split('/')[-1]
            # print(file_name)
            # dest = f'processed_data/{file_name}'

            try:
                if config.DEBUG_MODE:
                    print(f"Processing file {j + 1}/{len(files)}: {os.path.basename(filepath)}")
                
                # Read and downsample the audio
                Fs, y = wavfile.read(filepath)
                # print(f'y before 1 channel: {y[:5]}')
                if y.ndim > 1:
                    y = y[:,0]
                # print(f'y after 1 channel: {y[:5]}')
                y = y[::2]  # Downsample by a factor of 2
                Fs = Fs // 2
                N = 1  # Segment length in seconds
                
                # Segment the file and compute SNR for each segment
                for k in range(0, len(y), Fs * N):
                    segment = y[k:k + Fs * N]
                    if len(segment) < Fs * N:
                        continue  # Skip if segment is shorter than desired
                    
                    filename = os.path.join(output_folder, f"{k // Fs + 1}_{N}s_{os.path.basename(filepath)}")
                    
                    SNR, _ = self.demo(segment, Fs, filename, N)
                    
                    # Save SNR and filename info
                    SNR_all.append(SNR)
                    fileName_all.append(f"{k // Fs + 1}_{N}s_{os.path.basename(filepath)}")
            except Exception as e:
                with open(f'{config.LOGS_PATH}/{file_name}.txt','a') as logFile:
                    logFile.write(str(e))
            # move processed file to processed files directory
            # shutil.move(filepath,dest)
            # print(f'moved {filepath} to {dest}')
            
        return SNR_all, fileName_all

    # def plot_demon(Faxis, demon, fpass=300):
    #     plt.plot(Faxis, demon)
    #     plt.xlabel("Frequency (Hz)")
    #     plt.ylabel("Amplitude")
    #     plt.title("Demon Spectrum")
    #     plt.grid(True)
    #     plt.show()


    def main_process(self,directory):
        wav_files = glob.glob(os.path.join(directory, '*.wav'))
        wav_files.sort()
        wav_files = wav_files[:-1]
        if wav_files:
            # Process Boat files
            # output_folder = os.path.join(self.current_folder, 'Demon_1s_24khz')
            output_folder = config.DEMON_PATH
            os.makedirs(output_folder, exist_ok=True)
            SNR_boat, fileNames_boat = self.process_files(wav_files, output_folder)
            # savemat("SNR_Boat_Demon_1s_24khz.mat", {'SNR_all': SNR_boat, 'fileName_all': fileNames_boat})
            return True , wav_files
        else:
            print('no new files to process')
            return False , None

