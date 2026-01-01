
import os
import numpy as np
import scipy
from scipy.io import wavfile
import soundfile as sf  # more robust than scipy.io.wavfile

# =======================
# motorboat
# =======================
org_data_file = "./data/from_mark/IRB.1.48000.M36-V35-100/IRB.1.20230823T050001Z.wav"

if not os.path.exists(org_data_file):
    print(f"Error: File not found: {org_data_file}")
else:
    try:
        motorboat_data, motorboat_fs = sf.read(org_data_file)
        print(f"Successfully read file. Sample rate: {motorboat_fs}")
        print(f"Shape: {motorboat_data.shape}, Duration: {len(motorboat_data)/motorboat_fs:.2f}s")
    except Exception as e:
        print(f"Error reading file: {e}")
        print("File may be corrupted or in an unsupported format")

start_ix = 8 * 60 * 60 * motorboat_fs + 5 * 60 * motorboat_fs + 30 * motorboat_fs  # 8 hours + 5 minutes + 30 seconds
end_ix = start_ix + 60 * motorboat_fs  # 1 minute slice
motorboat_data = motorboat_data[start_ix:end_ix]
sf.write("data/motorboat_1m.wav", motorboat_data, motorboat_fs)
print("Motorboat 1-minute slice saved at data/motorboat_1m.wav")

# # =======================
# # large ship
# # =======================
# org_data_file = "./data/from_mark/IRB.1.48000.M36-V35-100/IRB.1.20230823T131707Z.wav"

# if not os.path.exists(org_data_file):
#     print(f"Error: File not found: {org_data_file}")
# else:
#     try:
#         motorboat_data, motorboat_fs = sf.read(org_data_file)
#         print(f"Successfully read file. Sample rate: {motorboat_fs}")
#         print(f"Shape: {motorboat_data.shape}, Duration: {len(motorboat_data)/motorboat_fs:.2f}s")
#     except Exception as e:
#         print(f"Error reading file: {e}")
#         print("File may be corrupted or in an unsupported format")

# large_ship_data, large_ship_fs = sf.read(org_data_file)
# start_ix = 2 * 60 * 60 * large_ship_fs + 36 * 60 * large_ship_fs + 30 * large_ship_fs  # 2 hours + 36 minutes + 30 seconds
# end_ix = start_ix + 60 * large_ship_fs  # 1 minute slice
# large_ship_data = large_ship_data[start_ix:end_ix]
# sf.write("data/large_ship_1m.wav", large_ship_data, large_ship_fs)
# print("Large ship 1-minute slice saved at data/large_ship_1m.wav")

# # =======================
# # bg noise 2
# # =======================
# org_data_file = "./data/from_mark/IRB.1.48000.M36-V35-100/IRB.1.20230823T050001Z.wav"

# if not os.path.exists(org_data_file):
#     print(f"Error: File not found: {org_data_file}")
# else:
#     try:
#         bg_noise_2_data, bg_noise_2_fs = sf.read(org_data_file)
#         print(f"Successfully read file. Sample rate: {bg_noise_2_fs}")
#         print(f"Shape: {bg_noise_2_data.shape}, Duration: {len(bg_noise_2_data)/bg_noise_2_fs:.2f}s")
#     except Exception as e:
#         print(f"Error reading file: {e}")
#         print("File may be corrupted or in an unsupported format")

# start_ix = 78 * 60 * bg_noise_2_fs  # 78 minutes
# end_ix = start_ix + 60 * bg_noise_2_fs  # 1 minute slice
# bg_noise_2 = bg_noise_2_data[start_ix:end_ix]
# sf.write("data/bg_noise_2_1m.wav", bg_noise_2, bg_noise_2_fs)
# print("BG Noise 2 1-minute slice saved at data/bg_noise_2_1m.wav")
