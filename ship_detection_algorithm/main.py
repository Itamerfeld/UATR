from processPredictDelete import main_processAndPredict
import config
import sys
sys.path.append("/home/pi")
import time 
from datetime import datetime, time as dt_time
import os
# import subprocess
# import shutil
# import pandas as pd
import csv
import shutil
# import numpy as np
from scipy.io import wavfile
from sensors import modem_control
# import pickle as pkl
from collections import defaultdict
import threading
from image_process_sys import yolo_prediction , camera_control , snapShot
from mail_sys import email_detection , email_config
from leq_sel_1_3oct_1min import leq_SEL_1_3oct_1min

yolo_instance = yolo_prediction.pt_yolo_predict()
# camera_controller_object = camera_control.camera_controller() # needed only for IP camera

if config.CONTROL_MODEM_FLAG == True:
    modem_controller = modem_control.Modem_controller()
last_email_time = 0

def save_predictions_to_csv(predictions_dict, csv_filename=f'{config.LOGS_PATH}/predictions_log.csv'):
    file_exists = os.path.isfile(csv_filename)
    write_header = not file_exists or os.stat(csv_filename).st_size == 0

    with open(csv_filename, mode='a', newline='') as csv_file:
        fieldnames = ["file_name", "label", "probability_no_ship", "probability_ship"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for file_name, prediction in predictions_dict.items():
            row = {
            "file_name": file_name,
            "label": prediction["label"],
            "probability_no_ship": prediction["probability_no-ship"],
            "probability_ship": prediction["probability_ship"]
            }
            writer.writerow(row)


def delete_processed_wavs(wav_files_predicted):
    delete_flag = 0
    # processed_wav_list = os.listdir(data_path)
    # processed_wav_list.sort()
    # processed_wav_list = processed_wav_list[:-1]
    if delete_flag:
        for file in wav_files_predicted:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting file {file}: {e}")
                continue
        print("finished deleting processed wavs")
    else:
        for wav_path in wav_files_predicted:
            base_name = os.path.basename(wav_path)
            # Move original WAV to moved_wavs directory
            destination = os.path.join(config.MOVED_WAVS_PATH, base_name)
            shutil.move(wav_path, destination)
        print("finished moving processed wavs")
    


def save_OneSec_raw_wav(data_path, data, output_path):
    # Step 1: Group detections per base WAV file
    detections_per_wav = defaultdict(list)  # base_name -> list of (second, probability_ship, key)
    varified_detection = False
    for key in data:
        if data[key]["label"] == 1:
            second = int(key.split('_')[0]) - 1  # 0-indexed
            base_name = key.split('_', 1)[1].replace('demo', '').replace('1s_', '')
            prob = data[key]["probability_ship"]
            detections_per_wav[base_name].append((second, prob, key))
    # print(detections_per_wav)
    # Step 2: For each WAV with ≥3 detections, save the one with highest probability
    for base_name, detections in detections_per_wav.items():
        if len(detections) < config.SAME_WAV_DETECTIONS:
            continue
        
        varified_detection = True

        # Get detection with max probability
        best_detection = max(detections, key=lambda x: x[1])  # (second, probability_ship, key)
        second = best_detection[0]

        # Read audio
        wav_path = os.path.join(data_path, base_name)
        try:
            sample_rate, samples = wavfile.read(wav_path)
        except FileNotFoundError:
            print(f"File not found: {wav_path}")
            continue

        # Use only the first channel if stereo
        if samples.ndim > 1:
            samples = samples[:, 0]

        # Extract segment
        start_sample = second * sample_rate
        end_sample = (second + 1) * sample_rate
        if end_sample > len(samples):
            print(f"Warning: {base_name} shorter than {second + 1} seconds.")
            continue

        one_second_clip = samples[start_sample:end_sample]

        # Save clip
        os.makedirs(output_path, exist_ok=True)
        raw_1sec_name = base_name.replace('.wav','')
        output_name = f"{output_path}/{raw_1sec_name}_{second}s_clip.wav"
        wavfile.write(output_name, sample_rate, one_second_clip)
        print(f"Saved: {output_name} (prob={best_detection[1]:.4f})")

        # Move original WAV to moved_wavs directory
        destination = os.path.join(config.MOVED_WAVS_PATH,base_name)
        shutil.move(wav_path, destination)
    return varified_detection

def is_in_block_window():
    now = datetime.now().time()
    return dt_time(10, 0) <= now <= dt_time(11, 0)

def main_target():
    global last_email_time
    # now = datetime.now()
    # date = now.strftime("%Y-%m-%d_%H-%M-%S")
    data_dict = {}
    data_path = config.DATA_PATH
    raw_oneSec_detection_path = config.RAW_ONESEC_DETECTION_PATH
    varified_detection = False
    if not config.DEBUG_MODE:
        try:
            data_dict , wav_files_predicted = main_processAndPredict(data_path)
            varified_detection = save_OneSec_raw_wav(data_path,data_dict,raw_oneSec_detection_path)
            # if not varified_detection:
            delete_processed_wavs(wav_files_predicted)
            # if varified_detection:
            #     if config.CONTROL_MODEM_FLAG == True:
            #         modem_controller.power_on()
            #         time.sleep(20)  # wait for modem to stabilize
        #     #################################################################################################
        #         # Deactivating image processing as long as there is an issue with the power consumption)
        #         print('Taking snapshot for verified detection...')
        #         snapShot.capture_snapshot_IP_zoom() # IP camera version
        #         # snapShot.take_snapShot() # USB camera version
        #         yolo_detection_flag = yolo_instance.predict_multi_images()
        #         print(f"Classified snapshot, found fishing vessle:{yolo_detection_flag}")
        #         if yolo_detection_flag and config.MAIL_RESULTS:
        #             if time.time() - last_email_time > email_config.TIME_BETWEEN_EMAILS:
        #                 email_detection.email_last_detection()
        #                 print('sent email with attachments')
        #                 last_email_time = time.time()
        #             else: print('Email was sent recently, skipping current email report')
        #     #################################################################################################
            #     if config.CONTROL_MODEM_FLAG == True and not is_in_block_window():
            #         modem_controller.power_off()
        except Exception as e:
            print(e)
            # camera_controller_object.clean_recources() # needed only for IP camera
                
        if data_dict:
            save_predictions_to_csv(data_dict)

def main():
    scan_print_flag = True
    try:
        while True:
            if scan_print_flag:
                print("scanning for new wav files...")
                scan_print_flag = False
            wav_list = os.listdir(config.DATA_PATH)
            wav_list.sort()
            wav_list = wav_list[:-1] # in order to not process a non-completed wav file
            if wav_list:
                scan_print_flag = True
                print(f"wav file detected: {wav_list}")
                run_system_thread = threading.Thread(target=main_target , args=[] , daemon=True)
                run_system_thread.start()
                run_system_thread.join()
            time.sleep(2.5)
            moved_wavs_list = os.listdir(config.MOVED_WAVS_PATH)
            if moved_wavs_list:
                moved_wavs_list.sort()
                print(f"moved wav files detected: {moved_wavs_list}")
                for file in moved_wavs_list:
                    leq_SEL_1_3oct_1min(config.MOVED_WAVS_PATH+'/'+file)
                    os.remove(config.MOVED_WAVS_PATH+'/'+file)
                print("finished calculating leq vectors for moved wavs")
                    
    except Exception as e:
        print(e)
        # camera_controller_object.clean_recources() # needed only for IP camera






if __name__ == "__main__":
    main()