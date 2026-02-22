from ship_noise_pkl_model import model_predict
from Process_demon_signal_1s_24khz import process_demon
# import threading
import config
from datetime import datetime
# import os
# import subprocess
import shutil
import pandas as pd
import pickle

# def rec(duration,records_path='/home/pi/data/'):
#     timestr = datetime.utcnow().strftime("%Y%m%d-%H%M%S.%f")[:-3]
#     # cmd_record = ['arecord', '-Dplughw:0,0', '-c2', '-r24000', os.path.join(records_path, timestr + '.wav'), '-d ' + str(duration)]
#     cmd_record = ['arecord', '-c2', '-r24000', os.path.join(records_path, timestr + '.wav'), '-d ' + str(duration)]
#     try:
#         # cmd_record[5] = '-d ' + str(duration)
#         # cmd_record[4] = os.path.join(records_path, timestr + '.wav')
#         p = subprocess.run(cmd_record, timeout=duration )
#     except Exception as e:
#         print('')

def save_predictions_to_csv(all_directories_dict, csv_path="predictions_summary.csv"):
    flat_data = []
    for directory, files_dict in all_directories_dict.items():
        for file_name, prediction in files_dict.items():
            if prediction["label"] == 1:
                flat_row = {
                    "directory(date)": directory,
                    "file_name": file_name,
                    "label": prediction["label"],
                    "probability_no_ship": prediction["probability_no-ship"],
                    "probability_ship": prediction["probability_ship"]
                }
                flat_data.append(flat_row)

    df = pd.DataFrame(flat_data)
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved to {csv_path}")

def main_processAndPredict(directory):
    threshold = config.THRESHOLD
    preds = model_predict()
    preProcesser = process_demon()
    result_dict = {}
    now = datetime.now()
    date = now.strftime("%Y-%m-%d_%H-%M-%S")
    try:
        # process the wav files with demon
        print(f'started Processing data')
        files_to_predict_flag , wav_files_predicted = preProcesser.main_process(directory)
        print(f'finished Processing data')
        if files_to_predict_flag:
            # classify the processed wav file (1 sec each wav)
            print(f'started prediction logging')
            result_dict = preds.main_pred(threshold)
            if config.SAVE_PKL_FILES:
                with open(f'{config.LOGS_PATH}/{date}.pkl','ab') as pkl_file:
                                pickle.dump(result_dict,pkl_file)
            print(f'finished prediction')

            # delete the processed demon wav files
            print(f'started deleting demon files')
            shutil.rmtree(config.DEMON_PATH)
            print(f'finished deleting demon files')
    except Exception as e:
            print(e)
            with open(f'{config.LOGS_PATH}/{date}.txt','a') as log_file:
                log_file.write(str(e))
                log_file.flush()
            print(f'started deleting demon files')
            shutil.rmtree(config.DEMON_PATH)
            print(f'finished deleting demon files')

    return result_dict , wav_files_predicted



# if __name__ == "__main__":
#     main_processAndPredict()