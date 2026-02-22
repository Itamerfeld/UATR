"""
Edited by : Eyal Cohen, Based on :
Ship noise test.py
------------------
Oct 2022
------------------
Modified by
Weihua
University of haifa

Needed files to run this script:
1. model_24khz.pkl
2. mobilenet1d.py

"""
import datetime
import config
import torch, torchaudio
import numpy as np
from torch.autograd import Variable
from mobilenet1d import MobileNetV2
# import scipy.io as scio
import os
class model_predict():

    def __init__(self):
        self.pt_file = config.SOUND_MODEL_PATH
        self.wavs_directory_path = config.DEMON_PATH
        self.to_log = config.TO_LOG_PREDICTION

    def run_model(self,log_file,threshold):
        wav_list = os.listdir(self.wavs_directory_path)
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        wlen = 297  # length of each segment (12.4 ms at 24 kHz)
        result_dict = {}
        for ii in range(len(wav_list)):
            if config.DEBUG_MODE:
                print(f'processing wav {ii+1}/{len(wav_list)}')
            wav_file = f'{self.wavs_directory_path}/{wav_list[ii]}'
            # Initialize MobileNet model
            class_lay = list([2])  # Two classes: boat and non-boat
            MOBILENET_net = MobileNetV2(num_classes=class_lay) # type: ignore
            # pt_file = './models/model_24khz.pkl'
            # checkpoint_load = torch.load(self.pt_file, map_location=torch.device('cpu'))
            checkpoint_load = torch.load(self.pt_file,map_location=torch.device('cpu'),weights_only=True)
            MOBILENET_net.load_state_dict(checkpoint_load['MOBILENET_model_par'])
            MOBILENET_net.eval()
            
            # Load signal
            [signal, fs] = torchaudio.load(wav_file, backend="soundfile")
            signal = signal[0, :]  # Select only the first channel → shape becomes [samples] (no need for this if wav file is only 1 channel, mono)

            # print("signal length:" , signal.shape[1] , signal.shape[1]//wlen)
            # Process in chunks of 297 samples

            # sig_arr = signal.unsqueeze(0) # good for 1 channel wavs only
            sig_arr = signal.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, samples]


            # Prepare input for model
            inp = Variable(sig_arr.float())  # Reshape and convert to Variable

            # Get prediction
            with torch.no_grad():
                pout = MOBILENET_net(inp)
                pred = torch.max(pout, dim=1)[1] # type: ignore
                ############# new addition 4/7/2025
                # pred.item() ==> 0 = ship, 1 = no ship
                # score = pout[0][0] / pout[0][1]
                # class_label = 0 if pred.item() == 0 and score > 1 else 1

                # softmax(a, b) = [e^a / (e^a + e^b), e^b / (e^a + e^b)]
                probs = torch.softmax(pout[0], dim=0)
                no_ship_prob = probs[0].item()  # probability it's not a ship
                ship_prob = probs[1].item()  # probability it's a ship
                # threshold (can tweak 0.5 to tune sensitivity)
                class_label = 1 if ship_prob > threshold else 0

                result_dict[wav_list[ii]] = {"label" : class_label , "probability_no-ship": no_ship_prob , "probability_ship": ship_prob}
                ###################################
                # if not self.to_log:
                #     print(f"\nFile: {wav_list[ii]}\n")
                #     print(f"Prediction: {class_label}, Probability no-ship: {no_ship_prob}, Probability ship: {ship_prob}\n")
                    # print(f"Prediction: {pred.item()}, Logits: {pout[0]}\n")
                    
                    # Ilan Shahar truth table labeling based on model's output
                    # if (pred.item() and pout[0][0].item() >= 0.4):
                    #     print('False negative')
                    # elif (not pred.item() and pout[0][0].item() < 0.5):
                    #     print('False positive')
                # else:
                if self.to_log:
                    log_file.write(f"\nFile: {wav_list[ii]}\n")
                    log_file.write(f"Prediction: {class_label}, Probability no-ship: {no_ship_prob}, Probability ship: {ship_prob}\n")
                    # log_file.write(f"Prediction: {pred.item()}, Logits: {pout[0]}\n")
                
        return result_dict
                        
    def main_pred(self,threshold):
        result_dict = {}
        # Variables        
        time_stamp = datetime.datetime.now()
        # Start running model
        log_file = f'{config.LOGS_PATH}/ship_noise_pkl_' + time_stamp.strftime("%y-%m-%d_%H-%M-%S") +'.log'
        if self.to_log:
            log_file =  open(log_file, 'a')
        try:
            result_dict = self.run_model(log_file,threshold)
        except Exception as e:
            print(e)
        finally:
            if self.to_log:
                log_file.close() # type: ignore
        return result_dict

# if __name__ == "__main__":
#     main()
  
