from pythonosc import dispatcher
from pythonosc import osc_server
import sys
from datetime import datetime
from sklearn.preprocessing import StandardScaler

import numpy as np
import torch

from EEGNet import EEGNet

sample = np.zeros((1664,24))
SAMPLE_SIZE = 1664
sc = StandardScaler()

abs_delta = np.zeros((1,4))
abs_theta = np.zeros((1,4))
abs_alpha = np.zeros((1,4))
abs_beta = np.zeros((1,4))
abs_gamma = np.zeros((1,4))

# model = EEGNet().cuda(0)
# device = torch.device('cuda')
# model.load_state_dict(torch.load('final_model_2021-12-20 194851'))
# model.eval()
# model.to(device)

f = open ('new_real_time_data.csv', 'w+')
f.write('TimeStamp,Prediction\n')

i = 0

def eeg_handler(address: str,*args):

  global sample
  global abs_delta
  global abs_theta
  global abs_alpha
  global abs_beta
  global abs_gamma
  print('working')

  global i

  global model

  abs_values = np.hstack((abs_delta, abs_theta, abs_alpha, abs_beta, abs_gamma))

  sample [i, 20] = args[0]
  sample [i, 21] = args[1]
  sample [i, 22] = args[2]
  sample [i, 23] = args[3]

  sample[i, :20] = abs_values
  i += 1
  if i == 1664:
    i = 0
    # Model Prediction
    with torch.no_grad():
      sample_numpy = np.load(r'C:\Users\rlpv9\MUSE\low\sample_800.npy', allow_pickle=True).astype(float)
      print(type(sample_numpy))
      sample_torch = torch.Tensor(sample_numpy).unsqueeze(0).unsqueeze(0).float()
      sample_torch = sample_torch.to(device)
      pred =  model(sample_torch)
    
    pred = pred.cpu().detach().numpy()
    prediction = np.argmax(pred)


    if prediction == 0:
      print("Blood glucose is high. Pump insulin!")
    elif prediction == 2:
      print("Blood glucose is low. Drink some juice!")
    else:
      print("Your blood glucose is good. Sweet dreams!")
    
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S.%f")
    try:
      f.write(timestampStr+","+str(prediction)+"\n")
    except KeyboardInterrupt:
      f.close()
      sys.exit(0)

def delta_absolute_handler(address: str,*args):
  global abs_delta

  abs_delta[0,0] = args[0]
  abs_delta[0,1] = args[1]
  abs_delta[0,2] = args[2]
  abs_delta[0,3] = args[3]
  print(abs_delta)

def theta_absolute_handler(address: str,*args):
  global abs_theta

  abs_theta[0,0] = args[0]
  abs_theta[0,1] = args[1]
  abs_theta[0,2] = args[2]
  abs_theta[0,3] = args[3]

def alpha_absolute_handler(address: str,*args):
  global abs_alpha

  abs_alpha[0,0] = args[0]
  abs_alpha[0,1] = args[1]
  abs_alpha[0,2] = args[2]
  abs_alpha[0,3] = args[3]


def beta_absolute_handler(address: str,*args):
  global abs_beta

  abs_beta[0,0] = args[0]
  abs_beta[0,1] = args[1]
  abs_beta[0,2] = args[2]
  abs_beta[0,3] = args[3]

def gamma_absolute_handler(address: str,*args):
  global abs_gamma

  abs_gamma[0,0] = args[0]
  abs_gamma[0,1] = args[1]
  abs_gamma[0,2] = args[2]
  abs_gamma[0,3] = args[3]

#Main
if __name__ == "__main__":
  #Init Muse Listeners    
  dispatcher = dispatcher.Dispatcher()
  dispatcher.map("/muse/eeg", eeg_handler)
  dispatcher.map("/muse/elements/delta_absolute", delta_absolute_handler)
  dispatcher.map("/muse/elements/theta_absolute", theta_absolute_handler)
  dispatcher.map("/muse/elements/alpha_absolute", alpha_absolute_handler)
  dispatcher.map("/muse/elements/beta_absolute", beta_absolute_handler)
  dispatcher.map("/muse/elements/gamma_absolute", gamma_absolute_handler)

  # Connect to OSC device
  server = osc_server.ThreadingOSCUDPServer(('192.168.1.158', 5000), dispatcher)
  print("Listening on UDP port " + str(5000))
  # Loop through messages
  server.serve_forever()