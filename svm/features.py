import numpy as np
from numpy.ma.core import array
from scipy import signal
from scipy.stats import kurtosis, skew
import os
import statsmodels.api as sm
import antropy as ant
import math
from tqdm import tqdm
import random
from pickle import dump

# Required input defintions are as follows;
# time:   Time between samples
# band:   The bandwidth around the centerline freqency that you wish to filter
# freq:   The centerline frequency to be filtered
# ripple: The maximum passband ripple that is allowed in db
# order:  The filter order.  For FIR notch filters this is best set to 2 or 3,
#         IIR filters are best suited for high values of order.  This algorithm
#         is hard coded to FIR filters
# filter_type: 'butter', 'bessel', 'cheby1', 'cheby2', 'ellip'
# data:         the data to be filtered
def Implement_Notch_Filter(time, band, freq, ripple, order, filter_type, data):
    from scipy.signal import iirfilter, lfilter
    fs   = 1/time
    nyq  = fs/2.0
    low  = freq - band/2.0
    high = freq + band/2.0
    low  = low/nyq
    high = high/nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop',
                     analog=False, ftype=filter_type)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def bandpower(data, sf, band, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Requires MNE-Python >= 0.14.

    Parameters
    ----------
    data : 1d-array
      Input signal in the time-domain.
    sf : float
      Sampling frequency of the data.
    band : list
      Lower and upper frequencies of the band of interest.
    method : string
      Periodogram method: 'welch' or 'multitaper'
    window_sec : float
      Length of each window in seconds. Useful only if method == 'welch'.
      If None, window_sec = (1 / min(band)) * 2.
    relative : boolean
      If True, return the relative power (= divided by the total power of the signal).
      If False (default), return the absolute power.

    Return
    ------
    bp : float
      Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    from mne.time_frequency import psd_array_multitaper

    band = np.asarray(band)
    low, high = band


    psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                        normalization='full', verbose=0)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

def main():
  cwd = os.getcwd()
  # sampling frequency
  sf = 256

  random.seed(10)

  # Number of sampling points per sample, 5 seconds of sample
  npoints_window = 5 * 256

  # Sampling points for max, min and derivative features
  p2 = int(npoints_window/2)
  p4 = int(npoints_window/4)

  sensors = ['tp9', 'af7', 'af8', 'tp10']
  wave_types = ['delta', 'theta', 'alpha', 'beta', 'gamma']
  # Create samples for targets low(0), high(1), and normal(2) glucose levels
  target_glucose_levels = ['low', 'high', 'normal']
  feature_names = ['abs_bp_delta_tp9', 'abs_bp_delta_af7', 'abs_bp_delta_af8', 'abs_bp_delta_tp10',
                    'abs_bp_theta_tp9', 'abs_bp_theta_af7', 'abs_bp_theta_af8', 'abs_bp_theta_tp10',
                    'abs_bp_alpha_tp9', 'abs_bp_alpha_af7', 'abs_bp_alpha_af8', 'abs_bp_alpha_tp10',
                    'abs_bp_beta_tp9', 'abs_bp_beta_af7', 'abs_bp_beta_af8', 'abs_bp_beta_tp10',
                    'abs_bp_gamma_tp9', 'abs_bp_gamma_af7', 'abs_bp_gamma_af8', 'abs_bp_gamma_tp10',
                    'rel_bp_delta_tp9', 'rel_bp_delta_af7', 'rel_bp_delta_af8', 'rel_bp_delta_tp10',
                    'rel_bp_theta_tp9', 'rel_bp_theta_af7', 'rel_bp_theta_af8', 'rel_bp_theta_tp10',
                    'rel_bp_alpha_tp9', 'rel_bp_alpha_af7', 'rel_bp_alpha_af8', 'rel_bp_alpha_tp10',
                    'rel_bp_beta_tp9', 'rel_bp_beta_af7', 'rel_bp_beta_af8', 'rel_bp_beta_tp10',
                    'rel_bp_gamma_tp9', 'rel_bp_gamma_af7', 'rel_bp_gamma_af8', 'rel_bp_gamma_tp10',
                    'mean_tp9', 'mean_af7', 'mean_af8', 'mean_tp10',
                    'std_tp9', 'std_af7', 'std_af8', 'std_tp10',
                    'sk_tp9', 'sk_af7', 'sk_af8', 'sk_tp10',
                    'kt_tp9', 'kt_af7', 'kt_af8', 'kt_tp10',
  ]

  size = 0
  for s in sensors:
    for i in range(42):
      feature_names.append('ac_' + s + '_' +str(i))
      size +=1

  feature_names += [
                    'uv_tp9', 'uv_af7', 'uv_af8', 'uv_tp10',
                    'maxt_tp9', 'maxt_af7', 'maxt_af8', 'maxt_tp10',
                    'mint_tp9', 'mint_af7', 'mint_af8', 'mint_tp10',
                    'du_12_tp9', 'du_13_tp9', 'du_14_tp9', 'du_23_tp9', 'du_24_tp9', 'du_34_tp9',
                    'du_12_af7', 'du_13_af7', 'du_14_af7', 'du_23_af7', 'du_24_af7', 'du_34_af7',
                    'du_12_af8', 'du_13_af8', 'du_14_af8', 'du_23_af8', 'du_24_af8', 'du_34_af8',
                    'du_12_tp10', 'du_13_tp10', 'du_14_tp10', 'du_23_tp10', 'du_24_tp10', 'du_34_tp10',
                    'dmax_12_tp9', 'dmax_13_tp9', 'dmax_14_tp9', 'dmax_23_tp9', 'dmax_24_tp9', 'dmax_34_tp9',
                    'dmax_12_af7', 'dmax_13_af7', 'dmax_14_af7', 'dmax_23_af7', 'dmax_24_af7', 'dmax_34_af7',
                    'dmax_12_af8', 'dmax_13_af8', 'dmax_14_af8', 'dmax_23_af8', 'dmax_24_af8', 'dmax_34_af8',
                    'dmax_12_tp10', 'dmax_13_tp10', 'dmax_14_tp10', 'dmax_23_tp10', 'dmax_24_tp10', 'dmax_34_tp10',
                    'dmin_12_tp9', 'dmin_13_tp9', 'dmin_14_tp9', 'dmin_23_tp9', 'dmin_24_tp9', 'dmin_34_tp9',
                    'dmin_12_af7', 'dmin_13_af7', 'dmin_14_af7', 'dmin_23_af7', 'dmin_24_af7', 'dmin_34_af7',
                    'dmin_12_af8', 'dmin_13_af8', 'dmin_14_af8', 'dmin_23_af8', 'dmin_24_af8', 'dmin_34_af8',
                    'dmin_12_tp10', 'dmin_13_tp10', 'dmin_14_tp10', 'dmin_23_tp10', 'dmin_24_tp10', 'dmin_34_tp10',
                    'sampe_tp9', 'sampe_af7', 'sampe_af8', 'sampe_tp10',
                    'spece_tp9', 'spece_af7', 'spece_af8', 'spece_tp10'
                    ]
                    
  coherence_features = ['Cxy_tp9_af8', 'Cxy_tp9_af7', 'Cxy_tp9_tp10', 'Cxy_af8_af7', 'Cxy_af8_tp10', 'Cxy_af7_tp10']

  for c in coherence_features:
    for w in wave_types:
      feature_names.append(c + '_' + w)
  feature_names += ['fluct_tp9', 'fluct_af7', 'fluct_af8', 'fluct_tp10']

  n_features = len(feature_names)
  samples = []
  list_array_samples = []
  for target_id, target in enumerate(target_glucose_levels):
    data_txt = open(cwd + '\CMSC715-IoT\labeled_muse/' + target + '_0' , "r")
    lines = data_txt.readlines()
    data = np.zeros((len(lines)-1, 4))

    for j, x in enumerate(lines[1:]):
        x = x.replace('\n', '')
        result = x.split("\t")
        result = list(map(float, result))
        data[j,:] = np.array(result[1:])

    
    # Number of samples, 50% overlapping
    nsamples = int(math.floor(data.shape[0]/npoints_window)*2 - 1)

    print(target_id)
    print(len(data))
    print(nsamples)
    
    print('Processing ' + target + '... Target id: ' + str(i))
    level_samples = np.zeros((nsamples,n_features+1)).copy()
    for j in tqdm(range(nsamples)):
      level_samples[j,-1] = target_id      
      # Get sample of 5 sec from data with 50 % overlaping
      x = data[int(j*(npoints_window/2)):int(j*(npoints_window/2) +(npoints_window-1))]
      
      # Get TP9, AF7, AF8, TP10 data
      tp9 = x[:,0]
      af7 = x[:,1]
      af8 = x[:,2]
      tp10 = x[:,3]

      # Filter the sample with bandpass 70 Hz IIR filter
      tp9_f = Implement_Notch_Filter(1.0/300.0, 5, 70, 3, 3, 'cheby1', tp9)
      af7_f = Implement_Notch_Filter(1.0/300.0, 5, 70, 3, 3, 'cheby1', af7)
      af8_f = Implement_Notch_Filter(1.0/300.0, 5, 70, 3, 3, 'cheby1', af8)
      tp10_f = Implement_Notch_Filter(1.0/300.0, 5, 70, 3, 3, 'cheby1', tp10)

      # Extract features
      features = []
      ###############################
      ## FREQUENCY DOMAIN FEATURES ##
      ###############################

      # Absolute bandpowers delta
      level_samples[j,0] = abs_bp_delta_tp9 = bandpower(tp9_f, sf, [0.5, 4])
      level_samples[j,1] = abs_bp_delta_af7 = bandpower(af7_f, sf, [0.5, 4])
      level_samples[j,2] = abs_bp_delta_af8 = bandpower(af8_f, sf, [0.5, 4])
      level_samples[j,3] = abs_bp_delta_tp10 = bandpower(tp10_f, sf, [0.5, 4])

      features += [abs_bp_delta_tp9, abs_bp_delta_af7, abs_bp_delta_af8, abs_bp_delta_tp10]

      # Absolute bandpowers theta
      level_samples[j,4] = abs_bp_theta_tp9 = bandpower(tp9_f, sf, [4, 8])
      level_samples[j,5] = abs_bp_theta_af7 = bandpower(af7_f, sf, [4, 8])
      level_samples[j,6] = abs_bp_theta_af8 = bandpower(af8_f, sf, [4, 8])
      level_samples[j,7] = abs_bp_theta_tp10 = bandpower(tp10_f, sf, [4, 8])

      features += [abs_bp_theta_tp9, abs_bp_theta_af7, abs_bp_theta_af8, abs_bp_theta_tp10]

      # Absolute bandpowers alpha
      level_samples[j,8] = abs_bp_alpha_tp9 = bandpower(tp9_f, sf, [8, 12])
      level_samples[j,9] = abs_bp_alpha_af7 = bandpower(af7_f, sf, [8, 12])
      level_samples[j,10] = abs_bp_alpha_af8 = bandpower(af8_f, sf, [8, 12])
      level_samples[j,11] = abs_bp_alpha_tp10 = bandpower(tp10_f, sf, [8, 12])

      features += [abs_bp_alpha_tp9, abs_bp_alpha_af7, abs_bp_alpha_af8, abs_bp_alpha_tp10]

      # Absolute bandpowers beta
      level_samples[j,12] = abs_bp_beta_tp9 = bandpower(tp9_f, sf, [12, 30])
      level_samples[j,13] = abs_bp_beta_af7 = bandpower(af7_f, sf, [12, 30])
      level_samples[j,14] = abs_bp_beta_af8 = bandpower(af8_f, sf, [12, 30])
      level_samples[j,15] = abs_bp_beta_tp10 = bandpower(tp10_f, sf, [12, 30])

      features += [abs_bp_beta_tp9, abs_bp_beta_af7, abs_bp_beta_af8, abs_bp_beta_tp10]

      # Absolute bandpowers gamma
      level_samples[j,16] = abs_bp_gamma_tp9 = bandpower(tp9_f, sf, [30, 100], relative=True)
      level_samples[j,17] = abs_bp_gamma_af7 = bandpower(af7_f, sf, [30, 100], relative=True)
      level_samples[j,18] = abs_bp_gamma_af8 = bandpower(af8_f, sf, [30, 100], relative=True)
      level_samples[j,19] = abs_bp_gamma_tp10 = bandpower(tp10_f, sf, [30, 100], relative=True)
      
      features += [abs_bp_gamma_tp9, abs_bp_gamma_af7, abs_bp_gamma_af8, abs_bp_gamma_tp10]

      # Relative bandpowers delta
      level_samples[j,20] = rel_bp_delta_tp9 = bandpower(tp9_f, sf, [0.5, 4], relative=True)
      level_samples[j,21] = rel_bp_delta_af7 = bandpower(af7_f, sf, [0.5, 4], relative=True)
      level_samples[j,22] = rel_bp_delta_af8 = bandpower(af8_f, sf, [0.5, 4], relative=True)
      level_samples[j,23] = rel_bp_delta_tp10 = bandpower(tp10_f, sf, [0.5, 4], relative=True)

      features += [rel_bp_delta_tp9, rel_bp_delta_af7, rel_bp_delta_af8, rel_bp_delta_tp10]

      # Relative bandpowers theta
      level_samples[j,24] = rel_bp_theta_tp9 = bandpower(tp9_f, sf, [4, 8], relative=True)
      level_samples[j,25] = rel_bp_theta_af7 = bandpower(af7_f, sf, [4, 8], relative=True)
      level_samples[j,26] = rel_bp_theta_af8 = bandpower(af8_f, sf, [4, 8], relative=True)
      level_samples[j,27] = rel_bp_theta_tp10 = bandpower(tp10_f, sf, [4, 8], relative=True)

      features += [rel_bp_theta_tp9, rel_bp_theta_af7, rel_bp_theta_af8, rel_bp_theta_tp10]

      # Relative bandpowers alpha
      level_samples[j,28] = rel_bp_alpha_tp9 = bandpower(tp9_f, sf, [8, 12], relative=True)
      level_samples[j,29] = rel_bp_alpha_af7 = bandpower(af7_f, sf, [8, 12], relative=True)
      level_samples[j,30] = rel_bp_alpha_af8 = bandpower(af8_f, sf, [8, 12], relative=True)
      level_samples[j,31] = rel_bp_alpha_tp10 = bandpower(tp10_f, sf, [8, 12], relative=True)

      features += [rel_bp_alpha_tp9, rel_bp_alpha_af7, rel_bp_alpha_af8, rel_bp_alpha_tp10]

      # Relative bandpowers beta
      level_samples[j,32] = rel_bp_beta_tp9 = bandpower(tp9_f, sf, [12, 30], relative=True)
      level_samples[j,33] = rel_bp_beta_af7 = bandpower(af7_f, sf, [12, 30], relative=True)
      level_samples[j,34] = rel_bp_beta_af8 = bandpower(af8_f, sf, [12, 30], relative=True)
      level_samples[j,35] = rel_bp_beta_tp10 = bandpower(tp10_f, sf, [12, 30], relative=True)

      features += [rel_bp_beta_tp9, rel_bp_beta_af7, rel_bp_beta_af8, rel_bp_beta_tp10]

      # Relative bandpowers gamma
      level_samples[j,36] = rel_bp_gamma_tp9 = bandpower(tp9_f, sf, [30, 100], relative=True)
      level_samples[j,37] = rel_bp_gamma_af7 = bandpower(af7_f, sf, [30, 100], relative=True)
      level_samples[j,38] = rel_bp_gamma_af8 = bandpower(af8_f, sf, [30, 100], relative=True)
      level_samples[j,39] = rel_bp_gamma_tp10 = bandpower(tp10_f, sf, [30, 100], relative=True)

      features += [rel_bp_gamma_tp9, rel_bp_gamma_af7, rel_bp_gamma_af8, rel_bp_gamma_tp10]

      ##########################
      ## STATISTICAL FEATURES ##
      ##########################

      # Mean values
      level_samples[j,40] = mean_tp9 = np.mean(tp9_f)
      level_samples[j,41] = mean_af7 = np.mean(af7_f)
      level_samples[j,42] = mean_af8 = np.mean(af8_f)
      level_samples[j,43] = mean_tp10 = np.mean(tp10_f)

      features += [mean_tp9, mean_af7, mean_af8, mean_tp10]

      # Standard deviation
      level_samples[j,44] = std_tp9 = np.std(tp9_f)
      level_samples[j,45] = std_af7 = np.std(af7_f)
      level_samples[j,46] = std_af8 = np.std(af8_f)
      level_samples[j,47] = std_tp10 = np.std(tp10_f)

      features += [std_tp9, std_af7, std_af8, std_tp10]

      # Skeness
      level_samples[j,48] = sk_tp9 = skew(tp9_f)
      level_samples[j,49] = sk_af7 = skew(af7_f)
      level_samples[j,50] = sk_af8 = skew(af8_f)
      level_samples[j,51] = sk_tp10 = skew(tp10_f)

      features += [sk_tp9, sk_af7, sk_af8, sk_tp10]

      # Kurtosis
      level_samples[j,52] = kt_tp9 = kurtosis(tp9_f)
      level_samples[j,53] = kt_af7 = kurtosis(af7_f)
      level_samples[j,54] = kt_af8 = kurtosis(af8_f)
      level_samples[j,55] = kt_tp10 = kurtosis(tp10_f)

      features += [kt_tp9, kt_af7, kt_af8, kt_tp10]

      # Autocorrelation
      level_samples[j,56:97] = ac_tp9 = sm.tsa.acf(tp9_f, nlags=40, fft=False)
      level_samples[j,98:139] = ac_af7 = sm.tsa.acf(af7_f, nlags=40, fft=False)
      level_samples[j,140:181] = ac_af8 = sm.tsa.acf(af8_f, nlags=40, fft=False)
      level_samples[j,182:223] = ac_tp10 = sm.tsa.acf(tp10_f, nlags=40, fft=False)
      ac = ac_tp9.tolist() + ac_af7.tolist() + ac_af8.tolist() +  ac_tp10.tolist()
      features += ac

      ############################
      ## MAX, MIN & DERIVATIVES ##
      ############################

      level_samples[j,224] = uv_tp9 = (np.mean(tp9_f[:p2]) - np.mean(tp9_f[p2+1:]))/2
      level_samples[j,225] = uv_af7 = (np.mean(af7_f[:p2]) - np.mean(af7_f[p2+1:]))/2
      level_samples[j,226] = uv_af8 = (np.mean(af8_f[:p2]) - np.mean(af8_f[p2+1:]))/2
      level_samples[j,227] = uv_tp10 = (np.mean(tp10_f[:p2]) - np.mean(tp10_f[p2+1:]))/2

      features += [uv_tp9, uv_af7, uv_af8, uv_tp10]

      level_samples[j,228] = maxt_tp9 = (np.max(tp9_f[:p2]) - np.max(tp9_f[p2+1:]))/2
      level_samples[j,229] = maxt_af7 = (np.max(af7_f[:p2]) - np.max(af7_f[p2+1:]))/2
      level_samples[j,230] = maxt_af8 = (np.max(af8_f[:p2]) - np.max(af8_f[p2+1:]))/2
      level_samples[j,231] = maxt_tp10 = (np.max(tp10_f[:p2]) - np.max(tp10_f[p2+1:]))/2

      features += [maxt_tp9, maxt_af7, maxt_af8, maxt_tp10]

      level_samples[j,232] = mint_tp9 = (np.min(tp9_f[:p2]) - np.min(tp9_f[p2+1:]))/2
      level_samples[j,233] = mint_af7 = (np.min(af7_f[:p2]) - np.min(af7_f[p2+1:]))/2
      level_samples[j,234] = mint_af8 = (np.min(af8_f[:p2]) - np.min(af8_f[p2+1:]))/2
      level_samples[j,235] = mint_tp10 = (np.min(tp10_f[:p2]) - np.min(tp10_f[p2+1:]))/2

      features += [mint_tp9, mint_af7, mint_af8, mint_tp10]

      u_tp9 = []
      u_af7 = []
      u_af8 = []
      u_tp10 = []

      max_tp9 = []
      max_af7 = []
      max_af8 = []
      max_tp10 = []

      min_tp9 = []
      min_af7 = []
      min_af8 = []
      min_tp10 = []

      for t in range(4):
        u_tp9.append(np.mean(tp9_f[t*(p4+1):(t+1)*p4]))
        u_af7.append(np.mean(tp9_f[t*(p4+1):(t+1)*p4]))
        u_af8.append(np.mean(af8_f[t*(p4+1):(t+1)*p4]))
        u_tp10.append(np.mean(tp10_f[t*(p4+1):(t+1)*p4]))

        max_tp9.append(np.max(tp9_f[t*(p4+1):(t+1)*p4]))
        max_af7.append(np.max(af7_f[t*(p4+1):(t+1)*p4]))
        max_af8.append(np.max(af8_f[t*(p4+1):(t+1)*p4]))
        max_tp10.append(np.max(tp10_f[t*(p4+1):(t+1)*p4]))

        min_tp9.append(np.min(tp9_f[t*(p4+1):(t+1)*p4]))
        min_af7.append(np.min(af7_f[t*(p4+1):(t+1)*p4]))
        min_af8.append(np.min(af8_f[t*(p4+1):(t+1)*p4]))
        min_tp10.append(np.min(tp10_f[t*(p4+1):(t+1)*p4]))
      

      level_samples[j,236] = du_12_tp9 = abs(u_tp9[0]-u_tp9[1])
      level_samples[j,237] = du_13_tp9 = abs(u_tp9[0]-u_tp9[2])
      level_samples[j,238] = du_14_tp9 = abs(u_tp9[0]-u_tp9[3])
      level_samples[j,239] = du_23_tp9 = abs(u_tp9[1]-u_tp9[2])
      level_samples[j,240] = du_24_tp9 = abs(u_tp9[1]-u_tp9[3])
      level_samples[j,241] = du_34_tp9 = abs(u_tp9[2]-u_tp9[3])

      features += [du_12_tp9, du_13_tp9, du_14_tp9, du_23_tp9, du_24_tp9, du_34_tp9]

      level_samples[j,242] = du_12_af7 = abs(u_af7[0]-u_af7[1])
      level_samples[j,243] = du_13_af7 = abs(u_af7[0]-u_af7[1])
      level_samples[j,244] = du_14_af7 = abs(u_af7[0]-u_af7[1])
      level_samples[j,245] = du_23_af7 = abs(u_af7[0]-u_af7[1])
      level_samples[j,246] = du_24_af7 = abs(u_af7[0]-u_af7[1])
      level_samples[j,247] = du_34_af7 = abs(u_af7[0]-u_af7[1])

      features += [du_12_af7, du_13_af7, du_14_af7, du_23_af7, du_24_af7, du_34_af7]

      level_samples[j,248] = du_12_af8 = abs(u_af8[0]-u_af8[1])
      level_samples[j,249] = du_13_af8 = abs(u_af8[0]-u_af8[1])
      level_samples[j,250] = du_14_af8 = abs(u_af8[0]-u_af8[1])
      level_samples[j,251] = du_23_af8 = abs(u_af8[0]-u_af8[1])
      level_samples[j,252] = du_24_af8 = abs(u_af8[0]-u_af8[1])
      level_samples[j,253] = du_34_af8 = abs(u_af8[0]-u_af8[1])

      features += [du_12_af8, du_13_af8, du_14_af8, du_23_af8, du_24_af8, du_34_af8]

      level_samples[j,254] = du_12_tp10 = abs(u_tp10[0]-u_tp10[1])
      level_samples[j,255] = du_13_tp10 = abs(u_tp10[0]-u_tp10[1])
      level_samples[j,256] = du_14_tp10 = abs(u_tp10[0]-u_tp10[1])
      level_samples[j,257] = du_23_tp10 = abs(u_tp10[0]-u_tp10[1])
      level_samples[j,258] = du_24_tp10 = abs(u_tp10[0]-u_tp10[1])
      level_samples[j,259] = du_34_tp10 = abs(u_tp10[0]-u_tp10[1])

      features += [du_12_tp10, du_13_tp10, du_14_tp10, du_23_tp10, du_24_tp10, du_34_tp10]

      level_samples[j,260] = dmax_12_tp9 = abs(max_tp9[0]-max_tp9[1])
      level_samples[j,261] = dmax_13_tp9 = abs(max_tp9[0]-max_tp9[2])
      level_samples[j,262] = dmax_14_tp9 = abs(max_tp9[0]-max_tp9[3])
      level_samples[j,263] = dmax_23_tp9 = abs(max_tp9[1]-max_tp9[2])
      level_samples[j,264] = dmax_24_tp9 = abs(max_tp9[1]-max_tp9[3])
      level_samples[j,265] = dmax_34_tp9 = abs(max_tp9[2]-max_tp9[3])

      features += [dmax_12_tp9, dmax_13_tp9, dmax_14_tp9, dmax_23_tp9, dmax_24_tp9, dmax_34_tp9]

      level_samples[j,266] = dmax_12_af7 = abs(max_af7[0]-max_af7[1])
      level_samples[j,267] = dmax_13_af7 = abs(max_af7[0]-max_af7[1])
      level_samples[j,268] = dmax_14_af7 = abs(max_af7[0]-max_af7[1])
      level_samples[j,269] = dmax_23_af7 = abs(max_af7[0]-max_af7[1])
      level_samples[j,270] = dmax_24_af7 = abs(max_af7[0]-max_af7[1])
      level_samples[j,271] = dmax_34_af7 = abs(max_af7[0]-max_af7[1])

      features += [dmax_12_af7, dmax_13_af7, dmax_14_af7, dmax_23_af7, dmax_24_af7, dmax_34_af7]

      level_samples[j,272] = dmax_12_af8 = abs(max_af8[0]-max_af8[1])
      level_samples[j,273] = dmax_13_af8 = abs(max_af8[0]-max_af8[1])
      level_samples[j,274] = dmax_14_af8 = abs(max_af8[0]-max_af8[1])
      level_samples[j,275] = dmax_23_af8 = abs(max_af8[0]-max_af8[1])
      level_samples[j,276] = dmax_24_af8 = abs(max_af8[0]-max_af8[1])
      level_samples[j,277] = dmax_34_af8 = abs(max_af8[0]-max_af8[1])

      features += [dmax_12_af8, dmax_13_af8, dmax_14_af8, dmax_23_af8, dmax_24_af8, dmax_34_af8]

      level_samples[j,278] = dmax_12_tp10 = abs(max_tp10[0]-max_tp10[1])
      level_samples[j,279] = dmax_13_tp10 = abs(max_tp10[0]-max_tp10[1])
      level_samples[j,280] = dmax_14_tp10 = abs(max_tp10[0]-max_tp10[1])
      level_samples[j,281] = dmax_23_tp10 = abs(max_tp10[0]-max_tp10[1])
      level_samples[j,282] = dmax_24_tp10 = abs(max_tp10[0]-max_tp10[1])
      level_samples[j,283] = dmax_34_tp10 = abs(max_tp10[0]-max_tp10[1])

      features += [dmax_12_tp10, dmax_13_tp10, dmax_14_tp10, dmax_23_tp10, dmax_24_tp10, dmax_34_tp10]

      level_samples[j,284] = dmin_12_tp9 = abs(min_tp9[0]-min_tp9[1])
      level_samples[j,285] = dmin_13_tp9 = abs(min_tp9[0]-min_tp9[2])
      level_samples[j,286] = dmin_14_tp9 = abs(min_tp9[0]-min_tp9[3])
      level_samples[j,287] = dmin_23_tp9 = abs(min_tp9[1]-min_tp9[2])
      level_samples[j,288] = dmin_24_tp9 = abs(min_tp9[1]-min_tp9[3])
      level_samples[j,289] = dmin_34_tp9 = abs(min_tp9[2]-min_tp9[3])

      features += [dmin_12_tp9, dmin_13_tp9, dmin_14_tp9, dmin_23_tp9, dmin_24_tp9, dmin_34_tp9]

      level_samples[j,290] = dmin_12_af7 = abs(min_af7[0]-min_af7[1])
      level_samples[j,291] = dmin_13_af7 = abs(min_af7[0]-min_af7[1])
      level_samples[j,292] = dmin_14_af7 = abs(min_af7[0]-min_af7[1])
      level_samples[j,293] = dmin_23_af7 = abs(min_af7[0]-min_af7[1])
      level_samples[j,294] = dmin_24_af7 = abs(min_af7[0]-min_af7[1])
      level_samples[j,295] = dmin_34_af7 = abs(min_af7[0]-min_af7[1])

      features += [dmin_12_af7, dmin_13_af7, dmin_14_af7, dmin_23_af7, dmin_24_af7, dmin_34_af7]

      level_samples[j,296] = dmin_12_af8 = abs(min_af8[0]-min_af8[1])
      level_samples[j,297] = dmin_13_af8 = abs(min_af8[0]-min_af8[1])
      level_samples[j,298] = dmin_14_af8 = abs(min_af8[0]-min_af8[1])
      level_samples[j,299] = dmin_23_af8 = abs(min_af8[0]-min_af8[1])
      level_samples[j,300] = dmin_24_af8 = abs(min_af8[0]-min_af8[1])
      level_samples[j,301] = dmin_34_af8 = abs(min_af8[0]-min_af8[1])

      features += [dmin_12_af8, dmin_13_af8, dmin_14_af8, dmin_23_af8, dmin_24_af8, dmin_34_af8]

      level_samples[j,302] = dmin_12_tp10 = abs(min_tp10[0]-min_tp10[1])
      level_samples[j,303] = dmin_13_tp10 = abs(min_tp10[0]-min_tp10[1])
      level_samples[j,304] = dmin_14_tp10 = abs(min_tp10[0]-min_tp10[1])
      level_samples[j,305] = dmin_23_tp10 = abs(min_tp10[0]-min_tp10[1])
      level_samples[j,306] = dmin_24_tp10 = abs(min_tp10[0]-min_tp10[1])
      level_samples[j,307] = dmin_34_tp10 = abs(min_tp10[0]-min_tp10[1])

      features += [dmin_12_tp10, dmin_13_tp10, dmin_14_tp10, dmin_23_tp10, dmin_24_tp10, dmin_34_tp10]


      #############
      ## ENTROPY ##
      #############

      # Sample entropy
      level_samples[j,308] = sampe_tp9 = ant.sample_entropy(tp9_f, order=2)
      level_samples[j,309] = sampe_af7 = ant.sample_entropy(af7_f, order=2)
      level_samples[j,310] = sampe_af8 = ant.sample_entropy(af8_f, order=2)
      level_samples[j,311] = sampe_tp10 = ant.sample_entropy(tp10_f, order=2)

      features += [sampe_tp9, sampe_af7, sampe_af8, sampe_tp10]

      # spectral entropy
      level_samples[j,312] = spece_tp9 = ant.spectral_entropy(tp9_f, sf=sf, normalize=True)
      level_samples[j,313] = spece_af7 = ant.spectral_entropy(af7_f, sf=sf, normalize=True)
      level_samples[j,314] = spece_af8 = ant.spectral_entropy(af8_f, sf=sf, normalize=True)
      level_samples[j,315] = spece_tp10 = ant.spectral_entropy(tp10_f, sf=sf, normalize=True)

      features += [spece_tp9, spece_af7, spece_af8, spece_tp10]

      ###########################
      ## CONNECTIVITY FEATURES ##
      ###########################

      # Coherence
      Cxy_tp9_af8 = signal.coherence(tp9_f, af8_f, fs=sf, window='tukey')

      idx_band_delta = np.logical_and(Cxy_tp9_af8[0] >= 0.5, Cxy_tp9_af8[0] <= 4)
      idx_band_theta = np.logical_and(Cxy_tp9_af8[0] >= 4, Cxy_tp9_af8[0] <= 8)
      idx_band_alpha = np.logical_and(Cxy_tp9_af8[0] >= 8, Cxy_tp9_af8[0] <= 12)
      idx_band_beta = np.logical_and(Cxy_tp9_af8[0] >= 12, Cxy_tp9_af8[0] <= 30)
      idx_band_gamma = np.logical_and(Cxy_tp9_af8[0] >= 30, Cxy_tp9_af8[0] <= 100)

      level_samples[j,316] = Cxy_tp9_af8_delta = np.mean(Cxy_tp9_af8[1][idx_band_delta])
      level_samples[j,317] = Cxy_tp9_af8_theta = np.mean(Cxy_tp9_af8[1][idx_band_theta])
      level_samples[j,318] = Cxy_tp9_af8_alpha = np.mean(Cxy_tp9_af8[1][idx_band_alpha])
      level_samples[j,319] = Cxy_tp9_af8_beta = np.mean(Cxy_tp9_af8[1][idx_band_beta])
      level_samples[j,320] = Cxy_tp9_af8_gamma = np.mean(Cxy_tp9_af8[1][idx_band_gamma])
     
      Cxy_tp9_af7 = signal.coherence(tp9_f, af7_f, fs=sf, window='tukey')

      idx_band_delta = np.logical_and(Cxy_tp9_af7[0] >= 0.5, Cxy_tp9_af7[0] <= 4)
      idx_band_theta = np.logical_and(Cxy_tp9_af7[0] >= 4, Cxy_tp9_af7[0] <= 8)
      idx_band_alpha = np.logical_and(Cxy_tp9_af7[0] >= 8, Cxy_tp9_af7[0] <= 12)
      idx_band_beta = np.logical_and(Cxy_tp9_af7[0] >= 12, Cxy_tp9_af7[0] <= 30)
      idx_band_gamma = np.logical_and(Cxy_tp9_af7[0] >= 30, Cxy_tp9_af7[0] <= 100)

      level_samples[j,321] = Cxy_tp9_af7_delta = np.mean(Cxy_tp9_af7[1][idx_band_delta])
      level_samples[j,322] = Cxy_tp9_af7_theta = np.mean(Cxy_tp9_af7[1][idx_band_theta])
      level_samples[j,323] = Cxy_tp9_af7_alpha = np.mean(Cxy_tp9_af7[1][idx_band_alpha])
      level_samples[j,324] = Cxy_tp9_af7_beta = np.mean(Cxy_tp9_af7[1][idx_band_beta])
      level_samples[j,325] = Cxy_tp9_af7_gamma = np.mean(Cxy_tp9_af7[1][idx_band_gamma])
      
      Cxy_tp9_tp10 = signal.coherence(tp9_f, tp10_f, fs=sf, window='tukey')

      idx_band_delta = np.logical_and(Cxy_tp9_tp10[0] >= 0.5, Cxy_tp9_tp10[0] <= 4)
      idx_band_theta = np.logical_and(Cxy_tp9_tp10[0] >= 4, Cxy_tp9_tp10[0] <= 8)
      idx_band_alpha = np.logical_and(Cxy_tp9_tp10[0] >= 8, Cxy_tp9_tp10[0] <= 12)
      idx_band_beta = np.logical_and(Cxy_tp9_tp10[0] >= 12, Cxy_tp9_tp10[0] <= 30)
      idx_band_gamma = np.logical_and(Cxy_tp9_tp10[0] >= 30, Cxy_tp9_tp10[0] <= 100)
      
      level_samples[j,326] = Cxy_tp9_tp10_delta = np.mean(Cxy_tp9_tp10[1][idx_band_delta])
      level_samples[j,327] = Cxy_tp9_tp10_theta = np.mean(Cxy_tp9_tp10[1][idx_band_theta])
      level_samples[j,328] = Cxy_tp9_tp10_alpha = np.mean(Cxy_tp9_tp10[1][idx_band_alpha])
      level_samples[j,329] = Cxy_tp9_tp10_beta = np.mean(Cxy_tp9_tp10[1][idx_band_beta])
      level_samples[j,330] = Cxy_tp9_tp10_gamma = np.mean(Cxy_tp9_tp10[1][idx_band_gamma])

      Cxy_af8_af7 = signal.coherence(af8_f, af7_f, fs=sf, window='tukey')

      idx_band_delta = np.logical_and(Cxy_af8_af7[0] >= 0.5, Cxy_af8_af7[0] <= 4)
      idx_band_theta = np.logical_and(Cxy_af8_af7[0] >= 4, Cxy_af8_af7[0] <= 8)
      idx_band_alpha = np.logical_and(Cxy_af8_af7[0] >= 8, Cxy_af8_af7[0] <= 12)
      idx_band_beta = np.logical_and(Cxy_af8_af7[0] >= 12, Cxy_af8_af7[0] <= 30)
      idx_band_gamma = np.logical_and(Cxy_af8_af7[0] >= 30, Cxy_af8_af7[0] <= 100)
      
      level_samples[j,331] = Cxy_af8_af7_delta = np.mean(Cxy_af8_af7[1][idx_band_delta])
      level_samples[j,332] = Cxy_af8_af7_theta = np.mean(Cxy_af8_af7[1][idx_band_theta])
      level_samples[j,333] = Cxy_af8_af7_alpha = np.mean(Cxy_af8_af7[1][idx_band_alpha])
      level_samples[j,334] = Cxy_af8_af7_beta = np.mean(Cxy_af8_af7[1][idx_band_beta])
      level_samples[j,335] = Cxy_af8_af7_gamma = np.mean(Cxy_af8_af7[1][idx_band_gamma])

      Cxy_af8_tp10 = signal.coherence(af8_f, tp10_f, fs=sf, window='tukey')

      idx_band_delta = np.logical_and(Cxy_af8_tp10[0] >= 0.5, Cxy_af8_tp10[0] <= 4)
      idx_band_theta = np.logical_and(Cxy_af8_tp10[0] >= 4, Cxy_af8_tp10[0] <= 8)
      idx_band_alpha = np.logical_and(Cxy_af8_tp10[0] >= 8, Cxy_af8_tp10[0] <= 12)
      idx_band_beta = np.logical_and(Cxy_af8_tp10[0] >= 12, Cxy_af8_tp10[0] <= 30)
      idx_band_gamma = np.logical_and(Cxy_af8_tp10[0] >= 30, Cxy_af8_tp10[0] <= 100)
      
      level_samples[j,336] = Cxy_af8_tp10_delta = np.mean(Cxy_af8_tp10[1][idx_band_delta])
      level_samples[j,337] = Cxy_af8_tp10_theta = np.mean(Cxy_af8_tp10[1][idx_band_theta])
      level_samples[j,338] = Cxy_af8_tp10_alpha = np.mean(Cxy_af8_tp10[1][idx_band_alpha])
      level_samples[j,339] = Cxy_af8_tp10_beta = np.mean(Cxy_af8_tp10[1][idx_band_beta])
      level_samples[j,340] = Cxy_af8_tp10_gamma = np.mean(Cxy_af8_tp10[1][idx_band_gamma])

      Cxy_af7_tp10 = signal.coherence(af7_f, tp10_f, fs=sf, window='tukey')

      idx_band_delta = np.logical_and(Cxy_af7_tp10[0] >= 0.5, Cxy_af7_tp10[0] <= 4)
      idx_band_theta = np.logical_and(Cxy_af7_tp10[0] >= 4, Cxy_af7_tp10[0] <= 8)
      idx_band_alpha = np.logical_and(Cxy_af7_tp10[0] >= 8, Cxy_af7_tp10[0] <= 12)
      idx_band_beta = np.logical_and(Cxy_af7_tp10[0] >= 12, Cxy_af7_tp10[0] <= 30)
      idx_band_gamma = np.logical_and(Cxy_af7_tp10[0] >= 30, Cxy_af7_tp10[0] <= 100)
      
      level_samples[j,341] = Cxy_af7_tp10_delta = np.mean(Cxy_af7_tp10[1][idx_band_delta])
      level_samples[j,342] = Cxy_af7_tp10_theta = np.mean(Cxy_af7_tp10[1][idx_band_theta])
      level_samples[j,343] = Cxy_af7_tp10_alpha = np.mean(Cxy_af7_tp10[1][idx_band_alpha])
      level_samples[j,344] = Cxy_af7_tp10_beta = np.mean(Cxy_af7_tp10[1][idx_band_beta])
      level_samples[j,345] = Cxy_af7_tp10_gamma = np.mean(Cxy_af7_tp10[1][idx_band_gamma])

      features += [Cxy_tp9_af8_delta, Cxy_tp9_af8_theta, Cxy_tp9_af8_alpha, Cxy_tp9_af8_beta, Cxy_tp9_af8_gamma,
                   Cxy_tp9_af7_delta, Cxy_tp9_af7_theta, Cxy_tp9_af7_alpha, Cxy_tp9_af7_beta, Cxy_tp9_af7_gamma,
                   Cxy_tp9_tp10_delta, Cxy_tp9_tp10_theta, Cxy_tp9_tp10_alpha, Cxy_tp9_tp10_beta,Cxy_tp9_tp10_gamma,
                   Cxy_af8_af7_delta, Cxy_af8_af7_theta, Cxy_af8_af7_alpha, Cxy_af8_af7_beta, Cxy_af8_af7_gamma,
                   Cxy_af8_tp10_delta, Cxy_af8_tp10_theta, Cxy_af8_tp10_alpha, Cxy_af8_tp10_beta, Cxy_af8_tp10_gamma,
                   Cxy_af7_tp10_delta, Cxy_af7_tp10_theta, Cxy_af7_tp10_alpha, Cxy_af7_tp10_beta, Cxy_af7_tp10_gamma]

      # Phase lag

      # Amplitude assymetry

      #######################
      ## FRACTAL DIMENSION ##
      #######################

      # Detrended fluctuation
      level_samples[j,346] = fluct_tp9 = ant.detrended_fluctuation(tp9_f)
      level_samples[j,347] = fluct_af7 = ant.detrended_fluctuation(af7_f)
      level_samples[j,348] = fluct_af8 = ant.detrended_fluctuation(af8_f)
      level_samples[j,349] = fluct_tp10 = ant.detrended_fluctuation(tp10_f)

      features += [fluct_tp9, fluct_af7, fluct_af8, fluct_tp10]

      curr_feature = len(features)

      # assert curr_feature == n_features, f"Final number of features({curr_feature}) is not equal to {n_features}"
      # Add target to feature
      features.append(i)
      samples.append(features)
    list_array_samples.append(level_samples)

  # Randomly shuffle samples, convert to numpy, create targets, and save csv file
  samples = np.array(samples, dtype=object)
  np.random.shuffle(samples)
  targets_list = samples[:,-1]
  samples_list = np.delete(samples, -1, axis=1) 
  list_data_dict = {'feature_names': feature_names, 'target': targets_list, 'data': samples_list}

  np_array_samples = np.vstack(tuple(list_array_samples))
  np.random.shuffle(np_array_samples)
  array_data_dict = {'feature_names': feature_names, 'target': np_array_samples[:,-1], 'data': np_array_samples[:,:-1]}
  
  
  with open("EEG_data_list.pkl", "wb") as tf:
    dump(list_data_dict,tf)
  with open("EEG_data_array.pkl", "wb") as tf:
    dump(array_data_dict,tf)

if __name__ == "__main__":
  main()

