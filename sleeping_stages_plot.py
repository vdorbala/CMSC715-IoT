import yasa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from tqdm import tqdm
import os

hypno_30s = np.loadtxt('hypno.txt')
# Read glucose data file
glucose_data = pd.read_csv("data/dexcom_glucose/glucose_data.csv")
glucose_data = glucose_data.drop(glucose_data.columns[[0, 2, 3, 4 , 5, 6, 8, 9 ,10, 11, 12, 13]], axis=1)
glucose_data = glucose_data [glucose_data['Glucose Value (mg/dL)'].isin(['High', 'Low']) == False]
glucose_data = glucose_data.dropna()
glucose_data = glucose_data.reset_index()

# glucose_data = float(glucose_data['Glucose Value (mg/dL)'][:])

glucose_data['Glucose Value (mg/dL)'] = glucose_data['Glucose Value (mg/dL)'].astype('float')

glucose_data['Timestamp (YYYY-MM-DDThh:mm:ss)'] =  glucose_data['Timestamp (YYYY-MM-DDThh:mm:ss)'].str.replace(r"T", "  ", regex=True)

glucose_data['Timestamp (YYYY-MM-DDThh:mm:ss)'] = pd.to_datetime(glucose_data['Timestamp (YYYY-MM-DDThh:mm:ss)'], infer_datetime_format=True)


muse_file_list = os.listdir("data/muse")
for file in tqdm(muse_file_list):

    filename = str("data/muse/" + file)


    print("\nFile reading currently is {}".format(filename))
    df = pd.read_csv(filename)
    eliminate_col = [val for val in range(25,39)]
    df = df.drop(df.columns[eliminate_col], axis=1)
    df = df.dropna()
    filtered_glucose = glucose_data.loc[(glucose_data['Timestamp (YYYY-MM-DDThh:mm:ss)'] >= df.iloc[0]['TimeStamp']) & (glucose_data['Timestamp (YYYY-MM-DDThh:mm:ss)'] <= df.iloc[-1]['TimeStamp'])]
    filtered_glucose = filtered_glucose.drop('index',axis=1)
    index = [i for i in np.arange(0,filtered_glucose.shape[0]*10,10)]
    filtered_glucose['x'] = index
    df = df.loc[:,'RAW_TP9':'RAW_TP10']
    df = df.to_numpy()
    data = np.transpose(df)
    
    plt.figure()
    plt.plot(filtered_glucose['x'],filtered_glucose['Glucose Value (mg/dL)'])
    plt.title("Glucose Levels")
    plt.xlabel("Time (30-sec epoch")
    plt.ylabel("Glucose Value (mg/dL)")
    plt.savefig('glucose_'+file+'.png')

    channel_names = ['tp9', 'af7', 'af8', 'tp10']

    sf = 256
    info = mne.create_info(channel_names, sfreq=sf)
    raw = mne.io.RawArray(data, info)
    for chan in tqdm(channel_names):
        sls = yasa.SleepStaging(raw, eeg_name=chan)
        y_pred = sls.predict()
        hypno = yasa.hypno_upsample_to_data(hypno=y_pred, sf_hypno=(1/30), data=data, sf_data=sf)
        hypno_int = yasa.hypno_str_to_int(hypno)
        if chan == 'tp9':
            dt = data[0,:]
        elif chan == 'af7':
            dt = data[1,:]
        elif chan == 'af8':
            dt = data[2,:]
        else:
            dt = data[3,:]
        fig = yasa.plot_spectrogram(dt, sf, hypno=hypno_int, fmax=30, cmap='Spectral_r', trimperc=5)
        plt.title("Spectrogram and Hypnogram for Channel "+chan)
        plt.savefig('spec_'+chan+file+'.png')
        y_prob = sls.predict_proba()
        sls.plot_predict_proba()
        plt.title("Sleeping Stages for Channel "+chan)
        plt.savefig('rem_'+chan+file+'.png')
        confidence = sls.predict_proba().max(1)

