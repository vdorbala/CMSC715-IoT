# CMSC715-IoT
A git repository for code and data for our IoT Project - Continuous Glucose Monitoring using EEG Signals from a MUSE.

# Dataset
The dataset can be downloaded from the drive link below.
Link for labeled data - https://drive.google.com/drive/folders/1tBDT9eOeNKahmpF6nMx5BMnyx-kMKano?usp=sharing

Link for Dexcom and Muse - https://drive.google.com/drive/folders/1Dct0YEdKABjVKfm4Bfl3_yfoTrD0YSQ9?usp=sharing

# Instructions for Usage
1. 'eeg-final.ipynb' contains the code for training and validating the EEGLab model. The hyperparamters can be changed in the file.
2. 'real_time_monitoring.py' contains code for reading data from the MUSE monitor, and predicting glucose values in real time.
3. 'sleeping_stages.py' contains code for classifying the sleeping stages from the MUSE data.
4. We also attempted to use an SVM to classify the data, and the SVM folder contains code for this.
