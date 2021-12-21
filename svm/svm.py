from itertools import count
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

from pickle import load
import numpy as np


with open("EEG_data_array.pkl", "rb") as tf:
    EEG_data = load(tf)

y = EEG_data['target']

sumation = 0

for i in y:
    if i == 1:
        sumation +=1
        

print('sumation: ', sumation)
print(len(y))
# print(data_dict['data'].shape)
cancer_data = load_breast_cancer()
raw_data = pd.DataFrame(cancer_data['data'], columns = cancer_data['feature_names'])
raw_data_EEG = pd.DataFrame(EEG_data['data'], columns=EEG_data['feature_names'])

# print('CANCER')
# print(cancer_data['data'])
# print('EEG')
# print(EEG_data['data'][0])
# print(raw_data)
# print(type(raw_data_EEG))

# print(type(cancer_data['target'][0]))


raw_data_EEG.fillna(0.0, inplace=True)

x_training_data,x_test_data, y_training_data, y_test_data = train_test_split(raw_data_EEG, y, test_size = 0.2)

model = SVC(cache_size=1000)

model.fit(x_training_data, y_training_data)

#Make predictions with the model

predictions = model.predict(x_test_data)

#Measure the performance of our model

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

print(classification_report(y_test_data, predictions))

print(confusion_matrix(y_test_data, predictions))