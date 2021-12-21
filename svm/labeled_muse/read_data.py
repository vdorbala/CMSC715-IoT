import numpy as np
import math

SAMPLE_SIZE = 1667

def process_data(data_txt, label):

    lines = data_txt.readlines()

    data = np.zeros((len(lines)-1, 4))

    for i, x in enumerate(lines[1:]):
        x = x.replace('\n', '')
        result = x.split("\t")
        result = list(map(float, result))
        data[i,:] = np.array(result[1:])
    
    n_samples = math.floor(data.shape[0]/SAMPLE_SIZE)

    for i in range(n_samples):
        np.save(label + f'/sample_{i}', data[i*1667:(i+1)*1667,:])
    

def main():
    f_high = open('high_0', "r")
    process_data(f_high, 'high')
    
    f_low = open('low_0', "r")
    process_data(f_low, 'low')

    f_normal = open('normal_0', "r")
    process_data(f_normal, 'normal')

if __name__ == "__main__":
    main()
