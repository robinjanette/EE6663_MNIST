import mnist
import numpy as np

def format():
    data, labels, _, _ = mnist.load()

    # label data and scale from (0,255) to (0,1)
    labeled = [[]] * 10 # 10 digits
    for i in range(30000):
        labeled[labels[i]].append(data[i] / 255)
    
    # stack pixels together with column_stack over the images per digit 
    for digit in range(len(labeled)):
        labeled[digit] = np.column_stack(labeled[digit])
    
    return labeled

def mmv(formatted):
    mmv_data = [{'mean': np.array([]), 'median': np.array([]), 'var': np.array([])}]*10
    for digit in range(10): # foreach digit
        for i in range(784): # foreach pixel
            mmv_data[digit]['mean'] = np.append(mmv_data[digit]['mean'], np.mean(formatted[digit][i]))
            mmv_data[digit]['median'] = np.append(mmv_data[digit]['median'], np.median(formatted[digit][i]))
            mmv_data[digit]['var'] = np.append(mmv_data[digit]['var'], np.var(formatted[digit][i]))

    print(list(mmv_data[0]['mean']))

