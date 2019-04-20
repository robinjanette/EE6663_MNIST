import mnist
import numpy as np

def format():
    data, labels, _, _ = mnist.load()

    labels = list(labels)

    # label data and scale from (0,255) to (0,1)
    labeled = [[], [], [], [], [],
               [], [], [], [], []]
    for i in range(len(data)):
        labeled[labels[i]].append(data[i] / 255)
    
    return labeled

def mmv(formatted):
    mmv_data = [{'mean': np.array([]), 'median': np.array([]), 'var': np.array([])},
                {'mean': np.array([]), 'median': np.array([]), 'var': np.array([])},
                {'mean': np.array([]), 'median': np.array([]), 'var': np.array([])},
                {'mean': np.array([]), 'median': np.array([]), 'var': np.array([])},
                {'mean': np.array([]), 'median': np.array([]), 'var': np.array([])},
                {'mean': np.array([]), 'median': np.array([]), 'var': np.array([])},
                {'mean': np.array([]), 'median': np.array([]), 'var': np.array([])},
                {'mean': np.array([]), 'median': np.array([]), 'var': np.array([])},
                {'mean': np.array([]), 'median': np.array([]), 'var': np.array([])},
                {'mean': np.array([]), 'median': np.array([]), 'var': np.array([])}]
                
    for digit in range(10):
        mmv_data[digit]['mean'] = np.mean(formatted[digit], axis=0)
        mmv_data[digit]['median'] = np.median(formatted[digit], axis=0)
        mmv_data[digit]['var'] = np.var(formatted[digit], axis=0)

    return mmv_data
