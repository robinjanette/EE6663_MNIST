import mnist
import numpy as np

def format():
    traindata, trainlabels, testdata, testlabels = mnist.load()

    trainlabels = list(trainlabels)

    # label data and scale from (0,255) to (0,1)
    trainlabeled = [[], [], [], [], [],
               [], [], [], [], []]
    for i in range(len(traindata)):
        trainlabeled[trainlabels[i]].append(traindata[i] / 255)

    testlabels = list(testlabels)

    # label data and scale from (0,255) to (0,1)
    testlabeled = [[], [], [], [], [],
               [], [], [], [], []]
    for i in range(len(testdata)):
        testlabeled[testlabels[i]].append(testdata[i] / 255)
    
    return trainlabeled, testlabeled

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
