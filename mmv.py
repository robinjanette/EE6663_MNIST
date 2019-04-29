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
    mmv_data = [{'mean': np.array([]), 'median': np.array([]), 'var': np.array([]), 'std':np.array([])},
                {'mean': np.array([]), 'median': np.array([]), 'var': np.array([]), 'std':np.array([])},
                {'mean': np.array([]), 'median': np.array([]), 'var': np.array([]), 'std':np.array([])},
                {'mean': np.array([]), 'median': np.array([]), 'var': np.array([]), 'std':np.array([])},
                {'mean': np.array([]), 'median': np.array([]), 'var': np.array([]), 'std':np.array([])},
                {'mean': np.array([]), 'median': np.array([]), 'var': np.array([]), 'std':np.array([])},
                {'mean': np.array([]), 'median': np.array([]), 'var': np.array([]), 'std':np.array([])},
                {'mean': np.array([]), 'median': np.array([]), 'var': np.array([]), 'std':np.array([])},
                {'mean': np.array([]), 'median': np.array([]), 'var': np.array([]), 'std':np.array([])},
                {'mean': np.array([]), 'median': np.array([]), 'var': np.array([]), 'std':np.array([])}]
                
    for digit in range(10):
        mmv_data[digit]['mean'] = np.mean(formatted[digit], axis=0)
        mmv_data[digit]['median'] = np.median(formatted[digit], axis=0)
        mmv_data[digit]['var'] = np.var(formatted[digit], axis=0)
        mmv_data[digit]['std'] = np.std(formatted[digit], axis=0)

    return mmv_data

def total_mmv():
    traindata, _, _, _ = mnist.load()

    scale = []
    for t in traindata:
        scale.append(t / 255)

    mmv_total = {'mean': np.array([]), 'median': np.array([]), 'var': np.array([]), 'std':np.array([])}

    mmv_total['mean'] = np.mean(scale, axis=0)
    mmv_total['median'] = np.median(scale, axis=0)
    mmv_total['var'] = np.var(scale, axis=0)
    mmv_total['std'] = np.std(scale, axis=0)

    return mmv_total