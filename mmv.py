import mnist
import numpy as np
import cv2

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

def display_total():
    mmvt = total_mmv()
    
    mmvt['mean'] = list(mmvt['mean'])
    mmvt['median'] = list(mmvt['median'])
    mmvt['var'] = list(mmvt['var'])

    for i in range(len(mmvt['mean'])):
        if mmvt['mean'][i] == 0 or mmvt['mean'][i] == 1:
            mmvt['mean'][i] = np.array([0, 0, 255]) # set red
        else:
            mmvt['mean'][i] = np.array([mmvt['mean'][i] * 255,  mmvt['mean'][i] * 255,  mmvt['mean'][i] * 255])
        
        if mmvt['median'][i] == 0 or mmvt['median'][i] == 1:
            mmvt['median'][i] = np.array([0, 0, 255]) # set red
        else:
            mmvt['median'][i] = np.array([mmvt['median'][i] * 255,  mmvt['median'][i] * 255,  mmvt['median'][i] * 255])
    
    mean_img = np.array([mmvt['mean'][i:i+28] for i in range(0, len(mmvt['mean']), 28)])
    median_img = np.array([mmvt['median'][i:i+28] for i in range(0, len(mmvt['median']), 28)])
    
    # mean_img = np.reshape(np.array(mmvt['median']), (28, 28))
    width = int(280)
    height = int(280)
    dim = (width, height)
    # resize image
    resized_mean = cv2.resize(mean_img, dim, interpolation = cv2.INTER_AREA)
    # resize image
    resized_median = cv2.resize(median_img, dim, interpolation = cv2.INTER_AREA)

    # cv2.imshow('Total Mean', resized_mean)
    cv2.imwrite('images/Total-MEAN.jpg', resized)

    # cv2.imshow('Total Median', resized_median)
    cv2.imwrite('images/Total-MEDIAN.jpg', resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()