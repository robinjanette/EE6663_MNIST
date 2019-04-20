import numpy as np


#generates the bias from the outputs of a training set
#formatted is the result of mmv.format
#outputs is a list of the results of applying W*X to each digit in formatted
#outputs is 10xN
#return a vector of size 10
#add bias to W*X to get the output to classify
def gen_bias(outputs):
    ar = np.zeros((10,10)) #ar[i][j] means for digit i, bias needs to be ar[i][j] for output[j] to be correct
    for i in range(10):
        for j in range(10):
            if i == j:
                ar[i][j] = 1-np.min(outputs[i])
            else:
                ar[i][j] = -1-np.max(outputs[i])
    return np.mean(ar,axis=0)
