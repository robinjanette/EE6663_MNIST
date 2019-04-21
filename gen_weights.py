import numpy as np


#generates a weight matrix for digits using the mean, median, and variance
#mmv is the output from mmv.py 
#return is a matrix of size 10x784
def gen_weights(mmv):
    ar = np.zeros((10,784))
    for i in range(10):
        ar[i] = mmv[i]["mean"]
        ar[i] /= np.linalg.norm(ar[i])
    return ar

    
    
