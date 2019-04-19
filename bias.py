import numpy as np


#generates the bias from the outputs of a training set
#takes in a Nx10 array
#return a 2-tuple of vectors of length 10
def gen_bias(outputs):
    #m*min+b = 0
    #m*max+b = 1
    #m*(max-min) = 1
    #m = 1/(max - min)
    #b = - min / (max-min)

    min_ar = np.min(outputs,axis=0)
    max_ar = np.max(outputs,axis=0)
    return (1/(max_ar-min_arr), - min_ar / (max_ar-min_ar))
