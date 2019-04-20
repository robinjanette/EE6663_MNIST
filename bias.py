import numpy as np


#generates the bias from the outputs of a training set
#takes in a 10xN array, from the outputs of W*X
#return a vector of size 10
def gen_bias(outputs):
    return np.zeros((10,))
