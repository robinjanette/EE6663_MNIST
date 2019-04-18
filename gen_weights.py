import numpy as np


#generates a weight matrix for digits using the mean, median, and variance
#mean,median, and mode are matrices of size 784x10
#return is a matrix of size
def gen_weights(means,medians,variances):
    return means*2-1
