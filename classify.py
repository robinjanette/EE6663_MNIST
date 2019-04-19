import numpy as np


#classify takes the output of step2 and returns a classification of each test case
#outputs is a Nx10 matrix
#variances is a 784x10 matrix from step1
def classify(outputs,bias):
    m,b = bias
    return np.argmax(m*outputs+b,axis=1)
