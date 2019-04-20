import numpy as np


#classify takes the output of step2 and returns a classification of each test case
#outputs is a size 10xN array from W*X
#bias is a vector of size 10 from gen_bias
#returns a size N array representing which digit each element is
def classify(outputs,bias):
    return np.argmax(outputs+bias)
