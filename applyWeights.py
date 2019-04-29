import numpy as np

# Takes the weight matrix and a test smaple then computes the dot product
# of the weights and test smaple
# If the weights and test sizes don't match return -1
def applyWeights(weights, test):

    # Check if the inputs are correct
    if len(weights[0]) != len(test):
        return -1

    test /= np.linalg.norm(test)
        

    return np.dot(weights,test)



##l = np.array([[0,0,0,0,0],[1,20,3,4,5]])
##m = np.array([1,0,0,0,0])
##
##res = applyWeights(l,m.T)
##
##print (res)
