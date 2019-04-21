import numpy as np
import mnist
import numpy as np
import os
import matplotlib
import cv2
import mmv
import gen_weights
import classify
import applyWeights
import bias

trainX,trainY,testX,testY = mnist.load()
trainX = trainX.astype(float)
#trainY = trainY[:100]
testX = testX.astype(float)
trainX /= 255.0
testX /= 255.0

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_d(x):
    return x*(1-x)

def softmax(x):
    return (np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True))

def softmax_deriv(softmax):
    s = softmax.reshape(-1,1)
    return (np.diagflat(s) - np.dot(s, s.T)).T

class NNOneLayer:
    def __init__(self,**args):
        self.weights = args.get("weights",np.random.rand(*args["shape"]))
        self.bias = args.get("bias",np.random.rand(args["shape"][0]))

    def outputs(self,X):
        return softmax((np.dot(X,self.weights.T)+self.bias))

    def classify(self,X):
        return np.argmax(self.outputs(X),axis=1)

    def backprop(self,X,Y,alpha=0.1):
        outputs = self.outputs(X)

        delta = outputs - Y

        self.weights -= (alpha* X.T.dot(delta)).T
        self.bias -= alpha * delta.sum(axis=0)


        """
        e_bias = Y-outputs
        e_weights = Y-outputs+self.bias
        d_bias = -e_bias
        d_weights = -e_weights
        #d_bias = e_bias * softmax_deriv(outputs)
        #d_weights = e_weights * softmax_deriv(outputs)
        self.bias += speed*np.sum(d_bias,axis=0)
        #print(np.min((speed * np.dot(X.T,d_weights)).T))
        self.weights += speed * ( np.dot(X.T,d_weights)).T
        """
        
trainY2 = np.zeros((trainX.shape[0],10))
for i in range(trainY2.shape[0]):
    trainY2[i][trainY[i]] = 1
testY2 = np.zeros((testX.shape[0],10))
for i in range(testY2.shape[0]):
    testY2[i][testY[i]] = 1

train, test = mmv.format()
    
#get the mean, median, and variance patterns for each number 0-9
meanmedianvar = mmv.mmv(train)
    
#get the final weight maps for each number 0-9
weightmatrix = gen_weights.gen_weights(meanmedianvar)
    
#get the bias
outputs = [applyWeights.applyWeights(weightmatrix,np.array(train[i]).T) for i in range(10) ]
b = bias.gen_bias(weightmatrix)

network = NNOneLayer(weights=weightmatrix,bias=b,shape=weightmatrix.shape)
network = NNOneLayer(shape=weightmatrix.shape)

for i in range(1001):
    outputs = network.outputs(trainX)
    c = network.classify(trainX)
    network.backprop(trainX,trainY2,0.00001)
    if i % 100 == 0:
        print(i)
        print(outputs.shape,c.shape)
        print(len(c[c==trainY])/len(c))
        print(np.mean(np.abs(outputs-trainY2)))
        print("")

outputs = network.outputs(testX)
c = network.classify(testX)
print(len(c[c==testY])/len(c))
print(np.mean(np.abs(outputs-testY2)))
    
