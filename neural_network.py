import numpy as np
import mnist
import numpy as np
import os
import matplotlib
import cv2
import scipy.ndimage
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

def softmax(x):
    return (np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True))


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


use_model = True

network = NNOneLayer(weights=weightmatrix,bias=b,shape=weightmatrix.shape)
if not use_model:
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
        for i in range(10):
            img = np.array(network.weights[i])
            img -= img.min()
            img /= img.max()
            img *= 255
            img = img.reshape(28,28)
            scale = 20
            img_pixel = scipy.ndimage.zoom(img,[scale,scale],order=0)
            img_blur = scipy.ndimage.zoom(img,[scale,scale])
            s = "model_" if use_model else "rand_"
            cv2.imwrite("images/" + s +str(i)+"_w.jpg",img_pixel.astype(int))
            cv2.imwrite("images/" + s +str(i)+"_w_blur.jpg",img_blur.astype(int))
outputs = network.outputs(testX)
c = network.classify(testX)
print(len(c[c==testY])/len(c))
print(np.mean(np.abs(outputs-testY2)))
    

