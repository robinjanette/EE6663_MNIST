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
import matplotlib
import matplotlib.pyplot as plt


#Normalizing the dataset
trainX,trainY,testX,testY = mnist.load()
trainX = trainX.astype(float)
testX = testX.astype(float)
trainX /= 255.0
testX /= 255.0


#the softmax activation function
def softmax(x):
    return (np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True))

class NNOneLayer:
    #takes in as keyword args
    #   "shape" which is the shape of the weight array which
    #       should be a tuple (output_size,input_size)
    #   "weights": the initial weight array of size "shape", if not supplied it will generate
    #       a random weight array from shape. 
    #   "bias": the bias vector should be size shape[0], if not supplied it will generate
    #       a random vector
    def __init__(self,**args):
        self.weights = args.get("weights",np.random.rand(*args["shape"]))
        self.bias = args.get("bias",np.random.rand(args["shape"][0]))


    #Returns the output of the neural network using softmax activation
    def outputs(self,X):
        return softmax((np.dot(X,self.weights.T)+self.bias))

    #Does an argmax on the output to produce a classification
    def classify(self,X):
        return np.argmax(self.outputs(X),axis=1)

    #Does one iteration of backpropagation on this neural network
    def backprop(self,X,Y,alpha=0.1):
        outputs = self.outputs(X)
        delta = outputs - Y
        self.weights -= (alpha* X.T.dot(delta)).T
        self.bias -= alpha * delta.sum(axis=0)

#Reformats trainY and testY to work with softmax backprop        
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

#Whether to use the model we generated or random weights
use_model = True
model_network = network = NNOneLayer(weights=weightmatrix,bias=b,shape=weightmatrix.shape)
random_network = NNOneLayer(shape=weightmatrix.shape)

x_points =[]
model_train_rmse = []
model_test_rmse = []
model_train_acc = []
model_test_acc = []

random_train_rmse = []
random_test_rmse = []
random_train_acc = []
random_test_acc = []

def getResults(network):
    ctrain = network.classify(trainX)
    ctest = network.classify(testX)
    train_output = network.outputs(trainX)
    test_output = network.outputs(testX)
    train_rmse = np.mean( (train_output-trainY2)**2)
    test_rmse = np.mean( (test_output-testY2)**2)
    train_acc = len(ctrain[ctrain==trainY])/len(ctrain)
    test_acc = len(ctest[ctest==testY])/len(ctest)
    return train_rmse,test_rmse,train_acc,test_acc

def weightToImage(w):
    w = np.array(w)
    w -= w.min()
    w /= np.linalg.norm(w)
    w *= 255
    w = w.reshape(28,28)
    scale = 20
    return scipy.ndimage.zoom(w,[scale,scale],order=0).astype(int)

for i in range(501):
    if i % 100 == 0:
        x_points.append(i)
        train_rmse,test_rmse,train_acc,test_acc = getResults(model_network)
        model_train_rmse.append(train_rmse)
        model_test_rmse.append(test_rmse)
        model_train_acc.append(train_acc)
        model_test_acc.append(test_acc)
        train_rmse,test_rmse,train_acc,test_acc = getResults(random_network)
        random_train_rmse.append(train_rmse)
        random_test_rmse.append(test_rmse)
        random_train_acc.append(train_acc)
        random_test_acc.append(test_acc)

        for i in range(10):
            model_img = weightToImage(model_network.weights[i])
            rand_img = weightToImage(random_network.weights[i])
            cv2.imwrite("images/model_"+str(i)+"_w.jpg",model_img)
            cv2.imwrite("images/rand_" +str(i)+"_w_blur.jpg",rand_img)
    model_network.backprop(trainX,trainY2,0.00001)
    random_network.backprop(trainX,trainY2,0.00001)



fig,ax = plt.subplots()
ax.set(xlabel = "iteration",ylabel="Accuracy %",title="Overall Training/Testing Accuracy")
ax.plot(x_points,model_train_acc,"k--",label="Model Train")
ax.plot(x_points,model_test_acc,"k",label="Model Test")

ax.plot(x_points,random_train_acc,"r--",label="Random Train")
ax.plot(x_points,random_test_acc,"r",label="Random Test")
legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
ax.grid()
plt.show()

plt.clf()
digitModelAcc = [0]*9
digitRandAcc = [0]*9

digitTests = mnist.get_training_set()
for i in range(10):
    digitModelAcc[i] = 100*list(model_network.classify(digitTests[i])).count(i)/len(digitTests[i])
    digitRandAcc[i] = 100*list(random_network.classify(digitTests[i])).count(i)/len(digitTests[i])



#means_frank = (90, 55, 40, 65)
#means_guido = (85, 62, 54, 20)

# create plot
fig, ax = plt.subplots()
index = np.arange(10)
bar_width = 0.3
opacity = 0.8

rects1 = plt.bar(index, digitModelAcc, bar_width,
alpha=opacity,
color='b',
label='Model')

rects2 = plt.bar(index + bar_width, digitRandomAcc, bar_width,
alpha=opacity,
color='g',
label='Random')

plt.xlabel('Digit')
plt.ylabel('Accuracy (%)')
plt.title('Digit accuracy')
plt.xticks(index + bar_width, tuple(range(10)))
plt.legend()

plt.tight_layout()
plt.show()



