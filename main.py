''' Driver code for MNIST classification'''
import numpy as np
import os
import matplotlib
import cv2
import mmv
import gen_weights
import classify
import applyWeights
import bias
import outputmeanmedian
    
def output_results(meanmedianvar, weightmatrix, misclassified, test):
    print("Numerical Results")
    
    #for now, print the number of misclassified examples for every digit
    for i in range(len(misclassified)):
        print("Digit", i, ":", len(misclassified[i]), "out of", len(test[i]))
        
    #outputmeanmedian.vis_task1(meanmedianvar)
    
    
    
    return

def main():
    #load the dataset
    train, test = mmv.format()
    
    #get the mean, median, and variance patterns for each number 0-9
    meanmedianvar = mmv.mmv(train)
    
    #get the final weight maps for each number 0-9
    weightmatrix = gen_weights.gen_weights(meanmedianvar)
    
    #get the bias
    outputs = [applyWeights.applyWeights(weightmatrix,np.array(train[i]).T) for i in range(10) ]
    b = bias.gen_bias(weightmatrix)
    
    #initialize variables
    misclassified = [[], [], [], [], [], [], [], [], [], []]
    
    #run on test data
    for i in range(len(test)):
        for j in test[i]:
            wx = applyWeights.applyWeights(weightmatrix, j)
            wxb = classify.classify(wx, b)
            
            if i != wxb:
                misclassified[i] += [j]
    
    #output images and the weight patterns after classification
    output_results(meanmedianvar, weightmatrix, misclassified, test)

if __name__ == '__main__':
    main()
