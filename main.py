''' Driver code for MNIST classification'''
import numpy
import os
import matplotlib
import cv2
import mmv
import gen_weights
import classify
import applyWeights
import bias
    
def output_results(weightmatrix, misclassified, test):
    print("Numerical Results")
    
    #for now, print the number of misclassified examples for every digit
    for i in range(len(misclassified)):
        print("Digit", i, ":", len(misclassified[i]), "out of", len(test[i]))
    return

def main():
    #load the dataset
    train, test = mmv.format()
    
    #get the mean, median, and variance patterns for each number 0-9
    meanmedianvar = mmv.mmv(train)
    
    #get the final weight maps for each number 0-9
    weightmatrix = gen_weights.gen_weights(meanmedianvar)
    
    #get the bias
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
    output_results(weightmatrix, misclassified, test)

if __name__ == '__main__':
    main()
