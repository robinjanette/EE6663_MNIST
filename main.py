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
    total = 0
    misc = 0
    for i in range(len(misclassified)):
        total += len(test[i])
        misc += len(misclassified[i])
        print("Digit", i, ":", len(misclassified[i]), "out of", len(test[i]), (len(misclassified[i])/len(test[i]))*100, "error")
    print("Total error:", (misc/total)*100)
        
    #outputmeanmedian.vis_task1(meanmedianvar)
    
    #output the weight matricies as an image
    row_05 = np.zeros((280, 1))
    row_16 = np.zeros((280, 1))
    row_27 = np.zeros((280, 1))
    row_38 = np.zeros((280, 1))
    row_49 = np.zeros((280, 1))
    for i in range(10):
        weight = np.reshape(weightmatrix[i], (28,28)) * 255
        weight = outputmeanmedian.img_resize(10, weight)
        
        if i % 5 == 0:
            row_05 = np.hstack([row_05, weight])
        elif i % 5 == 1:
            row_16 = np.hstack([row_16, weight])
        elif i % 5 == 2:
            row_27 = np.hstack([row_27, weight])
        elif i % 5 == 3:
            row_38 = np.hstack([row_38, weight])
        elif i % 5 == 4:
            row_49 = np.hstack([row_49, weight])
    
    all_rows = np.vstack([row_05, row_16])
    all_rows = np.vstack([all_rows, row_27])
    all_rows = np.vstack([all_rows, row_38])
    all_rows = np.vstack([all_rows, row_49])
    cv2.imwrite('images/Weights.jpg', all_rows)
    
    #output the first 20 misclassified images for each digit
    for i in range(10):
        misclassified[i] = misclassified[i][:20]
        row_05 = np.zeros((280, 1))
        row_16 = np.zeros((280, 1))
        row_27 = np.zeros((280, 1))
        row_38 = np.zeros((280, 1))
        row_49 = np.zeros((280, 1))
        for j in range(20):
            misc = np.reshape(misclassified[i][j], (28,28)) * 255
            misc = outputmeanmedian.img_resize(10, misc)
    
            if j % 5 == 0:
                row_05 = np.hstack([row_05, misc])
            elif j % 5 == 1:
                row_16 = np.hstack([row_16, misc])
            elif j % 5 == 2:
                row_27 = np.hstack([row_27, misc])
            elif j % 5 == 3:
                row_38 = np.hstack([row_38, misc])
            elif j % 5 == 4:
                row_49 = np.hstack([row_49, misc])
    
        all_rows = np.vstack([row_05, row_16])
        all_rows = np.vstack([all_rows, row_27])
        all_rows = np.vstack([all_rows, row_38])
        all_rows = np.vstack([all_rows, row_49])
        cv2.imwrite('images/Misclassified' + str(i) + '.jpg', all_rows)    
    
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
