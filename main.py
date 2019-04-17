''' Driver code for MNIST classification'''
import numpy
import os
import matplotlib
import cv2

def MNIST_classifier(img):
    return 0 #placeholder function, TODO

def determine_true_label(n):
    if n.find("0") > -1:
        return 0
    elif n.find("1") > -1:
        return 1
    elif n.find("2") > -1:
        return 2
    elif n.find("3") > -1:
        return 3
    elif n.find("4") > -1:
        return 4
    elif n.find("5") > -1:
        return 5
    elif n.find("6") > -1:
        return 6
    elif n.find("7") > -1:
        return 7
    elif n.find("8") > -1:
        return 8
    elif n.find("9") > -1:
        return 9
    else:
        return -1 #bad file name
    
def output_results(test_set, misclassified):
    #will also need weight patterns for each number TODO
    return

def main():
    #get the weight patterns for each number 0-9
    #TODO
    
    #initialize variables
    correct_count = 0
    misclassified = []
    
    #for every image in the testing set:
    test_set = os.listdir("test_set")
    for i in test_set:
        
        #read test image
        img = cv2.imread(i)
        
        #determine the true label of the image
        true_label = determine_true_label(i)
        if true_label == -1:
            break
        
        #classify the image and return the computed label
        label = MNIST_classifier(img)
        
        #determine if label is incorrect:
        if true_label != label:
            misclassified += [img]
        else:
            correct_count += 1
    
    #output images and the weight patterns after classification
    output_results(test_set, misclassified)

if __name__ == '__main__':
    main()
