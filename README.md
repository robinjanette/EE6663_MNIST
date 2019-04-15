# EE6663_MNIST
MNIST Digit Classifier for EE 6663

# Tasks to be Completed:

Task 1 (Jacob): Write a program that computes the mean value, the median, and the variance of every pixel for each of the 10 digits from a “training” set. (You can determine the number of samples in the training set as a team based on feasibility and workload.) 

 

Task 2 (David): Initialize the weight matrix using the nominal figure for each of the digits in Task 1, which can be either the normalized mean values or the normalized median. Both cases should be implemented and tested, unless the mean and the median are very close.

 

Task 3 (Nick): Write a program that computes the network output (output = weight_matrix * normalized_test_vector) on a set of test samples, again decide the size of test set based on feasibility and workload by yourselves.

 

Task 4 (David): Write a program that will perform the final classification by adding a bias term to the output nodes so that the final output is considered true if it is greater than 0. The bias can be determined heuristically, or based on the variance of sample set for each digit obtained in Task 1, assuming the probobility distribution is Gaussian.

 

Task 5 (Robin): Write a program that automates the test process and computes the misclassification rate. The program should also be able to identify the misclassified samples.

 

Task 6 (Robin): Write a program to display the training and testing image data and the weight patterns, i.e. the nominal digits as images.

 

Task 7 (All): Write a brief report to document the team structure and task delegation, the development process, and test results. Attach the program codes to the report.
