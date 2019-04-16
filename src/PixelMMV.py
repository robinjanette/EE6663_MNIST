import cv2
import os

# read images into matrices
# calculate mean
# calculate median
# calculate variance

class PixelMMV:
    '''image_data
        - n by n matrix (dimensions of image)
        - each element is a list (np.array) of the values for that pixel 
    '''
    image_data = []

    def __init__(self, dir_str):
        directory = os.fsencode(dir_str)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            print(filename) # if filename.endswith(".png") etc...

            continue

            image = cv2.imread(filename)
            for i in len(image):
                for j in len(image[i]):
                    pass
    
    def mean():
        means = []
        return means

    def median():
        medians = []
        return medians

    def variance():
        variances = []
        return variances