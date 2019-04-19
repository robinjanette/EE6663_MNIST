#Code from https://github.com/hsjeong5/MNIST-for-Numpy

import numpy as np
from urllib import request
import gzip
import pickle
import os

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

mnist = {}



def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

def get_test_set():
    
    # Create the holder lists
    one = []
    two = []
    three = []
    four = []
    five = []
    six = []
    seven = []
    eight = []
    nine = []
    ten = []

    # Append the data set to the proper list and normalize
    for i in range(0, len(mnist["test_labels"])):
        if mnist["test_labels"][i] == 0:
            one.append(mnist["test_images"][i]/255)
        elif mnist["test_labels"][i] == 1:
            two.append(mnist["test_images"][i]/255)
        elif mnist["test_labels"][i] == 2:
            three.append(mnist["test_images"][i]/255)
        elif mnist["test_labels"][i] == 3:
            four.append(mnist["test_images"][i]/255)
        elif mnist["test_labels"][i] == 4:
            five.append(mnist["test_images"][i]/255)
        elif mnist["test_labels"][i] == 5:
            six.append(mnist["test_images"][i]/255)
        elif mnist["test_labels"][i] == 6:
            seven.append(mnist["test_images"][i]/255)
        elif mnist["test_labels"][i] == 7:
            eight.append(mnist["test_images"][i]/255)
        elif mnist["test_labels"][i] == 8:
            nine.append(mnist["test_images"][i]/255)
        elif mnist["test_labels"][i] == 9:
            ten.append(mnist["test_images"][i]/255)

    # Convert to np arrays
    one = np.array(one)
    two = np.array(two)
    three = np.array(three)
    four = np.array(four)
    five = np.array(five)
    six = np.array(six)
    seven = np.array(seven)
    eight = np.array(eight)
    nine = np.array(nine)
    ten = np.array(ten)
    
    return one,two,three,four,five,six,seven,eight,nine,ten

def get_training_set():

    # Create the holder lists
    one = []
    two = []
    three = []
    four = []
    five = []
    six = []
    seven = []
    eight = []
    nine = []
    ten = []

    # Append the data set to the proper list and normalize
    for i in range(0, len(mnist["training_labels"])):
        if mnist["training_labels"][i] == 0:
            one.append(mnist["training_images"][i]/255)
        elif mnist["training_labels"][i] == 1:
            two.append(mnist["training_images"][i]/255)
        elif mnist["training_labels"][i] == 2:
            three.append(mnist["training_images"][i]/255)
        elif mnist["training_labels"][i] == 3:
            four.append(mnist["training_images"][i]/255)
        elif mnist["training_labels"][i] == 4:
            five.append(mnist["training_images"][i]/255)
        elif mnist["training_labels"][i] == 5:
            six.append(mnist["training_images"][i]/255)
        elif mnist["training_labels"][i] == 6:
            seven.append(mnist["training_images"][i]/255)
        elif mnist["training_labels"][i] == 7:
            eight.append(mnist["training_images"][i]/255)
        elif mnist["training_labels"][i] == 8:
            nine.append(mnist["training_images"][i]/255)
        elif mnist["training_labels"][i] == 9:
            ten.append(mnist["training_images"][i]/255)

    # Convert to np arrays
    one = np.array(one)
    two = np.array(two)
    three = np.array(three)
    four = np.array(four)
    five = np.array(five)
    six = np.array(six)
    seven = np.array(seven)
    eight = np.array(eight)
    nine = np.array(nine)
    ten = np.array(ten)
    
    return one,two,three,four,five,six,seven,eight,nine,ten

def load():
    if not os.path.exists('mnist.pkl'):
        init()
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


# init()

# one,two,three,four,five,six,seven,eight,nine,ten = get_training_set()

# print (len(one),len(two),len(three),len(four),len(five),len(six),len(seven),len(eight),len(nine),len(ten))
# print (len(one)+len(two)+len(three)+len(four)+len(five)+len(six)+len(seven)+len(eight)+len(nine)+len(ten))

# one,two,three,four,five,six,seven,eight,nine,ten = get_test_set()

# print (len(one),len(two),len(three),len(four),len(five),len(six),len(seven),len(eight),len(nine),len(ten))
# print (len(one)+len(two)+len(three)+len(four)+len(five)+len(six)+len(seven)+len(eight)+len(nine)+len(ten))


