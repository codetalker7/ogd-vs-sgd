import numpy as np
import time

class GD():
    def __init__(self, data, labels, class1, class2):
        '''
            data -> numpy array containing dataset
            labels -> numpy array containing labels
            class1 -> label of the first class
            class2 -> label of the second class
        '''
        #initialise data and labels
        self.data = data
        self.labels = labels

        #initialise classes
        self.class1 = class1
        self.class2 = class2

        #dimension -> dimension of data points
        #size -> number of data points
        self.dimension = np.shape(data)[1]
        self.size = np.shape(data)[0]

        #zeros -> vector of zeros
        self.zeros = np.zeros((self.dimension, 1))

        #training parameters
        self.theta = self.zeros
        self.trainingTime = 0

    def hingeLossGradient(self, x, a, b):
        if (b*np.dot(np.transpose(x), a) > 1):
            return self.zeros
        else:
            return -b*a

    # train to be overriden by subclass
    def train():
        pass

    # compute accuracy on test data
    def getTestAccuracy(self, test_data, test_labels):
        # n is the size of the test data
        n = np.shape(test_data)[0]

        # number of correctly classified points
        correct = 0

        for i in range(0, n):
            # get the ith test point
            ai = np.transpose(test_data[i:i + 1, :])

            # get the ith test label 
            if (test_labels[i] == self.class1):
                bi = -1
            else:
                bi = 1

            # check if self.theta correctly classifies point
            if (bi*np.dot(np.transpose(self.theta) ,ai) > 0):
                correct = correct + 1

        # return the accuracy
        return correct/n
    
    # get training parameters
    def getTrainingParameters(self):
        return (self.theta, self.trainingTime)

    # get zeros vector
    def getZeros(self):
        return self.zeros

    # get tuple of dimension and size
    def getShape(self):
        return (self.dimension, self.size)
