import numpy as np
from MNISTBinary import MNISTBinary
from VanillaGD import VanillaGD
from SGD import SGD

#writing the column headings of the results table
print('| Binary Labels | OGD Accuracy | OGD Running Time | SGD Accuracy | SGD Running Time |')
print('| ------------- | ------------ | ---------------- | ------------ | ---------------- |')

#looping over all possible combinations of the labels
for i in range(0, 10):
    for j in range(i + 1, 10):
        #get the training dataset for i-j classification
        dataset = MNISTBinary('./mnist_binary/train/mnist_' + str(i) + '_' + str(j) + '.csv')

        #getting the data as numpy arrays
        X = dataset.getData()
        Y = dataset.getLabels()

        #training vanilla GD and SGD with these sets
        ogd_classifier = VanillaGD(X, Y, i, j)
        sgd_classifier = SGD(X, Y, i, j)

        #training the classifiers 
        ogd_classifier.train(1000, 0.8)
        sgd_classifier.train(1000, 0.8)

        #testing the classifiers
        #getting test data
        dataset = MNISTBinary('./mnist_binary/test/mnist_' + str(i) + '_' + str(j) + '.csv')
        X_test = dataset.getData()
        Y_test = dataset.getLabels()

        #getting the accuracies and training times
        ogd_training_time = ogd_classifier.getTrainingParameters()[1]
        sgd_training_time = sgd_classifier.getTrainingParameters()[1]

        ogd_accuracy = ogd_classifier.getTestAccuracy(X_test, Y_test)
        sgd_accuracy = sgd_classifier.getTestAccuracy(X_test, Y_test)

        #printing it out to the table
        print(
            '|' +
            str(i) + ',' + str(j) + 
            '|' + 
            str(ogd_accuracy) +
            '|' + 
            str(ogd_training_time) + 
            '|' + 
            str(sgd_accuracy) + 
            '|' + 
            str(sgd_training_time) + 
            '|'
        )