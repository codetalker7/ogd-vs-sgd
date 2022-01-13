# ogd-vs-sgd

This repository contains an implementation of the *vanilla gradient descent* algorithm and the *stochastic gradient descent* algorithm. The algorithm implementations are inspired from [Elad Hazan's book on Online Convex Optimization](https://arxiv.org/abs/1909.05207) (see the implementations of SVM in the book).
The purpose of the code is to do binary classification (using the SVM algorithm) on the classes in the MNIST and CIFAR-10 datasets. Currently, we have included only the splitted MNIST dataset; however CIFAR-10 can be included with ease. 

# How to run the code?

The interface is very simple: there are three main classes, namely 
  - `GD`: This is the parent class for the two algorithms. The constructor expects four parameters: `data`, which is a 2D numpy array containing the data points, `labels`, which is a 1D numpy array containing the labels and parameters `class1`, `class2`, which are the labels of the classes we are trying to classify. For example, if we are trying to classify digits `0` and `1` from the MNIST dataset, we will pass `0`, `1` as these parameters. Look at the `mnist-0-1-classification.ipynb`, `mnist-5-8-classification.ipynb` and `mnist-6-9-classification.ipynb` example notebooks for instance.
  - Then there are two subclasses `VanillaGD` and `SGD`; as the names suggest, these two classes implement the two gradient descent algorithms. They inherit from the `GD` class, and implement the `train` method. The `train` method in each class takes two parameters, namely `T` and `lam`; `T` is the number of time steps to run the algorithms, and `lam` is the hyperparameter. See the example files.

# `mnist_binary`
The `mnist_binary` directory contains training and test data for all 45 combinations of the binary classifications that can be done on the MNIST dataset. Importing any of these datasets is easy to do using the `MNISTBinary` class; see the three examples. The interface is not hard to understand. 

# The three results

As can be seen from the three examples, the results are as expected: vanilla GD takes a much longer time than SGD, and has a higher accuracy on the test data than SGD (if run with the same parameters). However, in two of the three cases, their accuracies are comparable, and SGD beats vanilla GD in the required running time. 

# Results for all 45 combinations
The `./mnist_binary_results` directory contains a `results.md` file, which contains the experiment results for all 
45 combinations of the possible binary classifications in MNIST dataset. The table contains the accuracies and running times of the OGD and SGD algorithms.
