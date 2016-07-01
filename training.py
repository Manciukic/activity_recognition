#!/usr/bin/python
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from sklearn import cross_validation
from sklearn import svm, ensemble
from sklearn.tree import DecisionTreeClassifier


def multiple_tests_C(X, Y, start=1, end=1001, step=20):
    """
    This function tests the performance of several C in a LinearSVC
    """
    # List of test values for C to get the best performance
    Clist = range(start, end, step)
    M = 0  # MAX value reached
    Slist = []  # List keeping track of mean score to draw a graph
    for C in Clist:
        clf = svm.LinearSVC(C=C)
        scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
        Slist.append(scores.mean())
        # If score better than MAX, show it and save it!!
        if scores.mean() > M:
            print "C=%.2f\t Accuracy %.5f (+/- %.5f)" % (C, scores.mean(), scores.std() * 2)
            M = scores.mean()
    # Draw graph
    plt.plot(Clist, Slist)
    plt.show()


def multiple_tests_LinearSVC(X, Y, C, n=50):
    """
    This function tests the performance a LinearSVC classifier among several tests
    """
    M = 0
    Slist = []
    for i in range(n):
        clf = svm.LinearSVC(C=C)
        scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
        Slist.append(scores.mean())
        if scores.mean() > M:
            print "Accuracy %.5f (+/- %.5f)" % (scores.mean(), scores.std() * 2)
            M = scores.mean()
    plt.plot(range(n), Slist)
    plt.show()


def multiple_tests_Tree(X, Y, n=50):
    """
    This function tests the performance a DecisionTree classifier among several tests
    """
    M = 0
    Slist = []
    for i in range(n):
        clf = DecisionTreeClassifier()
        scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
        Slist.append(scores.mean())
        if scores.mean() > M:
            print "Accuracy %.5f (+/- %.5f)" % (scores.mean(), scores.std() * 2)
            M = scores.mean()
    plt.plot(range(n), Slist)
    plt.show()


def pick_random_values_stratified(X, Y, rate=0.7):
    """
    Stratified random sample of X and Y data
    Output: list containing [X, Y]
    """
    # Separation indexes between different data (0 is always included)
    indexes = [0]
    # index of type
    idx = 0
    # The result X and Y
    new_X = []
    new_Y = []

    size = Y.shape[0]

    for i in range(size):
        if int(Y[i]) != idx:  # If the index of type changes, save it in indexes
            idx += 1
            indexes.append(int(i))
    indexes.append(size)  # We have to include also the last element

    for i in range(1, len(indexes)):
        # Take the elements in the interval [start:end]
        start = int(indexes[i - 1])
        end = int(indexes[i])
        n = int(ceil(float(end - start) * rate))  # number of elements
        # print "start=%d end=%d n=%d" % (start, end, n)
        temp = X[start:end, :]  # Get the right values in the interval
        np.random.shuffle(temp)  # Shuffle them to have random values
        new_X.append(temp[:n, :])  # Pick the first n elements
        new_Y.append(np.ones(n) * (i - 1))  # Save their indexes!!

    return [np.concatenate(new_X), np.concatenate(new_Y)]


def pick_random_values_rate(X, Y, length=100, rates=[0.05, 0.05, 0.1, 0.2, 0.3, 0.3]):
    """
    Rate random sample of X and Y data
    Output: list containing [X, Y]
    """
    # Separation indexes between different data (0 is always included)
    indexes = [0]
    # index of type
    idx = 0
    # The result X and Y
    new_X = []
    new_Y = []

    size = Y.shape[0]
    ns = []
    for rate in rates:
        ns.append(int(rate * length))
    ns[-1] = length - sum(ns[0:-1])
    for i in range(size):
        if int(Y[i]) != idx:  # If the index of type changes, save it in indexes
            idx += 1
            indexes.append(int(i))
    indexes.append(size)  # We have to include also the last element

    for i in range(1, len(indexes)):
        # Take the elements in the interval [start:end]
        start = int(indexes[i - 1])
        end = int(indexes[i])
        n = ns[i - 1]  # number of elements
        temp = X[start:end, :]  # Get the right values in the interval
        np.random.shuffle(temp)  # Shuffle them to have random values
        new_X.append(temp[:n, :])  # Pick the first n elements
        new_Y.append(np.ones(n) * (i - 1))  # Save their indexes!!

    return [np.concatenate(new_X), np.concatenate(new_Y)]


def predict_evaluate(X, Y, clf, cv=5):
    test_size = int(X.shape[0] / cv)
    Y.shape = [Y.shape[0], 1]
    frame = np.concatenate((X, Y), axis=1)
    np.random.shuffle(frame)
    test_X = frame[:test_size, :-1]
    test_Y = frame[:test_size, -1]
    train_X = frame[test_size:, :-1]
    train_Y = frame[test_size:, -1]
    clf = clf.fit(train_X, train_Y)
    labels_predict = clf.predict(test_X)
    score = clf.score(test_X, test_Y)
    print "Accuracy %.5f" % (score)
    result = np.zeros((test_Y.shape[0], 2))
    for i in range(test_Y.shape[0]):
        result[i, 0] = test_Y[i]
        result[i, 1] = labels_predict[i]
    result = np.sort(result, axis=0)
    size = result.shape[0]
    plt.plot(range(size), result[:, 0], range(size), result[:, 1])
    plt.show()


def multiple_tests_RandomForest(X, Y, n=50):
    """
    This function tests the performance a Random Forest classifier among several tests
    """
    M = 0
    sum = 0
    Slist = []
    for i in range(n):
        print "i =", i
        clf = ensemble.RandomForestClassifier()
        scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
        Slist.append(scores.mean())
        sum += scores.mean()
        if scores.mean() > M:
            print "Accuracy %.5f (+/- %.5f)" % (scores.mean(), scores.std() * 2)
            M = scores.mean()
    plt.plot(range(n), Slist)
    plt.show()
    print "Mean = ", sum / n
