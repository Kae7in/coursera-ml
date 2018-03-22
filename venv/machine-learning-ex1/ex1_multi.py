import matplotlib as plt
plt.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import csv
import numpy as np
from warmUpExercise import id_matrix
from computeCost import compute_cost
from gradientDescent import gradient_descent
from featureNormalize import feature_normalize

if __name__ == '__main__':
    # ================ Part 1: Feature Normalization ================

    # Read the data file
    data = np.loadtxt('ex1data2.txt', delimiter=',')

    # Store the strings as float arrays
    X = np.array(data[:, :2])
    y = np.array(data[:, 2]).reshape(len(data), 1)
    m = len(data)

    # Plot size against price
    plt.plot(X[:, 0], y, 'rx')
    plt.ylabel('Price in $')
    plt.xlabel('sq.ft.')
    plt.axis([0, 5000, 0, 800000])
    # plt.show()

    # Plot beds against price
    plt.figure()
    plt.plot(X[:, 1], y, 'rx')
    plt.ylabel('Price in $')
    plt.xlabel('# of beds')
    plt.axis([0, 6, 0, 800000])
    # plt.show()

    X, mu, sigma = feature_normalize(X)
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    alpha = 0.01
    iterations = 1000
    thetas = np.zeros((3, 1))

    thetas, j_history = gradient_descent(X, y, thetas, alpha, iterations)

    test = np.array([1650, 3]).reshape(1, 2)
    test = (test - mu) / sigma  # normalize input values
    test = np.concatenate((np.ones((1, 1)), test), axis=1)
    predict1 = np.dot(test, thetas)
    print('Predicted price of a 1650 sq-ft, 3 br house: \n', predict1[0][0])
