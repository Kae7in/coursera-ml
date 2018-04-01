import matplotlib as plt
plt.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from warmUpExercise import id_matrix
from computeCost import compute_cost
from gradientDescent import gradient_descent
from featureNormalize import feature_normalize
from normalEqn import normal_eqn

if __name__ == '__main__':
    ''' ================ Part 1: Feature Normalization ================ '''

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
    plt.show()

    # Plot beds against price
    plt.figure()
    plt.plot(X[:, 1], y, 'rx')
    plt.ylabel('Price in $')
    plt.xlabel('# of beds')
    plt.axis([0, 6, 0, 800000])
    plt.show()

    # Feature Normalize
    X_norm, mu, sigma = feature_normalize(X)

    # Append x-sub-0 (vector of 1's)
    X_norm = np.concatenate((np.ones((m, 1)), X_norm), axis=1)

    ''' ================ Part 2: Gradient Descent ================ '''

    # Set gradient descent parameters
    alpha = 0.01
    iterations = 1000
    thetas1 = np.zeros((3, 1))

    # Gradient descent
    thetas1, j_history = gradient_descent(X_norm, y, thetas1, alpha, iterations)

    # Prediction test
    test1 = np.array([1650, 3]).reshape(1, 2)
    test1 = (test1 - mu) / sigma  # normalize input values
    test1 = np.concatenate((np.ones((1, 1)), test1), axis=1)
    predict1 = np.dot(test1, thetas1)
    # print(thetas1)
    print('Predicted price of a 1650 sq-ft, 3 br house: \n', predict1[0][0])

    ''' ================ Part 3: Normal Equations ================ '''

    # Let's try another way...
    X_copy = X.copy()

    # Append x-sub-0 (vector of 1's)
    X_copy = np.concatenate((np.ones((m, 1)), X_copy), axis=1)

    # Use the Normal Equation
    thetas2 = normal_eqn(X_copy, y)
    # print(thetas2)

    # Prediction test
    test2 = np.array([1, 1650, 3]).reshape(1, 3)
    predict2 = np.dot(test2, thetas2)
    print('Predicted price of a 1650 sq-ft, 3 br house: \n', predict2[0][0])
