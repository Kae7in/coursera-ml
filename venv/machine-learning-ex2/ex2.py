import matplotlib as plt
plt.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from plotData import plot_data
from sigmoid import sigmoid
from costFunction import cost_function


if __name__ == '__main__':
    ''' ================ Part 1: Feature Normalization ================ '''

    # Read the data file
    data = np.loadtxt('ex2data1.txt', delimiter=',')

    # Store the strings as float arrays
    X = np.array(data[:, :2])
    y = np.array(data[:, 2]).reshape(len(data), 1)
    m = len(data)

    # Plot the data
    # plot_data(X, y)

    # Test sigmoid function
    # print(sigmoid(0))  # Should equal 0.5
    # print(sigmoid(1))  # Should be close to 0.75
    # print(sigmoid(-1))  # Should be close to 0.25
    # print(sigmoid(5))  # Should be close to 1
    # print(sigmoid(-5))  # Should be close to 0
    # print(sigmoid(np.array([-5, -1, 0, 1, 5])))  # Should result in a vector of the same answers as above

    # Add x-sub-0 (vector of 1's)
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    theta = np.zeros((3, 1))

    J, gradient = cost_function(theta, X, y)
    print(J)
    print(gradient)
