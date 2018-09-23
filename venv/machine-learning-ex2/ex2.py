import matplotlib as plt
plt.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from plotData import plot_data
from costFunction import cost_function, gradient
from scipy.optimize import minimize


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

    # Add x-sub-0 (vector of 1's)
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    theta = np.zeros(X.shape[1])

    J = cost_function(theta, X, y)
    grad = gradient(theta, X, y)
    print(J)
    print(grad)

    result = minimize(cost_function, theta, args=(X, y), method=None, jac=gradient, options={'maxiter': 4m00})

    print(result)