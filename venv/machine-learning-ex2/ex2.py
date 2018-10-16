import matplotlib as plt
plt.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from plotData import plot_data
from costFunction import compute_cost, compute_gradient
from scipy.optimize import minimize
from sigmoid import sigmoid


if __name__ == '__main__':
    ''' ================ Part 1: Feature Normalization ================ '''

    # Read the data file
    data = np.loadtxt('ex2data1.txt', delimiter=',')

    # Store the strings as float arrays
    x = np.array(data[:, :2])
    y = np.array(data[:, 2]).reshape(len(data), 1)
    m = len(data)

    # Plot the data
    plot_data(x, y)

    # Add x-sub-0 (vector of 1's)
    X = np.concatenate((np.ones((m, 1)), x), axis=1)

    t = np.zeros((X.shape[1], 1))

    J = compute_cost(t, X, y)
    gradient = compute_gradient(t, X, y)

    result = minimize(compute_cost, t, args=(X, y), method=None, jac=compute_gradient, options={'maxiter': 400})

    sample_test = np.array([1, 45, 85])
    print(sigmoid(np.dot(sample_test, result.x.T)))

    plot_data(x, y, result.x.T)
