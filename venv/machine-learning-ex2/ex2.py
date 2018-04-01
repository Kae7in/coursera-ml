import matplotlib as plt
plt.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from plotData import plot_data
from sigmoid import sigmoid


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

    