import numpy as np
import matplotlib as plt
plt.use('TkAgg')
import matplotlib.pyplot as plt


def plot_data(X, y, theta=np.array([])):
    """ Plot student admission data on a graph """

    # Set y and x axis labels for scatter plot
    plt.ylabel('Exam 2 score')
    plt.xlabel('Exam 1 score')

    admitted = np.where(y == 1)[0]
    not_admitted = np.where(y == 0)[0]

    # Plot all admitted students
    plt.scatter(X[admitted, :1], X[admitted, 1:], marker='+', label='Admitted', c='black')

    # Plot all non-admitted students
    plt.scatter(X[not_admitted, :1], X[not_admitted, 1:], marker='o', label='Not admitted', c='yellow', edgecolors='black')

    # Set legend for scatter plot
    plt.legend(loc='upper right', fontsize=8)

    # Show best fit line
    if theta.size != 0:
        x_coords = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
        y_coords = (-1/theta[2])*(theta[0] + theta[1]*x_coords)
        plt.plot(x_coords, y_coords, 'b-', label='Decision boundary')

    plt.show()
