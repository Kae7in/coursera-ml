import numpy as np
import matplotlib as plt
from matplotlib import pyplot
plt.use('TkAgg')
import matplotlib.pyplot as plt
from mapFeature import mapFeature


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
        if theta.size <= 3:
            x_coords = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
            y_coords = (-1/theta[2])*(theta[0] + theta[1]*x_coords)
            plt.plot(x_coords, y_coords, 'b-', label='Decision boundary')
        else:
            # Here is the grid range
            u = np.linspace(-1, 1.5, 50)
            v = np.linspace(-1, 1.5, 50)

            z = np.zeros((u.size, v.size))
            # Evaluate z = theta*x over the grid
            for i, ui in enumerate(u):
                for j, vj in enumerate(v):
                    z[i, j] = np.dot(mapFeature(ui, vj), theta)

            z = z.T  # important to transpose z before calling contour
            # print(z)

            # Plot z = 0
            pyplot.contour(u, v, z, levels=[0], linewidths=2, colors='g')
            pyplot.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.4)

    plt.show()
