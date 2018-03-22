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

if __name__ == '__main__':
    #  ==================== Part 1: Basic Function ====================
    # print(id_matrix())  # warmUpExercise

    #  ==================== Part 2: Plotting ==========================
    # Read the data file
    data = np.loadtxt('ex1data1.txt', delimiter=',')

    # Store the strings as float arrays
    x = np.array(data[:, 0], ndmin=2).reshape(len(data), 1)
    y = np.array(data[:, 1], ndmin=2).reshape(len(data), 1)
    m = len(data)

    # Plot the data
    plt.plot(x, y, 'rx')
    plt.ylabel('Profit in $10,000\'s')
    plt.xlabel('Population of City in 10,000s')
    plt.axis([0, 25, -5, 25])
    plt.show()

    #  ==================== Part 3: Cost and Gradient descent =========
    thetas = np.zeros((2, 1))
    X = np.concatenate((np.ones((m, 1)), x), axis=1)

    cost = compute_cost(X, y, thetas)
    print(cost)

    thetas = np.array([[-1.0], [2.0]])
    cost = compute_cost(X, y, thetas)
    print(cost)

    # Find thetas using gradient descent.
    iterations = 1500
    alpha = 0.01
    thetas, j_history = gradient_descent(X, y, thetas, alpha, iterations)
    print(thetas)

    # Show best fit line.
    plt.figure()
    plt.plot(x, y, 'rx')
    plt.ylabel('Profit in $10,000\'s')
    plt.xlabel('Population of City in 10,000s')
    plt.axis([0, 25, -5, 25])
    plt.plot(x, X.dot(thetas), '-')
    plt.show()

    # Predict values!
    predict1 = np.dot([[1, 3.5]], thetas)  # Predict profit for population of 35,000
    print('For population = 35,000, we predict a profit of\n', predict1 * 10000)
    predict2 = np.dot([[1, 7]], thetas)  # Predict profit for population of 70,000
    print('For population = 70,000, we predict a profit of\n', predict2 * 10000)

    #  ==================== Part 4: Visualizing J(theta_0, theta_1) ===
    print('Visualizing J(theta_0, theta_1) ...\n')

    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # Initialize J_vals to a matrix of 0's
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    # Fill out J_vals
    for i in range(0, len(theta0_vals)):
        for j in range(0, len(theta1_vals)):
            t = [theta0_vals[i], theta1_vals[j]]
            J_vals[i, j] = compute_cost(X, y, t)

    J_vals = np.swapaxes(J_vals, 0, 1)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    meshx, meshy = np.meshgrid(theta0_vals, theta1_vals)
    surf = ax.plot_surface(meshx, meshy, J_vals, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.show()

