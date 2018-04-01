import numpy as np
import matplotlib as plt
plt.use('TkAgg')
import matplotlib.pyplot as plt


''' This is a longer but more efficient implementation that avoids calling plt.scatter() for every data point. '''
def plot_data(X, y):
    X_admitted = np.array([], ndmin=2)
    X_not_admitted = np.array([], ndmin=2)

    for row in range(len(X)):  # Group admitted and not-admitted data points into separate matrices
        if y[row]:
            X_admitted = np.append(X_admitted, X[row])
        else:
            X_not_admitted = np.append(X_not_admitted, X[row])

    # Set y and x axis labels for scatter plot
    plt.ylabel('Exam 2 score')
    plt.xlabel('Exam 1 score')

    # Now we can call plt.scatter for the entire admitted group
    X_admitted = X_admitted.reshape(int(len(X_admitted)/2), 2)
    x1 = X_admitted[:, :1]
    x2 = X_admitted[:, 1:]
    plt.scatter(x1, x2, marker='+', label='Admitted', c='black')

    # Similarly, we call plt.scatter for the entire not-admitted group
    X_not_admitted = X_not_admitted.reshape(int(len(X_not_admitted)/2), 2)
    x1 = X_not_admitted[:, :1]
    x2 = X_not_admitted[:, 1:]
    plt.scatter(x1, x2, marker='o', label='Not admitted', c='yellow', edgecolors='black')

    # Set legend for scatter plot
    plt.legend(loc='upper right', fontsize=8)

    plt.show()
