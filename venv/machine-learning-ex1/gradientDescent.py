import numpy as np
from computeCost import compute_cost


def gradient_descent(X, y, thetas, alpha, iterations):
    m = len(y)
    t = thetas.copy()
    j_history = []

    for _ in range(iterations):
        temptheta = t.copy()
        j_history.append(compute_cost(X, y, temptheta))  # Record cost
        for j in range(len(temptheta)):
            x = np.array(X[:, j]).reshape(len(X), 1)  # Slice x-sub-j feature
            temptheta[j] = t[j] - ((alpha/m) * np.sum((h(X, t) - y)*x))
        t = temptheta

    return t, j_history


def h(X, theta):
    return np.dot(X, theta)
