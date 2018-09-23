import numpy as np
from computeCost import compute_cost


def gradient_descent(X, y, thetas, alpha, iterations):
    m = len(y)
    t = thetas.copy()  # copy so as to not modify original thetas variable
    j_history = []

    for _ in range(iterations):
        new_t = t.copy()
        j_history.append(compute_cost(X, y, new_t))  # Record cost before making changes
        for j in range(len(new_t)):
            x = np.array(X[:, j]).reshape(len(X), 1)  # Slice j feature of X
            new_t[j] = t[j] - ((alpha/m) * np.sum((h(X, t) - y)*x))
        t = new_t

    return t, j_history


def h(X, theta):
    return np.dot(X, theta)
