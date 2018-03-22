import numpy as np


def compute_cost(X, y, thetas):
    m = len(y)
    return (1.0/(2*m)) * np.sum((np.dot(X, thetas) - y)**2)
