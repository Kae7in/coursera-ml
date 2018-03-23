import numpy as np
from numpy.linalg import inv


def normal_eqn(X, y):
    X_copy = X.copy()

    thetas = np.dot(inv(np.transpose(X_copy).dot(X_copy)), np.transpose(X_copy).dot(y))

    return thetas
