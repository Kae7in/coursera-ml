import numpy as np


def feature_normalize(X):
    X_norm = X.copy()
    mu = np.array(np.mean(X_norm, axis=0)).reshape((1, np.size(X_norm, axis=1)))
    sigma = np.array(np.std(X_norm, axis=0)).reshape((1, np.size(X_norm, axis=1)))

    X_norm = (X_norm - mu)/sigma

    return X_norm, mu, sigma
