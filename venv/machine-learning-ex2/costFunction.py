# function [J, grad] = costFunction(theta, X, y)
# %COSTFUNCTION Compute cost and gradient for logistic regression
# %   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
# %   parameter for logistic regression and the gradient of the cost
# %   w.r.t. to the parameters.
#
# % Initialize some useful values
# m = length(y); % number of training examples
#
# % You need to return the following variables correctly
# J = 0;
# grad = zeros(size(theta));
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: Compute the cost of a particular choice of theta.
# %               You should set J to the cost.
# %               Compute the partial derivatives and set grad to the partial
# %               derivatives of the cost w.r.t. each parameter in theta
# %
# % Note: grad should have the same dimensions as theta
# %
#
#
#
#
#
#
#
#
# % =============================================================
#
# end

import numpy as np
from sigmoid import sigmoid


def compute_cost(theta, X, y, lamb=0):
    m = y.size

    J = (1 / m) * (-y.T.dot(np.log(h(X, theta))) - (1 - y).T.dot(np.log(1 - h(X, theta)))) \
        + (lamb / (2 * m)) * np.sum(np.power(theta[1:], 2))

    if np.isnan(J[0]):
        return np.inf
    return J[0]


def compute_gradient(theta, X, y, lamb=0):
    m = len(y)
    t = theta.reshape(-1, 1)
    y = y.reshape((len(y), 1))
    gradient = np.zeros((len(t), 1))

    masked_t = np.array(t)
    masked_t[0] = 0

    # Compute gradient
    gradient = (1 / m) * np.dot((h(X, t) - y).T, X).reshape((len(X.T), 1)) + ((lamb / m) * masked_t)

    return gradient.flatten()


def h(x, theta):
    return sigmoid(np.dot(x, theta))
