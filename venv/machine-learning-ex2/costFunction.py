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


def cost_function(theta, X, y):
    m = y.size
    sig = sigmoid(X.dot(theta))

    J = -1 * (1 / m) * (np.log(sig).T.dot(y) + np.log(1 - sig).T.dot(1 - y))

    if np.isnan(J[0]):
        return (np.inf)
    return (J[0])


def gradient(theta, X, y):
    m = len(y)
    t = theta.reshape(-1, 1)
    gradient = np.zeros((len(t), 1))

    # Compute gradient
    for j in range(len(gradient)):
        x_j = np.array(X[:, j]).reshape(len(X), 1)  # Slice x-sub-j feature
        gradient[j] = (1 / m) * np.sum((h(X, t) - y) * x_j)

    return gradient.flatten()


def h(x, theta):
    return sigmoid(np.dot(x, theta))
