# function [J, grad] = lrCostFunction(theta, X, y, lambda)
# %LRCOSTFUNCTION Compute cost and gradient for logistic regression with
# %regularization
# %   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
# %   theta as the parameter for regularized logistic regression and the
# %   gradient of the cost w.r.t. to the parameters.
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
# % Hint: The computation of the cost function and gradients can be
# %       efficiently vectorized. For example, consider the computation
# %
# %           sigmoid(X * theta)
# %
# %       Each row of the resulting matrix will contain the value of the
# %       prediction for that example. You can make use of this to vectorize
# %       the cost function and gradient computations.
# %
# % Hint: When computing the gradient of the regularized cost function,
# %       there're many possible vectorized solutions, but one solution
# %       looks like:
# %           grad = (unregularized gradient for logistic regression)
# %           temp = theta;
# %           temp(1) = 0;   % because we don't add anything for j = 0
# %           grad = grad + YOUR_CODE_HERE (using the temp variable)
# %
#
#
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
# grad = grad(:);
#
# end


import numpy as np
from sigmoid import sigmoid


def lr_cost_function(theta, X, y, lamb=0):
    J = compute_cost(theta, X, y, lamb)
    grad = compute_gradient(theta, X, y, lamb)

    return J, grad


def compute_cost(theta, X, y, lamb=0):
    m = y.size
    y = y.reshape((len(y), 1))

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