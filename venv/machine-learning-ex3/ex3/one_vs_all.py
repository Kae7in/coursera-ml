# function [all_theta] = oneVsAll(X, y, num_labels, lambda)
# %ONEVSALL trains multiple logistic regression classifiers and returns all
# %the classifiers in a matrix all_theta, where the i-th row of all_theta
# %corresponds to the classifier for label i
# %   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
# %   logistic regression classifiers and returns each of these classifiers
# %   in a matrix all_theta, where the i-th row of all_theta corresponds
# %   to the classifier for label i
#
# % Some useful variables
# m = size(X, 1);
# n = size(X, 2);
#
# % You need to return the following variables correctly
# all_theta = zeros(num_labels, n + 1);
#
# % Add ones to the X data matrix
# X = [ones(m, 1) X];
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: You should complete the following code to train num_labels
# %               logistic regression classifiers with regularization
# %               parameter lambda.
# %
# % Hint: theta(:) will return a column vector.
# %
# % Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
# %       whether the ground truth is true/false for this class.
# %
# % Note: For this assignment, we recommend using fmincg to optimize the cost
# %       function. It is okay to use a for-loop (for c = 1:num_labels) to
# %       loop over the different classes.
# %
# %       fmincg works similarly to fminunc, but is more efficient when we
# %       are dealing with large number of parameters.
# %
# % Example Code for fmincg:
# %
# %     % Set Initial theta
# %     initial_theta = zeros(n + 1, 1);
# %
# %     % Set options for fminunc
# %     options = optimset('GradObj', 'on', 'MaxIter', 50);
# %
# %     % Run fmincg to obtain the optimal theta
# %     % This function will return theta and the cost
# %     [theta] = ...
# %         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
# %                 initial_theta, options);
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
#
#
# % =========================================================================
#
#
# end

import numpy as np
from scipy.optimize import minimize
from lr_cost_function import lr_cost_function


def one_vs_all(X, y, num_labels, lamb=0):
    m, n = X.shape
    y = y.reshape((len(y), 1))

    all_theta = np.zeros((num_labels, n + 1))

    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # print('m: ', m)
    # print('n: ', n)
    # print('X: ', X.shape)
    # print('y: ', y.shape)
    # print('all_theta: ', all_theta.shape)

    t = np.zeros((n + 1, 1))

    for i in range(0, num_labels):
        label = 10 if i == 0 else i
        result = minimize(fun=lr_cost_function, x0=t, args=(X, (y == label).astype(int), lamb), method='TNC', jac=True)
        all_theta[i] = result.x
        print(f'one_vs_all() label {label} classfied: {result.success}')

    return all_theta