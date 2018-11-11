# function p = predict(Theta1, Theta2, X)
# %PREDICT Predict the label of an input given a trained neural network
# %   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
# %   trained weights of a neural network (Theta1, Theta2)
#
# % Useful values
# m = size(X, 1);
# num_labels = size(Theta2, 1);
#
# % You need to return the following variables correctly
# p = zeros(size(X, 1), 1);
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: Complete the following code to make predictions using
# %               your learned neural network. You should set p to a
# %               vector containing labels between 1 to num_labels.
# %
# % Hint: The max function might come in useful. In particular, the max
# %       function can also return the index of the max element, for more
# %       information see 'help max'. If your examples are in rows, then, you
# %       can use max(A, [], 2) to obtain the max for each row.
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
# % =========================================================================
#
#
# end


import numpy as np
from sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    m, n = X.shape
    num_labels = Theta2.shape[0]

    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # print('num_labels: ', num_labels)
    # print('Theta1: ', Theta1.shape)
    # print('Theta2: ', Theta2.shape)
    # print('X: ', X.shape)

    a2 = sigmoid(X.dot(Theta1.T))
    m, n = a2.shape
    a2 = np.concatenate((np.ones((m, 1)), a2), axis=1)

    # print('a2: ', a2.shape)

    a3 = sigmoid(a2.dot(Theta2.T))

    # print('a3: ', a3.shape)

    p = np.argmax(a3, axis=1)

    return p + 1  # Matlab data is 1-indexed
