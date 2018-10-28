import numpy as np

'''
function out = mapFeature(X1, X2)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%

degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end
'''

def mapFeature(x1, x2):
    degree = 6
    # out = np.ones((x1.shape[0], 28))
    # for i in range(1, degree + 1):
    #     for j in range(0, i + 1):
    #         out[:, j + 1] = np.power(x1, i - j) * np.power(x2, j)
    #
    # return out

    if x1.ndim > 0:
        out = [np.ones(x1.shape[0])]
    else:
        out = [np.ones(1)]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((x1 ** (i - j)) * (x2 ** j))

    if x1.ndim > 0:
        return np.stack(out, axis=1)
    else:
        return np.array(out)
