import numpy as np


def _zero_threshold_binarizer(x):
    """activation function that outputs 1 if a number
    is above 0 and outputs 0 if it's equal or below
    """
    return (x > 0).astype(np.int0)


def neuron_above_n(n=5, tol=.0001):
    "creates a neuron that decides whether a number is below or above `n`"
    neuron = {
        'w': np.array([1.]).reshape(1, -1),
        'b': np.array([-n + tol]).reshape(1, -1),
        'sigma': _zero_threshold_binarizer
    }
    return neuron


def predict(values_above, n=5):
    """makes a prediction with a neuron that 
    decides whether or not a number is above `n`"""
    neuron = neuron_above_n(n=n)
    w = neuron['w']
    b = neuron['b']
    sigma = neuron['sigma']
    a = w.dot(x) + b
    output = sigma(a)
    return output



if __name__ == '__main__':
    # for 5
    # another way of thinking about this is 
    x = np.array([1, 3, 5, 7]).reshape(1, -1)
    y = np.array([0, 0, 1, 1]).reshape(1, -1)
    assert list(predict(x, 5).ravel()) == list(y.ravel())


    # for 50
    # simply change the `b` otherwise known as bias
    x = np.array([1, 3, 5, 60]).reshape(1, -1)
    y = np.array([0, 0, 0, 1]).reshape(1, -1)
    assert list(predict(x, 50).ravel()) == list(y.ravel())
    