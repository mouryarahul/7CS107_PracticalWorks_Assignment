import numbers
import numpy as np

def is_stochastic(X):
    """True if `X` contains probabilities that sum to 1 along the columns"""
    msg = "Array should be stochastic along the columns"
    assert len(X[X < 0]) == len(X[X > 1]) == 0, msg
    assert np.allclose(np.sum(X, axis=1), np.ones(X.shape[0])), msg
    return True


def is_number(a):
    """Check that a value `a` is numeric"""
    return isinstance(a, numbers.Number)

def indices_to_one_hot(indices, num_classes):
    N, d = indices.shape
    one_hot_vector = np.zeros((N, num_classes))
    # Set the index position to 1
    one_hot_vector[np.arange(N), indices.flatten()] = 1
    return one_hot_vector

def is_one_hot(x):
    """Return True if array `x` is a binary array with a single 1"""
    msg = "Matrix should be one-hot binary"
    assert np.array_equal(x, x.astype(bool)), msg
    assert np.allclose(np.sum(x, axis=1), np.ones(x.shape[0])), msg
    return True


def is_binary(x):
    """Return True if array `x` consists only of binary values"""
    msg = "Matrix must be binary"
    assert np.array_equal(x, x.astype(bool)), msg
    return True

def binary_acc(pred_y_prob, y_test):
    y_test = y_test.flatten().astype(int)
    class_indices = pred_y_prob.flatten()
    class_indices[class_indices < 0.5] = 0
    class_indices[class_indices >= 0.5] = 1
    correct_results_sum = (class_indices == y_test).sum().astype('float32')
    acc = correct_results_sum / y_test.shape[0]
    acc = np.round(acc * 100)
    return acc

def multiclass_acc(pred_y_prob, y_test):
    y_test = y_test.flatten().astype(int)
    class_indices = np.argmax(pred_y_prob, axis=1) # Use np.argmax to select the class with the highest probability for each sample
    correct_results_sum = (class_indices == y_test).sum().astype('float32')
    acc = correct_results_sum/y_test.shape[0]
    acc = np.round(acc * 100)
    return acc

def moving_average(x, window:np.ndarray):
    return np.convolve(x, window, mode='valid') / window.sum()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count