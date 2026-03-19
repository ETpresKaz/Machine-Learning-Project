import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_deriv(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy(y_true, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1. - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_deriv(y_true, y_pred):
    # For softmax + cross-entropy, dL/dz = (softmax(z) - y)/N
    return (y_pred - y_true) / y_true.shape[0]