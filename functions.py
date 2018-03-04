import numpy as np

def id(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x,-709,100000)))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

#def softmax(x):
#    c = np.max(x)
#    return np.exp(x-c) / np.sum(np.exp(x-c))

def tanh(x):
    x = x - np.max(x)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def relu(x):
    return np.maximum(1e-323, x)

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy(y, t):
    return (-1) * np.sum(t * np.log(y))

