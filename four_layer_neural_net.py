import sys, os
sys.path.append(os.pardir)
from functions import *
from gradient import numerical_gradient

class FourLayerNet:
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size_1)
        self.params["b1"] = np.zeros(hidden_size_1)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size_1, hidden_size_2)
        self.params["b2"] = np.zeros(hidden_size_2)
        self.params["W3"] = weight_init_std * np.random.randn(hidden_size_2, output_size)
        self.params["b3"] = np.zeros(output_size)
        
    def predict(self, x0):
        W1, W2, W3 = self.params["W1"], self.params["W2"], self.params["W3"]
        b1, b2, b3 = self.params["b1"], self.params["b2"], self.params["b3"]
        
        a1 = np.dot(x0, W1) + b1
        x1 = sigmoid(a1)
        a2 = np.dot(x1, W2) + b2
        x2 = sigmoid(a2)
        a3 = np.dot(x2, W3) + b3
        y = tanh(a3)
#        y = softmax(a2)
        
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        return mean_squared_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])
        grads["W3"] = numerical_gradient(loss_W, self.params["W3"])
        grads["b3"] = numerical_gradient(loss_W, self.params["b3"])
        
        return grads