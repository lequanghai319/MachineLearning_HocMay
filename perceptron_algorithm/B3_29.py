import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights)
                y_pred = 1 if linear_output >= 0 else -1
                
                if y[idx] * y_pred <= 0:
                    self.weights += self.lr * y[idx] * x_i

    def predict(self, X):
        linear_output = np.dot(X, self.weights)
        return np.where(linear_output >= 0, 1, -1)
      
