import numpy as np


def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total
    return accuracy

class LogisticRegression:
    def __init__(self, solver='saga', max_iter=1000, C=1.0, learning_rate=0.01, penalty="a"):
        self.solver = solver
        self.max_iter = max_iter
        self.C = C
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.penalty = penalty
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def initialize_parameters(self, num_features):
        self.weights = np.zeros(num_features)
        self.bias = 0
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.initialize_parameters(num_features)
        
        for _ in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)
            
            dw = (1/num_samples) * np.dot(X.T, (predictions - y))
            db = (1/num_samples) * np.sum(predictions - y)
            
            self.weights -= self.learning_rate * dw + (1 / (self.C * self.learning_rate)) * self.weights
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        return np.round(predictions)


