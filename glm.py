import numpy as np
from typing import Tuple, List

class PoissonGLM():

    def __init__(self, iterations=1000, learning_rate=1e-4):
        self.iterations = iterations
        self.learning_rate = learning_rate

    def link(self, y_pred: np.ndarray):
        return np.exp(y_pred)

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.float:
        return (y_pred - np.multiply(y_true.reshape(-1, 1), np.log(y_pred))).sum()

    def loss_derivative(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.float]:
        return ((np.multiply(X, (y_pred - y_true).reshape(-1, 1))).sum(axis=0),
                (y_pred - y_true).sum())

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.thetas = np.random.normal(size=(X.shape[1], ))
        self.bias = np.random.normal(size=1)
        self.losses: List = []
        for iteration in range(self.iterations):
            y_pred = self.predict(X)
            self.losses.append(self.loss(y, y_pred))
            loss_derivative = self.loss_derivative(X, y, y_pred)
            self.thetas = self.thetas - self.learning_rate * loss_derivative[0]
            self.bias = self.bias - self.learning_rate * loss_derivative[1]

    def predict(self, X):
        return self.link(np.dot(self.thetas, X.T) + self.bias)
